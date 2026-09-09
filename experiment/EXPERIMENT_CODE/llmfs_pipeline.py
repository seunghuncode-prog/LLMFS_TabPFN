"""Leakage-safe LLMFS × TabPFN experiment engine.

The notebook in the parent directory is the intended user interface.  This
module keeps the implementation importable, testable, and resumable.
"""

from __future__ import annotations

import gc
import getpass
import hashlib
import importlib.metadata as importlib_metadata
import itertools
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import median_abs_deviation, rankdata
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PIPELINE_VERSION = "4.1.1"
TERMINAL_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


@dataclass
class DatasetBundle:
    dataset: str
    X: pd.DataFrame
    y: pd.Series
    sample_ids: pd.Series
    registry: pd.DataFrame
    outer_folds: pd.DataFrame


_DATASET_CACHE: dict[tuple[str, str], DatasetBundle] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_int(*parts: Any, modulo: int = 2**32 - 1) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16) % modulo


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_slug(value: Any) -> str:
    text = str(value)
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["config_path"] = str(path)
    cfg["code_root"] = str(path.parent)
    for key in ("data_root", "results_root"):
        p = Path(cfg[key])
        if not p.is_absolute():
            p = (path.parent / p).resolve()
        cfg[key] = str(p)
    reuse_root = cfg.get("llm", {}).get("reuse_scores_root")
    if reuse_root:
        reuse_path = Path(reuse_root)
        if not reuse_path.is_absolute():
            reuse_path = (path.parent / reuse_path).resolve()
        cfg["llm"]["reuse_scores_root"] = str(reuse_path)
    checkpoint = Path(cfg["models"]["tabpfn"]["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (path.parent / checkpoint).resolve()
    cfg["models"]["tabpfn"]["checkpoint"] = str(checkpoint)
    return cfg


def init_result_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["results_root"])
    names = [
        "llm_jobs",
        "llm_raw",
        "llm_scores",
        "score_cache",
        "tuning_cache",
        "tuning",
        "selections",
        "predictions",
        "metrics",
        "embeddings",
        "figures",
        "logs",
        "splits",
    ]
    paths = {name: root / name for name in names}
    root.mkdir(parents=True, exist_ok=True)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    snapshot = dict(cfg)
    snapshot.pop("config_path", None)
    snapshot.pop("code_root", None)
    atomic_write_json(root / "resolved_experiment_config.json", snapshot)
    return paths


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def environment_report(cfg: dict[str, Any], require_cuda: bool | None = None) -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
        torch_cuda = torch.version.cuda
    except Exception as exc:  # pragma: no cover - diagnostic path
        cuda_available = False
        cuda_device = None
        torch_cuda = None
        torch_error = repr(exc)
    else:
        torch_error = None

    checkpoint = Path(cfg["models"]["tabpfn"]["checkpoint"])
    checkpoint_is_file = checkpoint.is_file()
    report = {
        "created_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "torch_cuda_runtime": torch_cuda,
        "torch_error": torch_error,
        "tabpfn_checkpoint": {
            "path": str(checkpoint),
            "exists": checkpoint.exists(),
            "is_file": checkpoint_is_file,
            "size_bytes": checkpoint.stat().st_size if checkpoint_is_file else None,
        },
        "packages": {
            name: package_version(name)
            for name in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "catboost",
                "tabpfn",
                "tabpfn-extensions",
                "google-genai",
                "torch",
                "umap-learn",
            ]
        },
        "config_hash": stable_hash(cfg),
    }
    paths = init_result_dirs(cfg)
    atomic_write_json(paths["logs"] / "environment.json", report)
    if require_cuda is None:
        require_cuda = bool(cfg["execution"].get("require_cuda_for_tabpfn", True))
    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA GPU는 보이지만 현재 Python의 PyTorch가 CUDA를 사용할 수 없습니다. "
            "CUDA-enabled PyTorch를 설치하고 커널을 재시작한 후 다시 확인하세요."
        )
    if require_cuda and not checkpoint_is_file:
        raise FileNotFoundError(f"TabPFN checkpoint file not found: {checkpoint}")
    return report


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_dataset(cfg: dict[str, Any], dataset: str, use_cache: bool = True) -> DatasetBundle:
    root = Path(cfg["data_root"])
    key = (str(root), dataset)
    if use_cache and key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    base = root / dataset
    X_raw = pd.read_csv(base / "data" / "X.csv", low_memory=False)
    y_df = pd.read_csv(base / "data" / "y.csv")
    registry = pd.read_csv(base / "metadata" / "feature_registry.csv", low_memory=False)
    folds = pd.read_csv(base / "splits" / "outer_folds.csv")

    x_ids = X_raw["sample_id"].astype(str)
    y_df["sample_id"] = y_df["sample_id"].astype(str)
    folds["sample_id"] = folds["sample_id"].astype(str)
    if x_ids.tolist() != y_df["sample_id"].tolist():
        raise ValueError(f"{dataset}: X/y sample order mismatch")
    features = X_raw.columns.drop("sample_id").tolist()
    if features != registry["feature_id"].astype(str).tolist():
        raise ValueError(f"{dataset}: feature registry does not align with X")

    X = X_raw.drop(columns="sample_id")
    for row in registry.itertuples(index=False):
        fid = str(row.feature_id)
        if str(row.data_type).lower() == "numeric":
            X[fid] = pd.to_numeric(X[fid], errors="coerce")
        else:
            X[fid] = X[fid].astype("string")
    bundle = DatasetBundle(
        dataset=dataset,
        X=X,
        y=y_df["target"].astype(int),
        sample_ids=x_ids,
        registry=registry,
        outer_folds=folds,
    )
    if use_cache:
        _DATASET_CACHE[key] = bundle
    return bundle


def validate_all_data(cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in cfg["datasets"]:
        b = load_dataset(cfg, dataset)
        class_counts = b.y.value_counts().to_dict()
        configured = configured_outer_assignments(cfg, b, repeat=1)
        rows.append(
            {
                "dataset": dataset,
                "n": len(b.X),
                "p": b.X.shape[1],
                "p_over_n": b.X.shape[1] / len(b.X),
                "class_0": class_counts.get(0, 0),
                "class_1": class_counts.get(1, 0),
                "source_outer_repeats": b.outer_folds["repeat"].nunique(),
                "source_outer_folds": b.outer_folds["test_fold"].nunique(),
                "outer_repeats": int(cfg["cv"]["outer_repeats"]),
                "outer_folds": configured["test_fold"].nunique(),
                "split_strategy": cfg["cv"].get("outer_split_strategy", "generated_stratified"),
                "smallest_test_fold": int(configured.groupby("test_fold").size().min()),
            }
        )
    return pd.DataFrame(rows)


def configured_outer_assignments(
    cfg: dict[str, Any], bundle: DatasetBundle, repeat: int
) -> pd.DataFrame:
    """Return and persist the exact configured stratified outer split.

    The source data contain a 10 x 5 split. This preset generates and persists
    one explicit stratified 5-fold split so every sample is tested exactly once.
    """
    n_splits = int(cfg["cv"]["outer_folds"])
    n_repeats = int(cfg["cv"]["outer_repeats"])
    if not 1 <= repeat <= n_repeats:
        raise ValueError(f"repeat must be in 1..{n_repeats}, got {repeat}")
    if int(np.bincount(bundle.y.to_numpy(dtype=int)).min()) < n_splits:
        raise ValueError(f"{bundle.dataset}: too few samples per class for {n_splits}-fold CV")
    paths = init_result_dirs(cfg)
    path = paths["splits"] / f"{bundle.dataset}__stratified_{n_splits}fold__repeat_{repeat:02d}.csv"
    if path.exists() and cfg["execution"].get("resume", True):
        split = pd.read_csv(path)
    else:
        base_seed = int(cfg["cv"].get("outer_split_seed", 0))
        split_seed = stable_int("outer_cv", base_seed, bundle.dataset, repeat, n_splits)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        assignment = np.zeros(len(bundle.y), dtype=int)
        for fold_index, (_, test_idx) in enumerate(
            cv.split(np.zeros(len(bundle.y)), bundle.y.to_numpy(dtype=int)), start=1
        ):
            assignment[test_idx] = fold_index
        split = pd.DataFrame(
            {
                "sample_id": bundle.sample_ids.astype(str),
                "repeat": repeat,
                "test_fold": assignment,
                "n_splits": n_splits,
                "split_seed": split_seed,
                "strategy": "StratifiedKFold(shuffle=True)",
            }
        )
        tmp = path.with_suffix(".csv.tmp")
        split.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(path)
    split["sample_id"] = split["sample_id"].astype(str)
    if len(split) != len(bundle.X) or set(split["sample_id"]) != set(bundle.sample_ids.astype(str)):
        raise ValueError(f"{bundle.dataset}: configured split/sample coverage mismatch")
    if set(split["test_fold"].astype(int)) != set(range(1, n_splits + 1)):
        raise ValueError(f"{bundle.dataset}: configured split does not contain folds 1..{n_splits}")
    return split


def get_outer_indices(
    cfg: dict[str, Any], bundle: DatasetBundle, repeat: int, test_fold: int
) -> tuple[np.ndarray, np.ndarray]:
    fold = configured_outer_assignments(cfg, bundle, repeat)
    fold_map = fold.set_index("sample_id")["test_fold"]
    assigned = bundle.sample_ids.astype(str).map(fold_map)
    if assigned.isna().any():
        raise ValueError(f"{bundle.dataset}: fold/sample alignment failure")
    if test_fold not in set(assigned.astype(int)):
        raise ValueError(f"{bundle.dataset}: unknown test fold {test_fold}")
    test_mask = assigned.to_numpy() == test_fold
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def stratified_prefix_subset(indices: np.ndarray, y: pd.Series, n: int, seed: int) -> np.ndarray:
    if n >= len(indices):
        return np.sort(indices.copy())
    y_arr = y.iloc[indices].to_numpy()
    classes, counts = np.unique(y_arr, return_counts=True)
    raw = n * counts / counts.sum()
    quotas = np.floor(raw).astype(int)
    quotas = np.maximum(quotas, 1)
    while quotas.sum() > n:
        candidates = np.flatnonzero(quotas > 1)
        quotas[candidates[np.argmax(quotas[candidates] - raw[candidates])]] -= 1
    while quotas.sum() < n:
        room = counts - quotas
        candidates = np.flatnonzero(room > 0)
        quotas[candidates[np.argmax(raw[candidates] - quotas[candidates])]] += 1
    chosen: list[int] = []
    for cls, quota in zip(classes, quotas, strict=True):
        cls_idx = indices[y_arr == cls].copy()
        rng = np.random.default_rng(stable_int(seed, int(cls)))
        rng.shuffle(cls_idx)
        chosen.extend(cls_idx[:quota].tolist())
    return np.array(sorted(chosen), dtype=int)


def training_subset(
    bundle: DatasetBundle,
    train_indices: np.ndarray,
    sample_regime: str | int,
    seed: int,
) -> np.ndarray:
    if str(sample_regime).lower() == "full":
        return train_indices
    return stratified_prefix_subset(train_indices, bundle.y, int(sample_regime), seed)


def feature_registry(cfg: dict[str, Any], dataset: str) -> pd.DataFrame:
    return load_dataset(cfg, dataset).registry.copy()


def semantic_mapping_report(cfg: dict[str, Any], raise_on_failure: bool = False) -> pd.DataFrame:
    """Validate that mandatory biomedical name mappings are present before Gemini use."""
    mapping_cfg = cfg.get("semantic_mapping", {})
    required = set(map(str, mapping_cfg.get("required_datasets", [])))
    thresholds = mapping_cfg.get("minimum_gene_symbol_coverage", {})
    real_name_column = str(mapping_cfg.get("real_name_column", "semantic_name"))
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset in cfg["datasets"]:
        registry = feature_registry(cfg, dataset)
        if real_name_column not in registry.columns:
            raise ValueError(f"{dataset}: missing real-name column {real_name_column!r}")
        names = registry[real_name_column].astype("string").fillna("").str.strip()
        symbols = (
            registry["gene_symbol"].astype("string").fillna("").str.strip()
            if "gene_symbol" in registry.columns
            else pd.Series("", index=registry.index, dtype="string")
        )
        n_features = int(len(registry))
        n_named = int(names.ne("").sum())
        n_symbols = int(symbols.ne("").sum())
        coverage = n_symbols / n_features if n_features else float("nan")
        minimum = float(thresholds.get(dataset, 0.0))
        passed = n_named == n_features and (dataset not in required or coverage >= minimum)
        rows.append(
            {
                "dataset": dataset,
                "n_features": n_features,
                "real_name_column": real_name_column,
                "n_nonempty_real_names": n_named,
                "n_gene_symbols": n_symbols,
                "gene_symbol_coverage": coverage,
                "required_minimum": minimum if dataset in required else 0.0,
                "mapping_required": dataset in required,
                "passed": bool(passed),
            }
        )
        if not passed:
            failures.append(
                f"{dataset}: names={n_named}/{n_features}, gene-symbol coverage={coverage:.3f}, "
                f"required={minimum:.3f}"
            )
    report = pd.DataFrame(rows)
    if raise_on_failure and failures:
        raise RuntimeError("Semantic mapping validation failed: " + "; ".join(failures))
    return report


def condition_map_path(cfg: dict[str, Any], dataset: str, condition: str) -> Path:
    base = Path(cfg["data_root"]) / dataset / "name_maps"
    if condition == "real":
        return base / "real.csv"
    if condition == "anonymous":
        return base / "anonymous.csv"
    if condition.startswith("permuted_seed_"):
        return base / f"{condition}.csv"
    raise ValueError(f"Unknown name condition: {condition}")


def displayed_names(cfg: dict[str, Any], dataset: str, condition: str) -> pd.DataFrame:
    registry = feature_registry(cfg, dataset)
    if condition == "real":
        semantic_mapping_report(cfg, raise_on_failure=True)
        real_name_column = str(cfg.get("semantic_mapping", {}).get("real_name_column", "semantic_name"))
        out = registry[["feature_id", real_name_column]].rename(columns={real_name_column: "display_name"})
        out["display_name"] = out["display_name"].astype("string").fillna("").str.strip()
        if out["display_name"].eq("").any():
            raise ValueError(f"{dataset}: empty mapped real feature name")
        return out
    if condition == "anonymous":
        return registry[["feature_id"]].assign(display_name=lambda d: d["feature_id"])
    name_map = pd.read_csv(condition_map_path(cfg, dataset, condition))
    source_names = displayed_names(cfg, dataset, "real").rename(
        columns={"feature_id": "semantic_source_feature_id"}
    )
    out = name_map.merge(source_names, on="semantic_source_feature_id", how="left", validate="one_to_one")
    if out["display_name"].isna().any():
        raise ValueError(f"{dataset}/{condition}: unresolved semantic source")
    return out[["feature_id", "display_name"]]


def llm_prompt(task: str, items: pd.DataFrame) -> str:
    compact_items = [[int(i), str(name)] for i, name in enumerate(items["display_name"], start=1)]
    return (
        "You are scoring feature-name priors for a binary biomedical classification task.\n\n"
        f"TASK: {task}\n\n"
        "You may use ONLY each displayed feature name and your pretrained knowledge. "
        "Do not use or infer feature values, distributions, sample size, dataset identity, "
        "neighboring items, external search, tools, or databases. Score every item independently.\n"
        "r relevance: 0 unrelated/technical nuisance; 1 weak/general; 2 plausible indirect; "
        "3 established association; 4 direct/canonical marker or determinant.\n"
        "c confidence that the displayed name is correctly recognized: 0 unknown; 1 low; 2 medium; 3 high.\n"
        "Return exactly one object for every input i. Do not add explanations.\n\n"
        f"ITEMS={json.dumps(compact_items, ensure_ascii=False, separators=(',', ':'))}"
    )


def make_llm_inline_requests(
    cfg: dict[str, Any], condition: str, run_index: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if condition not in {"real", "anonymous"}:
        raise ValueError("Only Real and Anonymous are sent to Gemini")
    llm_cfg = cfg["llm"]
    seed = llm_cfg["run_seeds"][min(run_index - 1, len(llm_cfg["run_seeds"]) - 1)]
    chunk_size = int(llm_cfg["chunk_size"])
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "i": {"type": "integer"},
                "r": {"type": "integer", "minimum": 0, "maximum": 4},
                "c": {"type": "integer", "minimum": 0, "maximum": 3},
            },
            "required": ["i", "r", "c"],
        },
    }
    for dataset, ds_cfg in cfg["datasets"].items():
        names = displayed_names(cfg, dataset, condition)
        rng = np.random.default_rng(stable_int("llm", condition, run_index, seed, dataset))
        order = rng.permutation(len(names))
        shuffled = names.iloc[order].reset_index(drop=True)
        for chunk_id, start in enumerate(range(0, len(shuffled), chunk_size), start=1):
            chunk = shuffled.iloc[start : start + chunk_size].reset_index(drop=True)
            key = f"{condition}_run{run_index}_{dataset}_chunk{chunk_id:03d}"
            requests.append(
                {
                    "contents": [{"parts": [{"text": llm_prompt(ds_cfg["task"], chunk)}], "role": "user"}],
                    "config": {
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                        "thinking_config": {"thinking_level": llm_cfg["thinking_level"]},
                    },
                }
            )
            manifest.append(
                {
                    "request_key": key,
                    "request_index": len(requests) - 1,
                    "dataset": dataset,
                    "condition": condition,
                    "run_index": run_index,
                    "run_seed": seed,
                    "chunk_id": chunk_id,
                    "feature_ids": chunk["feature_id"].astype(str).tolist(),
                    "display_names": chunk["display_name"].astype(str).tolist(),
                    "prompt_hash": stable_hash(llm_prompt(ds_cfg["task"], chunk)),
                }
            )
    return requests, manifest


def ensure_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        key = getpass.getpass("GEMINI_API_KEY (화면과 파일에 저장되지 않음): ").strip()
        if not key:
            raise RuntimeError("Gemini API key was not entered")
        os.environ["GEMINI_API_KEY"] = key
    return key


def submit_llm_batch(cfg: dict[str, Any], condition: str, run_index: int) -> dict[str, Any]:
    from google import genai

    ensure_gemini_api_key()
    paths = init_result_dirs(cfg)
    requests, manifest = make_llm_inline_requests(cfg, condition, run_index)
    job_dir = paths["llm_jobs"] / f"{condition}_run{run_index}"
    job_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(job_dir / "request_manifest.json", manifest)
    with genai.Client() as client:
        job = client.batches.create(
            model=cfg["llm"]["model"],
            src=requests,
            config={"display_name": f"llmfs-{condition}-run{run_index}"},
        )
    record = {
        "job_name": job.name,
        "condition": condition,
        "run_index": run_index,
        "model": cfg["llm"]["model"],
        "submitted_at": utc_now(),
        "n_requests": len(requests),
        "manifest_hash": stable_hash(manifest),
    }
    atomic_write_json(job_dir / "job.json", record)
    return record


def get_llm_job_status(cfg: dict[str, Any], condition: str, run_index: int) -> dict[str, Any]:
    from google import genai

    ensure_gemini_api_key()
    job_path = Path(cfg["results_root"]) / "llm_jobs" / f"{condition}_run{run_index}" / "job.json"
    record = json.loads(job_path.read_text(encoding="utf-8"))
    with genai.Client() as client:
        job = client.batches.get(name=record["job_name"])
    state = getattr(job.state, "name", str(job.state))
    return {"job_name": record["job_name"], "state": state, "terminal": state in TERMINAL_BATCH_STATES}


def wait_for_llm_job(
    cfg: dict[str, Any], condition: str, run_index: int, timeout_hours: float = 24.0
) -> dict[str, Any]:
    deadline = time.time() + timeout_hours * 3600
    while True:
        status = get_llm_job_status(cfg, condition, run_index)
        print(utc_now(), status)
        if status["terminal"]:
            return status
        if time.time() >= deadline:
            raise TimeoutError(status)
        time.sleep(int(cfg["llm"]["poll_seconds"]))


def retrieve_llm_batch(cfg: dict[str, Any], condition: str, run_index: int) -> pd.DataFrame:
    from google import genai

    ensure_gemini_api_key()
    paths = init_result_dirs(cfg)
    job_dir = paths["llm_jobs"] / f"{condition}_run{run_index}"
    job_record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    manifest = json.loads((job_dir / "request_manifest.json").read_text(encoding="utf-8"))
    with genai.Client() as client:
        job = client.batches.get(name=job_record["job_name"])
    state = getattr(job.state, "name", str(job.state))
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch is not successful: {state}")
    responses = list(job.dest.inlined_responses)
    if len(responses) != len(manifest):
        raise ValueError(f"Expected {len(manifest)} responses, got {len(responses)}")

    rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    for meta, inline in zip(manifest, responses, strict=True):
        if not inline.response:
            raise RuntimeError(f"Missing response for {meta['request_key']}: {inline.error}")
        text = inline.response.text
        raw_records.append({"request_key": meta["request_key"], "text": text})
        payload = json.loads(text)
        expected = len(meta["feature_ids"])
        if not isinstance(payload, list) or len(payload) != expected:
            raise ValueError(f"{meta['request_key']}: expected exactly {expected} response items")
        for item in payload:
            if not isinstance(item, dict) or set(item) != {"i", "r", "c"}:
                raise ValueError(f"{meta['request_key']}: each item must contain only i, r, and c")
        by_i = {int(item["i"]): item for item in payload}
        if set(by_i) != set(range(1, expected + 1)):
            raise ValueError(f"{meta['request_key']}: invalid or missing item indices")
        for i in range(1, expected + 1):
            item = by_i[i]
            r, c = int(item["r"]), int(item["c"])
            if not (0 <= r <= 4 and 0 <= c <= 3):
                raise ValueError(f"{meta['request_key']}/{i}: score outside rubric")
            rows.append(
                {
                    "dataset": meta["dataset"],
                    "condition": condition,
                    "run_index": run_index,
                    "run_seed": meta["run_seed"],
                    "feature_id": meta["feature_ids"][i - 1],
                    "display_name": meta["display_names"][i - 1],
                    "relevance": r,
                    "confidence": c,
                    "request_key": meta["request_key"],
                }
            )
    atomic_write_json(paths["llm_raw"] / f"{condition}_run{run_index}.json", raw_records)
    out = pd.DataFrame(rows)
    for dataset in cfg["datasets"]:
        expected_ids = set(feature_registry(cfg, dataset)["feature_id"].astype(str))
        actual_ids = set(out.loc[out["dataset"] == dataset, "feature_id"].astype(str))
        if actual_ids != expected_ids:
            raise ValueError(f"{dataset}/{condition}/run{run_index}: feature coverage mismatch")
    out_path = paths["llm_raw"] / f"{condition}_run{run_index}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def aggregate_llm_scores(cfg: dict[str, Any], condition: str) -> dict[str, pd.DataFrame]:
    paths = init_result_dirs(cfg)
    runs = int(cfg["llm"]["real_runs"] if condition == "real" else cfg["llm"]["anonymous_runs"])
    raw = [pd.read_csv(paths["llm_raw"] / f"{condition}_run{i}.csv") for i in range(1, runs + 1)]
    all_scores = pd.concat(raw, ignore_index=True)
    outputs: dict[str, pd.DataFrame] = {}
    for dataset in cfg["datasets"]:
        ds = all_scores.loc[all_scores["dataset"] == dataset].copy()
        pivot_r = ds.pivot(index="feature_id", columns="run_index", values="relevance").sort_index()
        pivot_c = ds.pivot(index="feature_id", columns="run_index", values="confidence").sort_index()
        r_scaled = pivot_r / 4.0
        relevance = r_scaled.median(axis=1)
        confidence = (pivot_c / 3.0).mean(axis=1)
        if runs >= 2:
            agreement = 1.0 - 2.0 * r_scaled.apply(
                lambda row: float(median_abs_deviation(row.to_numpy(), scale=1)), axis=1
            )
            agreement = agreement.clip(0.0, 1.0)
        else:
            agreement = pd.Series(1.0, index=relevance.index)
        reliability = confidence * agreement
        text_score = 0.5 + reliability * (relevance - 0.5)
        result = pd.DataFrame(
            {
                "feature_id": relevance.index,
                "relevance_scaled": relevance.to_numpy(),
                "confidence_scaled": confidence.to_numpy(),
                "agreement": agreement.to_numpy(),
                "reliability": reliability.to_numpy(),
                "text_score": text_score.to_numpy(),
                "condition": condition,
                "llm_runs": runs,
            }
        )
        out_dir = paths["llm_scores"] / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_dir / f"{condition}.csv", index=False, encoding="utf-8-sig")
        outputs[dataset] = result
    return outputs


def build_permuted_llm_scores(cfg: dict[str, Any]) -> list[Path]:
    paths = init_result_dirs(cfg)
    written: list[Path] = []
    for dataset in cfg["datasets"]:
        real = pd.read_csv(paths["llm_scores"] / dataset / "real.csv")
        source = real.rename(columns={c: f"source_{c}" for c in real.columns if c != "feature_id"})
        source = source.rename(columns={"feature_id": "semantic_source_feature_id"})
        for seed in cfg["feature_selection"]["permutation_seeds"]:
            condition = f"permuted_seed_{int(seed):04d}"
            mapping = pd.read_csv(condition_map_path(cfg, dataset, condition))
            out = mapping.merge(source, on="semantic_source_feature_id", how="left", validate="one_to_one")
            cols = {
                "source_relevance_scaled": "relevance_scaled",
                "source_confidence_scaled": "confidence_scaled",
                "source_agreement": "agreement",
                "source_reliability": "reliability",
                "source_text_score": "text_score",
                "source_llm_runs": "llm_runs",
            }
            out = out.rename(columns=cols)
            out["condition"] = condition
            keep = [
                "feature_id",
                "semantic_source_feature_id",
                "relevance_scaled",
                "confidence_scaled",
                "agreement",
                "reliability",
                "text_score",
                "condition",
                "llm_runs",
            ]
            out_path = paths["llm_scores"] / dataset / f"{condition}.csv"
            out[keep].to_csv(out_path, index=False, encoding="utf-8-sig")
            written.append(out_path)
    return written


def llm_score_table(cfg: dict[str, Any], dataset: str, condition: str) -> pd.DataFrame:
    score_root = Path(cfg.get("llm_scores_root", Path(cfg["results_root"]) / "llm_scores"))
    path = score_root / dataset / f"{condition}.csv"
    if not path.exists():
        raise FileNotFoundError(f"LLM score is missing: {path}")
    return pd.read_csv(path)


def percentile_midrank(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return np.array([0.5])
    return (rankdata(arr, method="average") - 1.0) / (len(arr) - 1.0)


def _numeric_auc_vector(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Column-wise binary AUC using average ranks."""
    if scores.ndim == 1:
        scores = scores[:, None]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], 0.5)
    ranks = rankdata(scores, axis=0, method="average")
    rank_sum = ranks[y == 1].sum(axis=0)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _numeric_matrix(X: pd.DataFrame, numeric_cols: list[str], train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = X[numeric_cols].to_numpy(dtype=float, na_value=np.nan)
    medians = np.nanmedian(matrix[train_rows], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return matrix, medians


def _smoothed_target_encode(
    train_values: pd.Series,
    train_y: np.ndarray,
    valid_values: pd.Series,
    smoothing: float = 5.0,
) -> np.ndarray:
    train_values = train_values.astype("string").fillna("__MISSING__")
    valid_values = valid_values.astype("string").fillna("__MISSING__")
    prior = float(np.mean(train_y))
    frame = pd.DataFrame({"x": train_values.to_numpy(), "y": train_y})
    stats = frame.groupby("x", dropna=False)["y"].agg(["mean", "count"])
    encoded = (stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)
    return valid_values.map(encoded).fillna(prior).to_numpy(dtype=float)


def cross_fitted_data_scores(
    X: pd.DataFrame,
    y: pd.Series,
    registry: pd.DataFrame,
    inner_folds: int = 3,
    inner_repeats: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    """Stable, fold-local univariate predictive score for Hybrid/Data-only."""
    y_arr = np.asarray(y, dtype=int)
    feature_ids = registry["feature_id"].astype(str).tolist()
    dtype_map = registry.set_index("feature_id")["data_type"].astype(str).str.lower()
    numeric_cols = [f for f in feature_ids if dtype_map.get(f, "numeric") == "numeric"]
    categorical_cols = [f for f in feature_ids if f not in numeric_cols]
    col_to_index = {f: i for i, f in enumerate(feature_ids)}
    split_scores: list[np.ndarray] = []
    cv = RepeatedStratifiedKFold(
        n_splits=min(inner_folds, int(np.bincount(y_arr).min())),
        n_repeats=inner_repeats,
        random_state=seed,
    )
    numeric_all = X[numeric_cols].to_numpy(dtype=float, na_value=np.nan) if numeric_cols else None
    for inner_train, inner_valid in cv.split(np.zeros(len(y_arr)), y_arr):
        scores = np.zeros(len(feature_ids), dtype=float)
        if numeric_cols:
            train = numeric_all[inner_train]
            valid = numeric_all[inner_valid]
            medians = np.nanmedian(train, axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            train_filled = np.where(np.isfinite(train), train, medians)
            valid_filled = np.where(np.isfinite(valid), valid, medians)
            med0 = np.median(train_filled[y_arr[inner_train] == 0], axis=0)
            med1 = np.median(train_filled[y_arr[inner_train] == 1], axis=0)
            direction = np.where(med1 >= med0, 1.0, -1.0)
            auc = _numeric_auc_vector(y_arr[inner_valid], valid_filled * direction)
            for f, a in zip(numeric_cols, auc, strict=True):
                scores[col_to_index[f]] = 2.0 * float(a) - 1.0
        for f in categorical_cols:
            pred = _smoothed_target_encode(
                X.iloc[inner_train][f], y_arr[inner_train], X.iloc[inner_valid][f]
            )
            auc = roc_auc_score(y_arr[inner_valid], pred) if np.unique(pred).size > 1 else 0.5
            scores[col_to_index[f]] = 2.0 * float(auc) - 1.0
        split_scores.append(scores)
    values = np.vstack(split_scores)
    stable = np.clip(values.mean(axis=0) - 0.5 * values.std(axis=0, ddof=1), 0.0, 1.0)
    return pd.DataFrame(
        {
            "feature_id": feature_ids,
            "data_score": stable,
            "data_rank": percentile_midrank(stable),
            "cv_mean_oriented_auc": (values.mean(axis=0) + 1.0) / 2.0,
            "cv_sd_oriented_effect": values.std(axis=0, ddof=1),
        }
    )


def _encode_for_mi(
    X: pd.DataFrame, registry: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = registry["feature_id"].astype(str).tolist()
    dtype_map = registry.set_index("feature_id")["data_type"].astype(str).str.lower()
    columns: list[np.ndarray] = []
    discrete: list[bool] = []
    for f in ids:
        if dtype_map.get(f, "numeric") == "numeric":
            arr = pd.to_numeric(X[f], errors="coerce").to_numpy(dtype=float)
            med = np.nanmedian(arr)
            arr = np.where(np.isfinite(arr), arr, med if np.isfinite(med) else 0.0)
            unique = np.unique(arr)
            is_discrete = unique.size <= max(10, int(np.sqrt(len(arr)))) and np.allclose(unique, np.round(unique))
            columns.append(arr)
            discrete.append(bool(is_discrete))
        else:
            codes, _ = pd.factorize(X[f].astype("string").fillna("__MISSING__"), sort=True)
            columns.append(codes.astype(float))
            discrete.append(True)
    return np.column_stack(columns), np.asarray(discrete, dtype=bool), ids


def mutual_information_scores(
    X: pd.DataFrame, y: pd.Series, registry: pd.DataFrame, seed: int
) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_classif

    matrix, discrete, ids = _encode_for_mi(X, registry)
    score = mutual_info_classif(
        matrix,
        np.asarray(y, dtype=int),
        discrete_features=discrete,
        n_neighbors=max(1, min(3, len(X) - 1)),
        random_state=seed,
    )
    return pd.DataFrame({"feature_id": ids, "mi_score": score, "mi_rank": percentile_midrank(score)})


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_supervised_preprocessor(registry: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    ids = registry["feature_id"].astype(str).tolist()
    dtype_map = registry.set_index("feature_id")["data_type"].astype(str).str.lower()
    numeric = [f for f in ids if dtype_map.get(f, "numeric") == "numeric"]
    categorical = [f for f in ids if f not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0), numeric, categorical


def sanitize_sklearn_frame(X: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Replace pandas nullable scalars with sklearn-safe float/object values."""
    out = X.copy()
    dtype_map = registry.set_index("feature_id")["data_type"].astype(str).str.lower()
    for feature in out.columns:
        if dtype_map.get(feature, "numeric") == "numeric":
            out[feature] = pd.to_numeric(out[feature], errors="coerce").astype(float)
        else:
            # pandas 3 nullable strings can retain pd.NA inside an object array.
            # sklearn's SimpleImputer compares object values with ``X != X``;
            # pd.NA makes that comparison raise "boolean value of NA is ambiguous".
            # Use an explicit category so both tuning and final estimators receive
            # ordinary Python strings only.
            out[feature] = (
                out[feature]
                .astype("string")
                .fillna("__MISSING__")
                .astype(object)
            )
    return out


def lasso_scores(
    X: pd.DataFrame,
    y: pd.Series,
    registry: pd.DataFrame,
    c_grid: Sequence[float],
    inner_folds: int,
    seed: int,
    max_iter: int = 5000,
) -> pd.DataFrame:
    X = sanitize_sklearn_frame(X, registry)
    pre, numeric, categorical = make_supervised_preprocessor(registry)
    model = LogisticRegression(
        l1_ratio=1.0,
        solver="saga",
        max_iter=max_iter,
        random_state=seed,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    min_class = int(np.bincount(np.asarray(y, dtype=int)).min())
    cv = StratifiedKFold(n_splits=min(inner_folds, min_class), shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipe,
        {"model__C": list(c_grid)},
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    fitted_pre: ColumnTransformer = search.best_estimator_.named_steps["pre"]
    coef = np.abs(search.best_estimator_.named_steps["model"].coef_[0])
    scores = {f: 0.0 for f in registry["feature_id"].astype(str)}
    cursor = 0
    for f in numeric:
        scores[f] = float(coef[cursor])
        cursor += 1
    if categorical:
        encoder = fitted_pre.named_transformers_["cat"].named_steps["onehot"]
        for f, categories in zip(categorical, encoder.categories_, strict=True):
            width = len(categories)
            scores[f] = float(np.linalg.norm(coef[cursor : cursor + width], ord=2))
            cursor += width
    if cursor != len(coef):
        raise RuntimeError("LASSO transformed feature mapping failed")
    out = pd.DataFrame({"feature_id": list(scores), "lasso_score": list(scores.values())})
    out["lasso_rank"] = percentile_midrank(out["lasso_score"])
    out["best_C"] = float(search.best_params_["model__C"])
    out["best_inner_auc"] = float(search.best_score_)
    return out


def prepare_catboost_frame(X: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    out = X.copy()
    dtype_map = registry.set_index("feature_id")["data_type"].astype(str).str.lower()
    categorical_indices: list[int] = []
    for i, f in enumerate(out.columns):
        if dtype_map.get(f, "numeric") != "numeric":
            out[f] = out[f].astype("string").fillna("__MISSING__").astype(str)
            categorical_indices.append(i)
        else:
            out[f] = pd.to_numeric(out[f], errors="coerce")
    return out, categorical_indices


def catboost_rfe_selection(
    X: pd.DataFrame,
    y: pd.Series,
    registry: pd.DataFrame,
    k: int,
    cfg: dict[str, Any],
    seed: int,
) -> list[str]:
    from catboost import CatBoostClassifier, EFeaturesSelectionAlgorithm, EShapCalcType, Pool

    X_cb, cat_indices = prepare_catboost_frame(X, registry)
    fs_cfg = cfg["feature_selection"]
    model = CatBoostClassifier(
        iterations=int(fs_cfg["catboost_rfe_iterations"]),
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        task_type=cfg["models"]["catboost"]["task_type"],
        devices=cfg["models"]["catboost"].get("devices", "0"),
        verbose=False,
        allow_writing_files=False,
    )
    pool = Pool(X_cb, label=y, cat_features=cat_indices, feature_names=X_cb.columns.tolist())
    result = model.select_features(
        pool,
        features_for_select=list(range(X_cb.shape[1])),
        num_features_to_select=min(k, X_cb.shape[1]),
        steps=min(int(fs_cfg["catboost_rfe_steps"]), max(1, X_cb.shape[1] - k)),
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=False,
        logging_level="Silent",
    )
    if "selected_features_names" in result:
        return [str(x) for x in result["selected_features_names"]]
    return [X_cb.columns[int(i)] for i in result["selected_features"]]


def deterministic_top_k(scores: pd.DataFrame, score_col: str, k: int, tie_seed: int) -> list[str]:
    frame = scores[["feature_id", score_col]].copy()
    frame["feature_id"] = frame["feature_id"].astype(str)
    frame["tie"] = frame["feature_id"].map(lambda f: stable_int("tie", tie_seed, f))
    frame = frame.sort_values([score_col, "tie"], ascending=[False, True], kind="mergesort")
    return frame.head(min(k, len(frame)))["feature_id"].tolist()


def adaptive_quota_selection(
    merged: pd.DataFrame,
    k: int,
    alpha: float,
    tie_seed: int,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    """Select a protected Text/Data quota and fill overlap by adaptive consensus.

    Data-score uncertainty raises the per-feature Text contribution, while LLM
    run agreement (``reliability``) prevents an unreliable semantic score from
    dominating.  The explicit quota avoids the destructive rank compromise
    observed with a single weighted-average ranking in the second experiment.
    """
    frame = merged.copy()
    q_text = percentile_midrank(frame["relevance_scaled"])
    q_data = frame["data_rank"].to_numpy(dtype=float)
    reliability = np.clip(frame["reliability"].to_numpy(dtype=float), 0.0, 1.0)
    uncertainty = percentile_midrank(frame["cv_sd_oriented_effect"].fillna(0.0))
    text_weight = np.clip(float(alpha) * reliability * uncertainty, 0.0, 1.0)
    frame["text_rank"] = q_text
    frame["data_rank_for_hybrid"] = q_data
    frame["data_uncertainty_rank"] = uncertainty
    frame["adaptive_text_weight"] = text_weight
    frame["text_priority"] = q_text * reliability * (0.5 + 0.5 * uncertainty)
    frame["adaptive_consensus"] = text_weight * q_text + (1.0 - text_weight) * q_data

    k = min(int(k), len(frame))
    k_text = int(np.clip(round(float(alpha) * k), 1, max(1, k - 1))) if k > 1 else k
    k_data = k - k_text
    text_ids = deterministic_top_k(frame, "text_priority", k_text, tie_seed)
    data_ids = deterministic_top_k(frame, "data_rank_for_hybrid", k_data, tie_seed)
    selected = list(dict.fromkeys([*text_ids, *data_ids]))
    if len(selected) < k:
        fillers = deterministic_top_k(frame, "adaptive_consensus", len(frame), tie_seed)
        selected.extend([feature_id for feature_id in fillers if feature_id not in selected][: k - len(selected)])
    meta = {
        "quota_text": int(k_text),
        "quota_data": int(k_data),
        "quota_overlap": int(k - len(set(text_ids) | set(data_ids))),
        "mean_adaptive_text_weight": float(np.mean(text_weight)),
    }
    return selected[:k], frame, meta


def score_cache_path(
    cfg: dict[str, Any], dataset: str, sample_regime: str | int, repeat: int, fold: int, method: str
) -> Path:
    name = f"{dataset}__n-{safe_slug(sample_regime)}__r{repeat:02d}__f{fold:02d}__{method}.csv"
    return Path(cfg["results_root"]) / "score_cache" / name


def get_or_compute_fold_score(
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    train_idx: np.ndarray,
    sample_regime: str | int,
    repeat: int,
    fold: int,
    method: str,
    seed: int,
) -> pd.DataFrame:
    path = score_cache_path(cfg, bundle.dataset, sample_regime, repeat, fold, method)
    if path.exists() and cfg["execution"].get("resume", True):
        return pd.read_csv(path)
    # A fold-level score must not depend on which selector happened to request it
    # first. This also guarantees that Hybrid and Data-only share the exact same
    # data evidence within a split.
    score_seed = stable_int(
        "fold_score",
        int(cfg.get("seeds", {}).get("data_score_seed", 20260814)),
        bundle.dataset,
        sample_regime,
        repeat,
        fold,
        method,
    )
    X_train = bundle.X.iloc[train_idx].reset_index(drop=True)
    y_train = bundle.y.iloc[train_idx].reset_index(drop=True)
    if method == "data":
        score = cross_fitted_data_scores(
            X_train,
            y_train,
            bundle.registry,
            inner_folds=int(cfg["cv"]["score_inner_folds"]),
            inner_repeats=int(cfg["cv"]["score_inner_repeats"]),
            seed=score_seed,
        )
    elif method == "mi":
        score = mutual_information_scores(X_train, y_train, bundle.registry, score_seed)
    elif method == "lasso":
        score = lasso_scores(
            X_train,
            y_train,
            bundle.registry,
            cfg["feature_selection"]["lasso_c_grid"],
            int(cfg["cv"]["tuning_inner_folds"]),
            score_seed,
            max_iter=int(cfg["models"]["logistic"]["max_iter"]),
        )
    else:
        raise ValueError(method)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.to_csv(path, index=False, encoding="utf-8-sig")
    return score


def selection_cache_path(cfg: dict[str, Any], row: dict[str, Any]) -> Path:
    def canonical_number(value: Any) -> Any:
        if value is None:
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        return int(number) if number.is_integer() else number

    fields = [
        row["dataset"],
        f"n-{row['sample_regime']}",
        f"r{int(row['repeat']):02d}",
        f"f{int(row['fold']):02d}",
        row["method"],
        row.get("condition", "name_invariant"),
        f"k-{canonical_number(row.get('k', 'all'))}",
        f"a-{canonical_number(row.get('alpha', 'na'))}",
        f"s-{row.get('selector_seed', 0)}",
    ]
    return Path(cfg["results_root"]) / "selections" / ("__".join(map(safe_slug, fields)) + ".json")


def _source_selection_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map analysis-only stages to the Core selection they must reuse."""
    if row.get("stage") in {"compatibility", "embedding"}:
        return {
            **row,
            "stage": "core",
            "model": "tabpfn",
            "k": float(row["k"]) if row.get("k") is not None else None,
        }
    return row


def _select_feature_ids_fixed(
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    train_idx: np.ndarray,
    row: dict[str, Any],
) -> list[str]:
    path = selection_cache_path(cfg, row)
    if path.exists() and cfg["execution"].get("resume", True):
        return json.loads(path.read_text(encoding="utf-8"))["feature_ids"]
    method = row["method"]
    k = bundle.X.shape[1] if method == "no_fs" else int(row["k"])
    seed = int(row.get("selector_seed", stable_int(row["repeat"], row["fold"], method)))
    condition = row.get("condition", "name_invariant")
    if method == "no_fs":
        selected = bundle.X.columns.astype(str).tolist()
    elif method == "random":
        rng = np.random.default_rng(seed)
        selected = rng.permutation(bundle.X.columns.to_numpy())[: min(k, bundle.X.shape[1])].tolist()
    elif method == "text":
        score = llm_score_table(cfg, bundle.dataset, condition)
        selected = deterministic_top_k(score, "text_score", k, seed)
    elif method in {"hybrid", "adaptive_hybrid", "data_only"}:
        data = get_or_compute_fold_score(
            cfg,
            bundle,
            train_idx,
            row["sample_regime"],
            int(row["repeat"]),
            int(row["fold"]),
            "data",
            seed,
        )
        if method == "data_only":
            selected = deterministic_top_k(data, "data_score", k, seed)
        else:
            text = llm_score_table(cfg, bundle.dataset, condition)
            merged = data.merge(text, on="feature_id", how="inner", validate="one_to_one")
            alpha = float(row.get("alpha", cfg["feature_selection"]["hybrid_alpha_primary"]))
            adaptive_meta: dict[str, Any] = {}
            if method == "adaptive_hybrid":
                selected, merged, adaptive_meta = adaptive_quota_selection(merged, k, alpha, seed)
            else:
                q_r = percentile_midrank(merged["relevance_scaled"])
                q_d = merged["data_rank"].to_numpy(dtype=float)
                reliability = merged["reliability"].to_numpy(dtype=float)
                weight = alpha * reliability
                merged["hybrid_score"] = weight * q_r + (1.0 - weight) * q_d
                selected = deterministic_top_k(merged, "hybrid_score", k, seed)
            if cfg["execution"].get("save_fold_scores", True):
                fold_score_path = score_cache_path(
                    cfg,
                    bundle.dataset,
                    row["sample_regime"],
                    int(row["repeat"]),
                    int(row["fold"]),
                    f"{method}_{condition}_a{alpha}",
                )
                merged.to_csv(fold_score_path, index=False, encoding="utf-8-sig")
    elif method == "mi":
        score = get_or_compute_fold_score(
            cfg, bundle, train_idx, row["sample_regime"], int(row["repeat"]), int(row["fold"]), "mi", seed
        )
        selected = deterministic_top_k(score, "mi_score", k, seed)
    elif method == "lasso":
        score = get_or_compute_fold_score(
            cfg,
            bundle,
            train_idx,
            row["sample_regime"],
            int(row["repeat"]),
            int(row["fold"]),
            "lasso",
            seed,
        )
        selected = deterministic_top_k(score, "lasso_score", k, seed)
    elif method == "catboost_rfe":
        selected = catboost_rfe_selection(
            bundle.X.iloc[train_idx].reset_index(drop=True),
            bundle.y.iloc[train_idx].reset_index(drop=True),
            bundle.registry,
            k,
            cfg,
            seed,
        )
    else:
        raise ValueError(f"Unknown FS method: {method}")
    record = {
        "created_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        **row,
        "n_train": len(train_idx),
        "n_selected": len(selected),
        "feature_ids": list(map(str, selected)),
        "train_sample_hash": stable_hash(bundle.sample_ids.iloc[train_idx].tolist()),
    }
    if method == "adaptive_hybrid":
        record.update(adaptive_meta)
    atomic_write_json(path, record)
    return list(map(str, selected))


def _nested_tuning_enabled(cfg: dict[str, Any], row: dict[str, Any]) -> bool:
    tuning = cfg.get("nested_tuning", {})
    return bool(tuning.get("enabled", False)) and row.get("stage") in {
        "core",
        "semantic",
        "compatibility",
        "embedding",
        "full",
    }


def _nested_candidate_configs(cfg: dict[str, Any], method: str) -> list[dict[str, Any]]:
    tuning = cfg["nested_tuning"]
    if method == "no_fs":
        return [{"k": None, "alpha": None}]
    k_values = [int(k) for k in tuning["k_grid"]]
    if method in {"hybrid", "adaptive_hybrid"}:
        return [
            {"k": k, "alpha": float(alpha)}
            for k, alpha in itertools.product(k_values, tuning["alpha_grid"])
        ]
    return [{"k": k, "alpha": None} for k in k_values]


def _tuning_logistic_probability(
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    selected: Sequence[str],
    seed: int,
) -> np.ndarray:
    registry = subset_registry(bundle.registry, selected)
    pre, _, _ = make_supervised_preprocessor(registry)
    train_frame = sanitize_sklearn_frame(
        bundle.X.iloc[train_idx][list(selected)].reset_index(drop=True), registry
    )
    valid_frame = sanitize_sklearn_frame(
        bundle.X.iloc[valid_idx][list(selected)].reset_index(drop=True), registry
    )
    tuning_cfg = cfg["nested_tuning"]
    model = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegression(
                    C=float(tuning_cfg.get("logistic_c", 1.0)),
                    l1_ratio=0.0,
                    solver="liblinear",
                    max_iter=int(cfg["models"]["logistic"]["max_iter"]),
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(train_frame, bundle.y.iloc[train_idx].reset_index(drop=True))
    prob = np.asarray(model.predict_proba(valid_frame))[:, 1]
    return np.clip(prob.astype(float), 0.0, 1.0)


def _nested_tune_and_select(
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    train_idx: np.ndarray,
    row: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    method = str(row["method"])
    if method == "no_fs":
        selected = bundle.X.columns.astype(str).tolist()
        return selected, {
            "selection_strategy": "no_fs",
            "selected_k": len(selected),
            "selected_alpha": None,
            "inner_auroc": None,
            "inner_log_loss": None,
            "candidate_metrics": [],
        }

    tuning_cfg = cfg["nested_tuning"]
    candidates = _nested_candidate_configs(cfg, method)
    y_outer = bundle.y.iloc[train_idx].to_numpy(dtype=int)
    split_seed = stable_int(
        "nested_inner",
        int(cfg.get("seeds", {}).get("tuning_seed", 20260817)),
        bundle.dataset,
        row["sample_regime"],
        row["repeat"],
        row["fold"],
    )
    strategy = str(tuning_cfg.get("strategy", "stratified_holdout"))
    if strategy == "stratified_holdout":
        local_indices = np.arange(len(train_idx), dtype=int)
        inner_train_local, inner_valid_local = train_test_split(
            local_indices,
            test_size=float(tuning_cfg.get("validation_fraction", 0.25)),
            random_state=split_seed,
            shuffle=True,
            stratify=y_outer,
        )
        splits = [(np.asarray(inner_train_local), np.asarray(inner_valid_local))]
    else:
        min_class = int(np.bincount(y_outer).min())
        n_splits = min(int(tuning_cfg["inner_folds"]), min_class)
        if n_splits < 2:
            raise ValueError(f"{bundle.dataset}/{row['sample_regime']}: inner CV needs two classes")
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        splits = list(inner_cv.split(np.zeros(len(train_idx)), y_outer))
    internal_cfg = deepcopy(cfg)
    internal_cfg["nested_tuning"]["enabled"] = False
    internal_cfg["llm_scores_root"] = str(Path(cfg["results_root"]) / "llm_scores")
    internal_cfg["results_root"] = str(Path(cfg["results_root"]) / "tuning_cache")
    candidate_rows: list[dict[str, Any]] = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        pooled_y: list[int] = []
        pooled_prob: list[float] = []
        selected_sizes: list[int] = []
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(splits, start=1):
            inner_train_idx = train_idx[np.asarray(inner_train_local, dtype=int)]
            inner_valid_idx = train_idx[np.asarray(inner_valid_local, dtype=int)]
            internal_row = {
                **row,
                "stage": "tuning_internal",
                "sample_regime": (
                    f"{row['sample_regime']}__outer-r{int(row['repeat']):02d}"
                    f"f{int(row['fold']):02d}__inner-f{inner_fold:02d}"
                ),
                "repeat": int(row["repeat"]),
                "fold": int(inner_fold),
                "k": int(candidate["k"]),
                "alpha": (
                    float(candidate["alpha"])
                    if candidate["alpha"] is not None
                    else float(cfg["feature_selection"]["hybrid_alpha_primary"])
                ),
            }
            selected_inner = _select_feature_ids_fixed(
                internal_cfg, bundle, inner_train_idx, internal_row
            )
            model_seed = stable_int(
                "nested_logistic",
                int(cfg.get("seeds", {}).get("model_seed", 20260813)),
                row["repeat"],
                row["fold"],
                inner_fold,
            )
            prob = _tuning_logistic_probability(
                cfg, bundle, inner_train_idx, inner_valid_idx, selected_inner, model_seed
            )
            pooled_y.extend(bundle.y.iloc[inner_valid_idx].astype(int).tolist())
            pooled_prob.extend(prob.tolist())
            selected_sizes.append(len(selected_inner))
        y_arr = np.asarray(pooled_y, dtype=int)
        p_arr = np.asarray(pooled_prob, dtype=float)
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "k": int(candidate["k"]),
                "alpha": float(candidate["alpha"]) if candidate["alpha"] is not None else None,
                "inner_auroc": float(roc_auc_score(y_arr, p_arr)),
                "inner_log_loss": float(
                    log_loss(y_arr, np.column_stack([1.0 - p_arr, p_arr]), labels=[0, 1])
                ),
                "mean_selected": float(np.mean(selected_sizes)),
                "inner_splits": len(splits),
                "inner_samples": len(y_arr),
            }
        )

    candidates_frame = pd.DataFrame(candidate_rows)
    best_auc = float(candidates_frame["inner_auroc"].max())
    tolerance = float(tuning_cfg.get("auroc_tolerance", 0.005))
    eligible = candidates_frame.loc[
        candidates_frame["inner_auroc"] >= best_auc - tolerance
    ].copy()
    eligible["alpha_distance"] = (
        eligible["alpha"].fillna(0.5) - float(cfg["feature_selection"]["hybrid_alpha_primary"])
    ).abs()
    winner = eligible.sort_values(
        ["inner_log_loss", "k", "alpha_distance", "candidate_index"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).iloc[0]
    selected_k = int(winner["k"])
    selected_alpha = (
        float(winner["alpha"])
        if method in {"hybrid", "adaptive_hybrid"} and pd.notna(winner["alpha"])
        else None
    )
    final_row = {
        **row,
        "stage": "tuning_final",
        "sample_regime": (
            f"{row['sample_regime']}__outer-r{int(row['repeat']):02d}f{int(row['fold']):02d}__final"
        ),
        "k": selected_k,
        "alpha": (
            selected_alpha
            if selected_alpha is not None
            else float(cfg["feature_selection"]["hybrid_alpha_primary"])
        ),
    }
    selected = _select_feature_ids_fixed(internal_cfg, bundle, train_idx, final_row)
    decision = {
        "selection_strategy": "nested_logistic",
        "tuning_evaluator": "logistic",
        "tuning_strategy": strategy,
        "selection_rule": "AUROC within tolerance, then log-loss, smaller k, alpha near 0.5",
        "selected_k": selected_k,
        "selected_alpha": selected_alpha,
        "inner_auroc": float(winner["inner_auroc"]),
        "inner_log_loss": float(winner["inner_log_loss"]),
        "inner_best_auroc": best_auc,
        "auroc_tolerance": tolerance,
        "inner_split_seed": int(split_seed),
        "candidate_metrics": candidate_rows,
    }
    return selected, decision


def select_feature_ids(
    cfg: dict[str, Any],
    bundle: DatasetBundle,
    train_idx: np.ndarray,
    row: dict[str, Any],
) -> list[str]:
    source_row = _source_selection_row(row)
    if source_row is not row:
        source_path = selection_cache_path(cfg, source_row)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Core feature selection must exist before {row['stage']}: {source_path}"
            )
        return json.loads(source_path.read_text(encoding="utf-8"))["feature_ids"]
    if not _nested_tuning_enabled(cfg, row):
        return _select_feature_ids_fixed(cfg, bundle, train_idx, row)
    path = selection_cache_path(cfg, row)
    if path.exists() and cfg["execution"].get("resume", True):
        return json.loads(path.read_text(encoding="utf-8"))["feature_ids"]
    selected, decision = _nested_tune_and_select(cfg, bundle, train_idx, row)
    record = {
        "created_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        **row,
        **decision,
        "n_train": len(train_idx),
        "n_selected": len(selected),
        "feature_ids": list(map(str, selected)),
        "train_sample_hash": stable_hash(bundle.sample_ids.iloc[train_idx].tolist()),
    }
    atomic_write_json(path, record)
    return list(map(str, selected))


def consolidate_tuning_decisions(cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((Path(cfg["results_root"]) / "selections").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if path.resolve() != selection_cache_path(cfg, record).resolve():
            continue
        if record.get("selection_strategy") != "nested_logistic":
            continue
        rows.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"feature_ids", "candidate_metrics"}
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(
            Path(cfg["results_root"]) / "metrics" / "nested_tuning_decisions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return out


def subset_registry(registry: pd.DataFrame, feature_ids: Sequence[str]) -> pd.DataFrame:
    lookup = registry.set_index("feature_id", drop=False)
    return lookup.loc[list(feature_ids)].reset_index(drop=True)


def prepare_tabular_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    train, test = X_train.copy(), X_test.copy()
    dtype_map = registry.set_index("feature_id")["data_type"].astype(str).str.lower()
    categorical_indices: list[int] = []
    for i, f in enumerate(train.columns):
        if dtype_map.get(f, "numeric") != "numeric":
            train[f] = train[f].astype("string").fillna("__MISSING__")
            test[f] = test[f].astype("string").fillna("__MISSING__")
            categorical_indices.append(i)
        else:
            train[f] = pd.to_numeric(train[f], errors="coerce")
            test[f] = pd.to_numeric(test[f], errors="coerce")
    return train, test, categorical_indices


def make_tabpfn_classifier(
    cfg: dict[str, Any], categorical_indices: Sequence[int], seed: int, no_fs: bool = False
):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    tcfg = cfg["models"]["tabpfn"]
    checkpoint = Path(tcfg["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TabPFN checkpoint file not found: {checkpoint}")
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5,
        model_path=checkpoint,
        n_estimators=int(tcfg["n_estimators"]),
        categorical_features_indices=list(categorical_indices),
        device=tcfg["device"],
        fit_mode=tcfg.get("fit_mode", "fit_preprocessors"),
        ignore_pretraining_limits=bool(no_fs and tcfg.get("ignore_pretraining_limits_for_no_fs", True)),
        random_state=seed,
        eval_metric=None,
        tuning_config=None,
    )


def _gpu_memory_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _gpu_memory_peak_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024**2)
    except Exception:
        pass
    return None


def predict_with_model(
    cfg: dict[str, Any],
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    registry: pd.DataFrame,
    seed: int,
    no_fs: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    _gpu_memory_reset()
    train, test, cat_indices = prepare_tabular_frames(X_train, X_test, registry)
    if model_name == "tabpfn":
        if cfg["execution"].get("require_cuda_for_tabpfn", True):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("TabPFN GPU run requested but torch.cuda.is_available() is False")
        model = make_tabpfn_classifier(cfg, cat_indices, seed, no_fs=no_fs)
        model.fit(train, y_train)
        prob = np.asarray(model.predict_proba(test))[:, 1]
    elif model_name == "catboost":
        from catboost import CatBoostClassifier

        train_cb, cat_indices = prepare_catboost_frame(train, registry)
        test_cb, _ = prepare_catboost_frame(test, registry)
        ccfg = cfg["models"]["catboost"]
        model = CatBoostClassifier(
            iterations=int(ccfg["iterations"]),
            depth=int(ccfg["depth"]),
            learning_rate=float(ccfg["learning_rate"]),
            l2_leaf_reg=float(ccfg["l2_leaf_reg"]),
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            task_type=ccfg["task_type"],
            devices=ccfg.get("devices", "0"),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(train_cb, y_train, cat_features=cat_indices)
        prob = np.asarray(model.predict_proba(test_cb))[:, 1]
    elif model_name == "logistic":
        train = sanitize_sklearn_frame(train, registry)
        test = sanitize_sklearn_frame(test, registry)
        pre, _, _ = make_supervised_preprocessor(registry)
        lcfg = cfg["models"]["logistic"]
        pipe = Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    LogisticRegression(
                        l1_ratio=0.0,
                        solver="liblinear",
                        max_iter=int(lcfg["max_iter"]),
                        random_state=seed,
                    ),
                ),
            ]
        )
        min_class = int(np.bincount(np.asarray(y_train, dtype=int)).min())
        cv = StratifiedKFold(
            n_splits=min(int(cfg["cv"]["tuning_inner_folds"]), min_class),
            shuffle=True,
            random_state=seed,
        )
        model = GridSearchCV(
            pipe,
            {"model__C": lcfg["c_grid"]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        model.fit(train, y_train)
        prob = np.asarray(model.predict_proba(test))[:, 1]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    elapsed = time.perf_counter() - start
    meta = {
        "fit_predict_seconds": elapsed,
        "gpu_peak_memory_mb": _gpu_memory_peak_mb(),
        "n_categorical": len(cat_indices),
    }
    del model
    gc.collect()
    return np.clip(prob.astype(float), 0.0, 1.0), meta


def prediction_path(cfg: dict[str, Any], row: dict[str, Any]) -> Path:
    identity = {
        key: row.get(key)
        for key in [
            "stage",
            "dataset",
            "sample_regime",
            "repeat",
            "fold",
            "method",
            "condition",
            "k",
            "alpha",
            "selector_seed",
            "model",
        ]
    }
    prefix = "__".join(
        [
            safe_slug(row["dataset"]),
            safe_slug(row["method"]),
            safe_slug(row.get("condition", "name_invariant")),
            safe_slug(row["model"]),
            f"r{int(row['repeat']):02d}f{int(row['fold']):02d}",
        ]
    )
    return Path(cfg["results_root"]) / "predictions" / f"{prefix}__{stable_hash(identity)[:16]}.csv"


def run_experiment_row(cfg: dict[str, Any], row: dict[str, Any]) -> Path:
    out_path = prediction_path(cfg, row)
    if out_path.exists() and cfg["execution"].get("resume", True):
        return out_path
    paths = init_result_dirs(cfg)
    bundle = load_dataset(cfg, row["dataset"])
    outer_train, outer_test = get_outer_indices(cfg, bundle, int(row["repeat"]), int(row["fold"]))
    subset_seed = stable_int(
        "sample",
        int(cfg.get("seeds", {}).get("subset_seed", 20260813)),
        row["dataset"],
        row["repeat"],
        row["fold"],
    )
    train_idx = training_subset(bundle, outer_train, row["sample_regime"], subset_seed)
    selected = select_feature_ids(cfg, bundle, train_idx, row)
    selected_registry = subset_registry(bundle.registry, selected)
    selection_source_path = selection_cache_path(cfg, _source_selection_row(row))
    selection_record = json.loads(selection_source_path.read_text(encoding="utf-8"))
    # Every method in the same repeat/fold receives the same paired model seed.
    # The seed changes across repeat/fold pairs so conclusions do not depend on
    # a single TabPFN random realization.
    model_seed = stable_int(
        "paired_model",
        int(cfg.get("seeds", {}).get("model_seed", 20260813)),
        row["repeat"],
        row["fold"],
    )
    seed_everything(model_seed)
    prob, timing = predict_with_model(
        cfg,
        row["model"],
        bundle.X.iloc[train_idx][selected].reset_index(drop=True),
        bundle.y.iloc[train_idx].reset_index(drop=True),
        bundle.X.iloc[outer_test][selected].reset_index(drop=True),
        selected_registry,
        model_seed,
        no_fs=row["method"] == "no_fs",
    )
    pred = pd.DataFrame(
        {
            "sample_id": bundle.sample_ids.iloc[outer_test].to_numpy(),
            "y_true": bundle.y.iloc[outer_test].to_numpy(dtype=int),
            "y_prob": prob,
            "y_pred": (prob >= 0.5).astype(int),
        }
    )
    metadata = {
        **row,
        "n_train": len(train_idx),
        "n_test": len(outer_test),
        "n_selected": len(selected),
        "selected_k": selection_record.get("selected_k", len(selected)),
        "selected_alpha": selection_record.get("selected_alpha"),
        "inner_auroc": selection_record.get("inner_auroc"),
        "inner_log_loss": selection_record.get("inner_log_loss"),
        "selection_strategy": selection_record.get("selection_strategy", "fixed"),
        "model_seed_used": int(model_seed),
        "selection_path": str(selection_source_path),
        "pipeline_version": PIPELINE_VERSION,
        "created_at": utc_now(),
        **timing,
    }
    for key, value in metadata.items():
        pred[key] = value
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False, encoding="utf-8-sig")
    append_jsonl(paths["logs"] / "completed_runs.jsonl", {"path": str(out_path), **metadata})
    return out_path


def all_name_conditions(cfg: dict[str, Any]) -> list[str]:
    return ["real", "anonymous"] + [
        f"permuted_seed_{int(seed):04d}" for seed in cfg["feature_selection"]["permutation_seeds"]
    ]


def _grid_conditions(method: str, stage: str, cfg: dict[str, Any]) -> list[str]:
    if method not in {"text", "hybrid", "adaptive_hybrid"}:
        return ["name_invariant"]
    if stage == "semantic" or stage == "full":
        conditions = all_name_conditions(cfg)
        if stage == "semantic" and cfg.get("stage_scope", {}).get(
            "reuse_core_real_for_semantic", True
        ):
            conditions = [condition for condition in conditions if condition != "real"]
        return conditions
    return ["real"]


def _resolve_stage_sample_regimes(stage: str, ds_cfg: dict[str, Any], cfg: dict[str, Any]) -> list[str | int]:
    available = list(ds_cfg["sample_regimes"])
    scope = cfg.get("stage_scope", {})
    requested = scope.get(f"{stage}_sample_regimes")
    if not requested:
        return available
    minimum = next((x for x in reversed(available) if str(x).lower() != "full"), None)
    resolved: list[str | int] = []
    for value in requested:
        actual = minimum if str(value).lower() == "minimum" else value
        if actual is not None and actual in available and actual not in resolved:
            resolved.append(actual)
    if not resolved:
        raise ValueError(f"No sample regimes resolved for stage={stage}; available={available}")
    return resolved


def build_experiment_grid(
    cfg: dict[str, Any],
    stage: str,
    *,
    semantic_all_k: bool = False,
    hybrid_all_alpha: bool = False,
    max_repeats: int | None = None,
    max_folds: int | None = None,
) -> pd.DataFrame:
    if stage not in {"smoke", "core", "semantic", "compatibility", "full"}:
        raise ValueError(stage)
    if stage == "smoke":
        return pd.DataFrame(
            [
                {
                    "stage": "smoke",
                    "dataset": "colon",
                    "sample_regime": "full",
                    "repeat": 1,
                    "fold": 1,
                    "method": method,
                    "condition": "name_invariant",
                    "k": 8,
                    "alpha": 0.5,
                    "selector_seed": 7,
                    "model": "logistic",
                }
                for method in ["mi", "random"]
            ]
        )
    if stage == "core":
        methods = cfg["feature_selection"]["methods"]
        models = [cfg["models"]["primary"]]
    elif stage == "semantic":
        methods = cfg.get("stage_scope", {}).get(
            "semantic_methods", ["text", "hybrid", "adaptive_hybrid"]
        )
        models = [cfg["models"]["primary"]]
    elif stage == "compatibility":
        methods = cfg.get("stage_scope", {}).get(
            "compatibility_methods",
            ["text", "hybrid", "adaptive_hybrid", "data_only", "mi", "random"],
        )
        # TabPFN predictions at the fixed primary k are reused from the core
        # stage; only the comparison classifiers need additional fits.
        models = list(cfg["models"]["compatibility"])
    else:
        methods = cfg["feature_selection"]["methods"]
        models = cfg["models"]["compatibility"]

    repeats = range(1, min(max_repeats or cfg["cv"]["outer_repeats"], cfg["cv"]["outer_repeats"]) + 1)
    folds = range(1, min(max_folds or cfg["cv"]["outer_folds"], cfg["cv"]["outer_folds"]) + 1)
    rows: list[dict[str, Any]] = []
    for dataset, ds_cfg in cfg["datasets"].items():
        sample_regimes = _resolve_stage_sample_regimes(stage, ds_cfg, cfg)
        for sample_regime, repeat, fold, method, model in itertools.product(
            sample_regimes, repeats, folds, methods, models
        ):
            conditions = _grid_conditions(method, stage, cfg)
            if stage == "compatibility" and method in {"text", "hybrid", "adaptive_hybrid"}:
                conditions = ["real"]
            if method == "no_fs":
                k_values: list[int | None] = [None]
            elif stage == "compatibility":
                k_values = [int(cfg["feature_selection"]["primary_k"])]
            elif stage == "semantic" and not semantic_all_k:
                k_values = [int(cfg["feature_selection"]["primary_k"])]
            else:
                k_values = [int(k) for k in cfg["feature_selection"]["k_values"]]
            selector_seeds = (
                cfg["feature_selection"]["random_seeds"]
                if method == "random"
                else [int(cfg.get("seeds", {}).get("selector_tie_seed", 20260814))]
            )
            alpha_values = (
                [float(a) for a in cfg["feature_selection"]["hybrid_alpha_grid"]]
                if method in {"hybrid", "adaptive_hybrid"} and hybrid_all_alpha
                else [float(cfg["feature_selection"]["hybrid_alpha_primary"])]
            )
            for condition, k, selector_seed, alpha in itertools.product(
                conditions, k_values, selector_seeds, alpha_values
            ):
                rows.append(
                    {
                        "stage": stage,
                        "dataset": dataset,
                        "sample_regime": sample_regime,
                        "repeat": repeat,
                        "fold": fold,
                        "method": method,
                        "condition": condition,
                        "k": k,
                        "alpha": alpha,
                        "selector_seed": int(selector_seed),
                        "model": model,
                    }
                )
    return pd.DataFrame(rows)


def grid_readiness(cfg: dict[str, Any], grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, condition in grid.loc[
        grid["method"].isin(["text", "hybrid", "adaptive_hybrid"]),
        ["dataset", "condition"],
    ].drop_duplicates().itertuples(index=False):
        path = Path(cfg["results_root"]) / "llm_scores" / dataset / f"{condition}.csv"
        rows.append({"dataset": dataset, "condition": condition, "score_exists": path.exists(), "path": str(path)})
    return pd.DataFrame(rows)


def build_hybrid_alpha_grid(cfg: dict[str, Any]) -> pd.DataFrame:
    grid = build_experiment_grid(cfg, "core", hybrid_all_alpha=True)
    primary = float(cfg["feature_selection"]["hybrid_alpha_primary"])
    grid = grid.loc[(grid["method"] == "hybrid") & (grid["alpha"] != primary)].copy()
    allowed: set[str] = set()
    for dataset, ds_cfg in cfg["datasets"].items():
        regimes = _resolve_stage_sample_regimes("hybrid_alpha", ds_cfg, cfg)
        for regime in regimes:
            allowed.add(f"{dataset}|{regime}")
    grid = grid.loc[
        grid.apply(lambda row: f"{row['dataset']}|{row['sample_regime']}" in allowed, axis=1)
    ].reset_index(drop=True)
    return grid


def build_hybrid_k_grid(cfg: dict[str, Any]) -> pd.DataFrame:
    """Additional Hybrid-only k sensitivity rows; primary k is reused from Core."""
    base = build_experiment_grid(cfg, "core")
    base = base.loc[
        (base["method"] == "hybrid")
        & (base["condition"] == "real")
        & (base["alpha"] == float(cfg["feature_selection"]["hybrid_alpha_primary"]))
    ].copy()
    requested = cfg.get("stage_scope", {}).get("hybrid_k_sample_regimes", ["full"])
    allowed: set[str] = set()
    for dataset, ds_cfg in cfg["datasets"].items():
        regimes = _resolve_stage_sample_regimes("hybrid_k", ds_cfg, cfg)
        for regime in regimes:
            allowed.add(f"{dataset}|{regime}")
    base = base.loc[
        base.apply(lambda row: f"{row['dataset']}|{row['sample_regime']}" in allowed, axis=1)
    ].copy()
    rows: list[pd.DataFrame] = []
    primary_k = int(cfg["feature_selection"]["primary_k"])
    for k in cfg["feature_selection"].get("hybrid_k_sensitivity", []):
        k = int(k)
        if k == primary_k:
            continue
        part = base.copy()
        part["k"] = k
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else base.head(0).copy()


def experiment_execution_plan(cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available = {
        "core": build_experiment_grid(cfg, "core"),
        "semantic": build_experiment_grid(cfg, "semantic"),
        "compatibility": build_experiment_grid(cfg, "compatibility"),
        "embedding": build_embedding_grid(cfg),
    }
    enabled = set(cfg.get("execution", {}).get("enabled_stages", available))
    grids = {name: grid for name, grid in available.items() if name in enabled}
    for stage, grid in grids.items():
        if stage == "embedding":
            rows.append({"stage": stage, "model": "tabpfn_embedding", "model_runs": len(grid)})
        else:
            for model, count in grid["model"].value_counts().items():
                rows.append({"stage": stage, "model": model, "model_runs": int(count)})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame([{"stage": "TOTAL", "model": "all", "model_runs": 0}])
    out.loc[len(out)] = {"stage": "TOTAL", "model": "all", "model_runs": int(out["model_runs"].sum())}
    return out


def expected_execution_counts(cfg: dict[str, Any]) -> dict[str, int]:
    available = {
        "core": build_experiment_grid(cfg, "core"),
        "semantic": build_experiment_grid(cfg, "semantic"),
        "compatibility": build_experiment_grid(cfg, "compatibility"),
        "embedding": build_embedding_grid(cfg),
    }
    enabled = set(cfg.get("execution", {}).get("enabled_stages", available))
    grids = {name: grid for name, grid in available.items() if name in enabled}
    return {
        **{name: int(len(grid)) for name, grid in grids.items()},
        "outer_total": int(sum(len(grid) for grid in grids.values())),
        "smoke": int(len(build_experiment_grid(cfg, "smoke"))),
    }


def experiment_completion_report(cfg: dict[str, Any]) -> pd.DataFrame:
    expected = expected_execution_counts(cfg)
    rows: list[dict[str, Any]] = []
    enabled = set(cfg.get("execution", {}).get("enabled_stages", ["core", "semantic", "compatibility", "embedding"]))
    for stage in ["smoke", *[name for name in ["core", "semantic", "compatibility"] if name in enabled]]:
        grid = build_experiment_grid(cfg, stage)
        paths = [prediction_path(cfg, row.where(pd.notna(row), None).to_dict()) for _, row in grid.iterrows()]
        completed = sum(path.is_file() for path in paths)
        rows.append(
            {
                "stage": stage,
                "expected": len(paths),
                "completed": completed,
                "remaining": len(paths) - completed,
                "complete": completed == len(paths),
            }
        )
    if "embedding" in enabled:
        embedding_grid = build_embedding_grid(cfg)
        embedding_completed = 0
        for _, row_series in embedding_grid.iterrows():
            row = row_series.where(pd.notna(row_series), None).to_dict()
            identity = {
                key: row.get(key)
                for key in ["dataset", "sample_regime", "repeat", "fold", "method", "condition", "k"]
            }
            path = Path(cfg["results_root"]) / "embeddings" / f"embedding_{stable_hash(identity)[:16]}.json"
            embedding_completed += int(path.is_file())
        rows.append(
            {
                "stage": "embedding",
                "expected": expected["embedding"],
                "completed": embedding_completed,
                "remaining": expected["embedding"] - embedding_completed,
                "complete": embedding_completed == expected["embedding"],
            }
        )
    out = pd.DataFrame(rows)
    total_expected = int(out["expected"].sum())
    total_completed = int(out["completed"].sum())
    out.loc[len(out)] = {
        "stage": "TOTAL",
        "expected": total_expected,
        "completed": total_completed,
        "remaining": total_expected - total_completed,
        "complete": total_expected == total_completed,
    }
    return out


def run_grid(cfg: dict[str, Any], grid: pd.DataFrame, max_runs: int | None = None) -> pd.DataFrame:
    paths = init_result_dirs(cfg)
    if max_runs is not None:
        grid = grid.head(max_runs)
    status_rows: list[dict[str, Any]] = []
    for i, row_series in grid.reset_index(drop=True).iterrows():
        row = row_series.where(pd.notna(row_series), None).to_dict()
        cached_path = prediction_path(cfg, row)
        resume_hit = bool(cfg["execution"].get("resume", True) and cached_path.is_file())
        cache_label = " [saved fold reused]" if resume_hit else ""
        print(
            f"[{i + 1}/{len(grid)}] {row['dataset']} {row['method']} "
            f"{row['model']} r{row['repeat']}f{row['fold']}{cache_label}"
        )
        started = time.perf_counter()
        try:
            # Bypass dataset loading, feature selection and model fitting entirely
            # when this exact experiment identity already has a prediction file.
            path = cached_path if resume_hit else run_experiment_row(cfg, row)
            status = "completed"
            error = None
        except Exception as exc:
            path = None
            status = "failed"
            error = repr(exc)
            append_jsonl(
                paths["logs"] / "errors.jsonl",
                {"created_at": utc_now(), "row": row, "error": error, "traceback": traceback.format_exc()},
            )
            print("FAILED:", error)
            if cfg["execution"].get("fail_fast", False):
                raise
        status_rows.append(
            {
                **row,
                "status": status,
                "prediction_path": str(path) if path else None,
                "resume_hit": resume_hit,
                "wall_seconds": time.perf_counter() - started,
                "error": error,
            }
        )
    status_df = pd.DataFrame(status_rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_df.to_csv(paths["logs"] / f"run_status_{stamp}.csv", index=False, encoding="utf-8-sig")
    return status_df


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if len(y) == 0:
        return float("nan")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(prob, quantiles))
    if len(edges) <= 2:
        return float(abs(prob.mean() - y.mean()))
    bins = np.digitize(prob, edges[1:-1], right=True)
    ece = 0.0
    for b in np.unique(bins):
        mask = bins == b
        ece += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(ece)


def binary_metrics(y: Sequence[int], prob: Sequence[float], cfg: dict[str, Any]) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    eps = float(cfg["metrics"]["probability_clip"])
    p = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    prevalence = float(y_arr.mean())
    ap = float(average_precision_score(y_arr, p))
    return {
        "auroc": float(roc_auc_score(y_arr, p)),
        "average_precision": ap,
        "ap_lift": ap / prevalence if prevalence > 0 else float("nan"),
        "log_loss": float(log_loss(y_arr, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "brier": float(brier_score_loss(y_arr, p)),
        "mcc": float(matthews_corrcoef(y_arr, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_arr, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ece": expected_calibration_error(y_arr, p, int(cfg["metrics"]["ece_bins"])),
        "prevalence": prevalence,
        "n_oof": int(len(y_arr)),
    }


def consolidate_predictions(cfg: dict[str, Any]) -> pd.DataFrame:
    paths = sorted((Path(cfg["results_root"]) / "predictions").glob("*.csv"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(Path(cfg["results_root"]) / "metrics" / "all_fold_predictions.csv", index=False, encoding="utf-8-sig")
    return out


def aggregate_repeat_metrics(cfg: dict[str, Any], predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    if predictions is None:
        predictions = consolidate_predictions(cfg)
    if predictions.empty:
        return pd.DataFrame()
    group_cols = [
        "stage",
        "dataset",
        "sample_regime",
        "method",
        "condition",
        "k",
        "alpha",
        "selector_seed",
        "model",
        "repeat",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(group_cols, dropna=False, sort=False):
        duplicate = group.duplicated("sample_id").any()
        if duplicate:
            raise ValueError(f"Duplicate OOF sample in group: {dict(zip(group_cols, key, strict=True))}")
        metrics = binary_metrics(group["y_true"], group["y_prob"], cfg)
        rows.append(
            {
                **dict(zip(group_cols, key, strict=True)),
                **metrics,
                "n_train": float(group["n_train"].median()),
                "n_selected": float(group["n_selected"].median()),
                "selected_k_median": float(group["selected_k"].median())
                if "selected_k" in group and group["selected_k"].notna().any()
                else float("nan"),
                "selected_alpha_median": float(group["selected_alpha"].median())
                if "selected_alpha" in group and group["selected_alpha"].notna().any()
                else float("nan"),
                # Timing metadata is repeated on every prediction row. Count
                # each outer fold once, then sum the five fold runtimes.
                "fit_predict_seconds": float(
                    group.groupby("fold", dropna=False)["fit_predict_seconds"].max().sum()
                ),
                "gpu_peak_memory_mb": float(group["gpu_peak_memory_mb"].max())
                if group["gpu_peak_memory_mb"].notna().any()
                else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(Path(cfg["results_root"]) / "metrics" / "repeat_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def aggregate_fold_metrics(cfg: dict[str, Any], predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute descriptive per-fold metrics for the configured outer CV."""
    if predictions is None:
        predictions = consolidate_predictions(cfg)
    if predictions.empty:
        return pd.DataFrame()
    group_cols = [
        "stage",
        "dataset",
        "sample_regime",
        "method",
        "condition",
        "k",
        "alpha",
        "selector_seed",
        "model",
        "repeat",
        "fold",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in predictions.groupby(group_cols, dropna=False, sort=False):
        rows.append(
            {
                **dict(zip(group_cols, key, strict=True)),
                **binary_metrics(group["y_true"], group["y_prob"], cfg),
                "n_train": float(group["n_train"].median()),
                "n_selected": float(group["n_selected"].median()),
                "selected_k": float(group["selected_k"].median())
                if "selected_k" in group and group["selected_k"].notna().any()
                else float("nan"),
                "selected_alpha": float(group["selected_alpha"].median())
                if "selected_alpha" in group and group["selected_alpha"].notna().any()
                else float("nan"),
                "fit_predict_seconds": float(group["fit_predict_seconds"].max()),
                "gpu_peak_memory_mb": float(group["gpu_peak_memory_mb"].max())
                if group["gpu_peak_memory_mb"].notna().any()
                else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(Path(cfg["results_root"]) / "metrics" / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def summarize_fold_variability(cfg: dict[str, Any], fold_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Report mean/SD/range across folds as descriptive variability, not a CI."""
    if fold_metrics is None:
        fold_metrics = aggregate_fold_metrics(cfg)
    if fold_metrics.empty:
        return pd.DataFrame()
    group_cols = [
        "stage",
        "dataset",
        "sample_regime",
        "method",
        "condition",
        "k",
        "alpha",
        "selector_seed",
        "model",
        "repeat",
    ]
    metric_cols = [
        "auroc",
        "average_precision",
        "ap_lift",
        "log_loss",
        "brier",
        "mcc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "ece",
    ]
    summary = fold_metrics.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["n_folds"] = int(cfg["cv"]["outer_folds"])
    summary.to_csv(
        Path(cfg["results_root"]) / "metrics" / "fold_variability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def summarize_metrics(cfg: dict[str, Any], repeat_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    if repeat_metrics is None:
        repeat_metrics = aggregate_repeat_metrics(cfg)
    if repeat_metrics.empty:
        return pd.DataFrame()
    group_cols = [
        "stage",
        "dataset",
        "sample_regime",
        "method",
        "condition",
        "k",
        "alpha",
        "selector_seed",
        "model",
    ]
    metric_cols = [
        "auroc",
        "average_precision",
        "ap_lift",
        "log_loss",
        "brier",
        "mcc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "ece",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in repeat_metrics.groupby(group_cols, dropna=False, sort=False):
        record = dict(zip(group_cols, key, strict=True))
        record["n_repeats"] = group["repeat"].nunique()
        for metric in metric_cols:
            vals = group[metric].dropna().to_numpy(dtype=float)
            mean = float(np.mean(vals)) if len(vals) else float("nan")
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            if len(vals) > 1:
                rng = np.random.default_rng(stable_int("bootstrap", key, metric))
                n_boot = int(cfg["metrics"]["bootstrap_iterations"])
                boot = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
                tail = (1.0 - float(cfg["metrics"]["confidence_level"])) / 2.0
                low, high = np.quantile(boot, [tail, 1.0 - tail])
                record[f"{metric}_ci_low"] = float(low)
                record[f"{metric}_ci_high"] = float(high)
            else:
                record[f"{metric}_ci_low"] = float("nan")
                record[f"{metric}_ci_high"] = float("nan")
        record["fit_predict_seconds_mean"] = float(group["fit_predict_seconds"].mean())
        record["gpu_peak_memory_mb_max"] = float(group["gpu_peak_memory_mb"].max())
        rows.append(record)
    out = pd.DataFrame(rows)
    out.to_csv(Path(cfg["results_root"]) / "metrics" / "metric_summary.csv", index=False, encoding="utf-8-sig")
    return out


def _kuncheva(a: set[str], b: set[str], p: int) -> float:
    if len(a) == 0 or len(b) == 0 or len(a) >= p or len(b) >= p:
        return float("nan")
    # Generalized Kuncheva correction for nested-CV selections whose chosen k
    # can differ between folds. It reduces to the standard index when sizes match.
    expected = len(a) * len(b) / p
    denom = min(len(a), len(b)) - expected
    return float((len(a & b) - expected) / denom) if denom else float("nan")


def consolidate_selections(cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((Path(cfg["results_root"]) / "selections").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if path.resolve() != selection_cache_path(cfg, record).resolve():
            continue
        if record.get("stage") == "tuning_internal":
            continue
        rows.append({**{k: v for k, v in record.items() if k != "feature_ids"}, "feature_ids": record["feature_ids"], "path": str(path)})
    return pd.DataFrame(rows)


def compute_selection_stability(cfg: dict[str, Any]) -> pd.DataFrame:
    selections = consolidate_selections(cfg)
    if selections.empty:
        return pd.DataFrame()
    group_cols = [
        "stage",
        "dataset",
        "sample_regime",
        "repeat",
        "method",
        "condition",
        "k",
        "alpha",
        "selector_seed",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in selections.groupby(group_cols, dropna=False, sort=False):
        key_record = dict(zip(group_cols, key, strict=True))
        p = load_dataset(cfg, str(key_record["dataset"])).X.shape[1]
        group = group.sort_values("fold")
        sets = [set(x) for x in group["feature_ids"]]
        jaccard: list[float] = []
        kuncheva: list[float] = []
        for a, b in itertools.combinations(sets, 2):
            jaccard.append(len(a & b) / len(a | b) if a | b else 1.0)
            kuncheva.append(_kuncheva(a, b, p))
        rows.append(
            {
                **key_record,
                "n_selections": len(sets),
                "n_pairs": len(jaccard),
                "expected_pairs": int(cfg["cv"]["outer_folds"] * (cfg["cv"]["outer_folds"] - 1) / 2),
                "complete_outer_cv": len(sets) == int(cfg["cv"]["outer_folds"]),
                "mean_jaccard": float(np.nanmean(jaccard)) if jaccard else float("nan"),
                "mean_kuncheva": float(np.nanmean(kuncheva)) if kuncheva else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(Path(cfg["results_root"]) / "metrics" / "selection_stability.csv", index=False, encoding="utf-8-sig")
    return out


def compute_selection_frequency(cfg: dict[str, Any]) -> pd.DataFrame:
    """Count how often each feature is selected across the five outer folds."""
    selections = consolidate_selections(cfg)
    if selections.empty:
        return pd.DataFrame()
    group_cols = [
        "stage",
        "dataset",
        "sample_regime",
        "repeat",
        "method",
        "condition",
        "k",
        "alpha",
        "selector_seed",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in selections.groupby(group_cols, dropna=False, sort=False):
        n_folds = int(group["fold"].nunique())
        counts: dict[str, int] = {}
        for features in group["feature_ids"]:
            for feature_id in set(map(str, features)):
                counts[feature_id] = counts.get(feature_id, 0) + 1
        consensus_threshold = max(1, int(np.ceil(0.8 * n_folds)))
        base = dict(zip(group_cols, key, strict=True))
        registry = load_dataset(cfg, str(base["dataset"])).registry.set_index("feature_id", drop=False)
        for feature_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            meta = registry.loc[feature_id] if feature_id in registry.index else None
            rows.append(
                {
                    **base,
                    "feature_id": feature_id,
                    "raw_name": str(meta["raw_name"]) if meta is not None else "",
                    "semantic_name": str(meta["semantic_name"]) if meta is not None else "",
                    "gene_symbol": str(meta.get("gene_symbol", "")) if meta is not None else "",
                    "selection_count": int(count),
                    "n_outer_folds_observed": n_folds,
                    "selection_frequency": float(count / n_folds) if n_folds else float("nan"),
                    "consensus_threshold": consensus_threshold,
                    "consensus_feature": bool(count >= consensus_threshold),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(
        Path(cfg["results_root"]) / "metrics" / "selection_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def compute_llm_run_stability(cfg: dict[str, Any]) -> pd.DataFrame:
    paths = init_result_dirs(cfg)
    rows: list[dict[str, Any]] = []
    for dataset in cfg["datasets"]:
        run_tables = []
        for run in range(1, int(cfg["llm"]["real_runs"]) + 1):
            raw = pd.read_csv(paths["llm_raw"] / f"real_run{run}.csv")
            raw = raw.loc[raw["dataset"] == dataset].copy()
            raw["run_text_score"] = 0.5 + (raw["confidence"] / 3.0) * (raw["relevance"] / 4.0 - 0.5)
            run_tables.append(raw)
        p = len(run_tables[0])
        k_values = sorted(
            {
                int(cfg["feature_selection"]["primary_k"]),
                *map(int, cfg["feature_selection"].get("hybrid_k_sensitivity", [])),
            }
        )
        for k in k_values:
            sets = []
            for run, table in enumerate(run_tables, start=1):
                sets.append(set(deterministic_top_k(table, "run_text_score", int(k), run)))
            jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
            kun = [_kuncheva(a, b, p) for a, b in itertools.combinations(sets, 2)]
            rows.append(
                {
                    "dataset": dataset,
                    "k": int(k),
                    "n_llm_runs": len(sets),
                    "llm_mean_jaccard": float(np.mean(jac)),
                    "llm_mean_kuncheva": float(np.mean(kun)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths["metrics"] / "llm_run_stability.csv", index=False, encoding="utf-8-sig")
    return out


def compatibility_table(cfg: dict[str, Any], repeat_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    if repeat_metrics is None:
        repeat_metrics = aggregate_repeat_metrics(cfg)
    primary_k = int(cfg["feature_selection"]["primary_k"])
    primary_alpha = float(cfg["feature_selection"]["hybrid_alpha_primary"])
    methods = ["text", "hybrid", "adaptive_hybrid", "data_only", "mi", "random", "no_fs"]
    frame = repeat_metrics.loc[
        (repeat_metrics["stage"] == "compatibility")
        & repeat_metrics["method"].isin(methods)
        & ((repeat_metrics["k"] == primary_k) | (repeat_metrics["method"] == "no_fs"))
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    baseline_method = str(cfg.get("stage_scope", {}).get("compatibility_baseline", "no_fs"))
    baseline = (
        frame.loc[frame["method"] == baseline_method]
        .groupby(["stage", "dataset", "sample_regime", "model", "repeat"], dropna=False)[
            ["auroc", "log_loss", "brier"]
        ]
        .mean()
        .add_prefix("baseline_")
        .reset_index()
    )
    target = frame.loc[
        frame["method"].isin(["text", "hybrid", "adaptive_hybrid", "data_only", "mi", "random"])
    ].merge(
        baseline, on=["stage", "dataset", "sample_regime", "model", "repeat"], how="inner"
    )
    target["baseline_method"] = baseline_method
    target["gain_auroc"] = target["auroc"] - target["baseline_auroc"]
    target["gain_log_loss"] = target["baseline_log_loss"] - target["log_loss"]
    target["gain_brier"] = target["baseline_brier"] - target["brier"]
    index_cols = ["stage", "dataset", "sample_regime", "k", "repeat", "method", "condition"]
    pivot = target.pivot_table(
        index=index_cols,
        columns="model",
        values=["gain_auroc", "gain_log_loss", "gain_brier"],
        aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    pivot["baseline_method"] = baseline_method
    for metric in ["gain_auroc", "gain_log_loss", "gain_brier"]:
        t, c = f"{metric}_tabpfn", f"{metric}_catboost"
        if t in pivot and c in pivot:
            pivot[f"compatibility_{metric.removeprefix('gain_')}"] = pivot[t] - pivot[c]
    pivot.to_csv(Path(cfg["results_root"]) / "metrics" / "compatibility_gain.csv", index=False, encoding="utf-8-sig")
    return pivot


def semantic_gain_table(cfg: dict[str, Any], repeat_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    if repeat_metrics is None:
        repeat_metrics = aggregate_repeat_metrics(cfg)
    semantic = repeat_metrics.loc[
        (repeat_metrics["stage"] == "semantic")
        & repeat_metrics["method"].isin(["text", "hybrid", "adaptive_hybrid"])
    ].copy()
    if semantic.empty:
        return pd.DataFrame()
    core_real = repeat_metrics.loc[
        (repeat_metrics["stage"] == "core")
        & (repeat_metrics["condition"] == "real")
        & repeat_metrics["method"].isin(["text", "hybrid", "adaptive_hybrid"])
    ].copy()
    core_real["stage"] = "semantic"
    frame = pd.concat([semantic, core_real], ignore_index=True)
    if frame.empty:
        return pd.DataFrame()
    frame["condition_group"] = np.where(
        frame["condition"].astype(str).str.startswith("permuted_seed_"), "permuted", frame["condition"]
    )
    averaged = (
        frame.groupby(
            ["stage", "dataset", "sample_regime", "method", "condition_group", "k", "model", "repeat"],
            dropna=False,
        )[["auroc", "log_loss", "brier"]]
        .mean()
        .reset_index()
    )
    wide = averaged.pivot_table(
        index=["stage", "dataset", "sample_regime", "method", "k", "model", "repeat"],
        columns="condition_group",
        values=["auroc", "log_loss", "brier"],
    )
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    if {"auroc_real", "auroc_permuted"}.issubset(wide.columns):
        wide["semantic_gain_auroc"] = wide["auroc_real"] - wide["auroc_permuted"]
        wide["semantic_gain_log_loss"] = wide["log_loss_permuted"] - wide["log_loss_real"]
        wide["semantic_gain_brier"] = wide["brier_permuted"] - wide["brier_real"]
    wide.to_csv(Path(cfg["results_root"]) / "metrics" / "semantic_gain.csv", index=False, encoding="utf-8-sig")
    return wide


def primary_core_oof_predictions(
    cfg: dict[str, Any], predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return one repeat-averaged OOF probability per sample and method.

    Random selector seeds are averaged within repeat first, then all outer-CV
    repeats are averaged.  This produces one paired probability per sample for
    the confirmatory metric and paired bootstrap, while repeat-level metrics
    remain available separately in ``repeat_metrics.csv``.
    """
    if predictions is None:
        predictions = consolidate_predictions(cfg)
    if predictions.empty:
        return pd.DataFrame()
    primary_k = int(cfg["feature_selection"]["primary_k"])
    core_frame = predictions.loc[
        (predictions["stage"] == "core")
        & (predictions["model"] == cfg["models"]["primary"])
        & predictions["condition"].isin(["real", "name_invariant"])
        & ((predictions["k"] == primary_k) | (predictions["method"] == "no_fs"))
    ].copy()
    if core_frame.empty:
        return pd.DataFrame()
    frame = core_frame
    repeat_cols = ["dataset", "sample_regime", "repeat", "method", "sample_id"]
    within_repeat = (
        frame.groupby(repeat_cols, as_index=False, dropna=False)
        .agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean"),
            n_selector_seeds=("selector_seed", "nunique"),
        )
    )
    truth_check = frame.groupby(repeat_cols, dropna=False)["y_true"].nunique()
    if int(truth_check.max()) != 1:
        raise ValueError("Inconsistent y_true values while pooling primary OOF predictions")
    out = (
        within_repeat.groupby(
            ["dataset", "sample_regime", "method", "sample_id"],
            as_index=False,
            dropna=False,
        )
        .agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean"),
            n_outer_repeats=("repeat", "nunique"),
            n_selector_seeds=("n_selector_seeds", "max"),
        )
    )
    out.to_csv(
        Path(cfg["results_root"]) / "metrics" / "primary_core_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def primary_core_oof_metrics(
    cfg: dict[str, Any], predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    pooled = primary_core_oof_predictions(cfg, predictions)
    if pooled.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in pooled.groupby(["dataset", "sample_regime", "method"], sort=False):
        rows.append(
            {
                "dataset": key[0],
                "sample_regime": key[1],
                "method": key[2],
                **binary_metrics(group["y_true"], group["y_prob"], cfg),
                "n_outer_repeats": int(group["n_outer_repeats"].max()),
                "n_selector_seeds": int(group["n_selector_seeds"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(
        Path(cfg["results_root"]) / "metrics" / "primary_core_oof_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (m - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_bootstrap_auroc(
    cfg: dict[str, Any], predictions: pd.DataFrame | None = None, reference_method: str = "adaptive_hybrid"
) -> pd.DataFrame:
    """Paired sample bootstrap CIs for pooled OOF AUROC differences."""
    pooled = primary_core_oof_predictions(cfg, predictions)
    if pooled.empty:
        return pd.DataFrame()
    iterations = int(cfg["metrics"].get("bootstrap_iterations", 2000))
    confidence = float(cfg["metrics"].get("confidence_level", 0.95))
    tail = (1.0 - confidence) / 2.0
    base_seed = int(cfg.get("seeds", {}).get("bootstrap_seed", 20260816))
    rows: list[dict[str, Any]] = []
    for (dataset, sample_regime), block in pooled.groupby(
        ["dataset", "sample_regime"], sort=False
    ):
        reference = block.loc[block["method"] == reference_method, ["sample_id", "y_true", "y_prob"]].rename(
            columns={"y_prob": "reference_prob"}
        )
        if reference.empty:
            continue
        for comparator_method in sorted(set(block["method"]) - {reference_method}):
            comparator = block.loc[
                block["method"] == comparator_method, ["sample_id", "y_true", "y_prob"]
            ].rename(columns={"y_true": "comparator_y", "y_prob": "comparator_prob"})
            pair = reference.merge(comparator, on="sample_id", how="inner", validate="one_to_one")
            if pair.empty or not np.array_equal(
                pair["y_true"].to_numpy(dtype=int), pair["comparator_y"].to_numpy(dtype=int)
            ):
                raise ValueError(
                    f"Paired OOF alignment failed: {dataset}/{sample_regime}/{comparator_method}"
                )
            y = pair["y_true"].to_numpy(dtype=int)
            ref_prob = pair["reference_prob"].to_numpy(dtype=float)
            cmp_prob = pair["comparator_prob"].to_numpy(dtype=float)
            observed_ref = float(roc_auc_score(y, ref_prob))
            observed_cmp = float(roc_auc_score(y, cmp_prob))
            rng = np.random.default_rng(
                stable_int("paired_bootstrap", base_seed, dataset, sample_regime, comparator_method)
            )
            deltas: list[float] = []
            while len(deltas) < iterations:
                indices = rng.integers(0, len(y), size=len(y))
                sampled_y = y[indices]
                if len(np.unique(sampled_y)) < 2:
                    continue
                deltas.append(
                    float(
                        roc_auc_score(sampled_y, ref_prob[indices])
                        - roc_auc_score(sampled_y, cmp_prob[indices])
                    )
                )
            boot = np.asarray(deltas, dtype=float)
            lower, upper = np.quantile(boot, [tail, 1.0 - tail])
            p_lower = (np.count_nonzero(boot <= 0.0) + 1.0) / (len(boot) + 1.0)
            p_upper = (np.count_nonzero(boot >= 0.0) + 1.0) / (len(boot) + 1.0)
            rows.append(
                {
                    "dataset": dataset,
                    "sample_regime": sample_regime,
                    "n_outer_repeats": int(block["n_outer_repeats"].max()),
                    "metric": "auroc",
                    "reference_method": reference_method,
                    "comparator_method": comparator_method,
                    "reference_auroc": observed_ref,
                    "comparator_auroc": observed_cmp,
                    "delta_auroc": observed_ref - observed_cmp,
                    "ci_level": confidence,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "p_two_sided": float(min(1.0, 2.0 * min(p_lower, p_upper))),
                    "bootstrap_iterations": iterations,
                    "n_samples": int(len(pair)),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = _holm_adjust(out["p_two_sided"])
        out["significant_0_05_holm"] = out["p_holm"] < 0.05
        out.to_csv(
            Path(cfg["results_root"]) / "metrics" / "paired_bootstrap_auroc.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return out


def plot_paired_bootstrap_auroc(cfg: dict[str, Any], bootstrap: pd.DataFrame) -> Path:
    if bootstrap.empty:
        raise ValueError("Paired bootstrap results are not available")
    set_plot_style()
    frame = bootstrap.copy()
    frame["comparison"] = (
        frame["dataset"].astype(str)
        + " / "
        + frame["sample_regime"].astype(str)
        + " / vs "
        + frame["comparator_method"].astype(str)
    )
    frame = frame.sort_values(["dataset", "sample_regime", "delta_auroc"]).reset_index(drop=True)
    y = np.arange(len(frame))
    lower = frame["delta_auroc"].to_numpy() - frame["ci_lower"].to_numpy()
    upper = frame["ci_upper"].to_numpy() - frame["delta_auroc"].to_numpy()
    fig, ax = plt.subplots(figsize=(12, max(7, 0.32 * len(frame))))
    ax.errorbar(
        frame["delta_auroc"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=3,
        color="#2878B5",
        ecolor="#6B7280",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, frame["comparison"])
    ax.set(
        xlabel=f"Paired AUROC difference ({frame['reference_method'].iloc[0]} - comparator)",
        ylabel="Dataset / sample regime / comparison",
        title="Paired bootstrap 95% confidence intervals",
    )
    base = Path(cfg["results_root"]) / "figures" / "paired_bootstrap_auroc"
    _save_figure(fig, base)
    frame.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(base.with_suffix(suffix), dpi=300, bbox_inches="tight")


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def plot_metric_vs_sample_size(
    cfg: dict[str, Any], repeat_metrics: pd.DataFrame, metric: str = "auroc", stage: str = "core"
) -> Path:
    set_plot_style()
    frame = repeat_metrics.loc[repeat_metrics["stage"] == stage].copy()
    frame = frame.loc[(frame["condition"].isin(["real", "name_invariant"]))]
    available_k = sorted(frame["k"].dropna().astype(int).unique().tolist())
    preferred_k = int(cfg["feature_selection"]["primary_k"])
    primary_k = preferred_k if preferred_k in available_k else (available_k[0] if available_k else preferred_k)
    frame = frame.loc[(frame["k"] == primary_k) | (frame["method"] == "no_fs")]
    if frame.empty:
        raise ValueError(f"No {stage} metrics")
    averaged = (
        frame.groupby(["dataset", "method", "model", "n_train"], dropna=False)[metric].mean().reset_index()
    )
    g = sns.relplot(
        data=averaged,
        x="n_train",
        y=metric,
        hue="method",
        col="dataset",
        col_wrap=3,
        kind="line",
        marker="o",
        facet_kws={"sharex": False},
        height=4,
    )
    g.set_axis_labels("Outer-training samples", metric.upper())
    base = Path(cfg["results_root"]) / "figures" / f"{stage}_{metric}_vs_sample_size"
    _save_figure(g.figure, base)
    averaged.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(g.figure)
    return base.with_suffix(".png")


def plot_metric_vs_k(
    cfg: dict[str, Any], repeat_metrics: pd.DataFrame, metric: str = "auroc", stage: str = "core"
) -> Path:
    set_plot_style()
    frame = repeat_metrics.loc[
        (repeat_metrics["stage"] == stage)
        & (repeat_metrics["method"] == "hybrid")
        & repeat_metrics["k"].notna()
        & (repeat_metrics["condition"] == "real")
        & (repeat_metrics["model"] == cfg["models"]["primary"])
    ].copy()
    primary_alpha = float(cfg["feature_selection"]["hybrid_alpha_primary"])
    frame = frame.loc[frame["alpha"] == primary_alpha]
    averaged = frame.groupby(["dataset", "sample_regime", "k"], dropna=False)[metric].mean().reset_index()
    g = sns.relplot(
        data=averaged,
        x="k",
        y=metric,
        hue="sample_regime",
        col="dataset",
        col_wrap=3,
        kind="line",
        marker="o",
        height=4,
    )
    g.set_axis_labels("Selected features (k)", metric.upper())
    base = Path(cfg["results_root"]) / "figures" / f"{stage}_{metric}_vs_k"
    _save_figure(g.figure, base)
    averaged.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(g.figure)
    return base.with_suffix(".png")


def plot_name_condition_effect(cfg: dict[str, Any], repeat_metrics: pd.DataFrame, metric: str = "auroc") -> Path:
    set_plot_style()
    semantic = repeat_metrics.loc[
        (repeat_metrics["stage"] == "semantic")
        & (repeat_metrics["model"] == cfg["models"]["primary"])
        & repeat_metrics["method"].isin(["text", "hybrid", "adaptive_hybrid"])
    ].copy()
    core_real = repeat_metrics.loc[
        (repeat_metrics["stage"] == "core")
        & (repeat_metrics["model"] == cfg["models"]["primary"])
        & (repeat_metrics["condition"] == "real")
        & repeat_metrics["method"].isin(["text", "hybrid", "adaptive_hybrid"])
    ].copy()
    core_real["stage"] = "semantic"
    frame = pd.concat([semantic, core_real], ignore_index=True)
    primary_k = int(cfg["feature_selection"]["primary_k"])
    frame = frame.loc[frame["k"] == primary_k]
    frame["condition_group"] = np.where(
        frame["condition"].astype(str).str.startswith("permuted_seed_"), "Permuted", frame["condition"].str.title()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=frame, x="dataset", y=metric, hue="condition_group", ax=ax, showfliers=False)
    sns.stripplot(data=frame, x="dataset", y=metric, hue="condition_group", dodge=True, alpha=0.25, ax=ax, legend=False)
    ax.set_title(f"Name-condition effect on {metric.upper()}")
    base = Path(cfg["results_root"]) / "figures" / f"semantic_{metric}_name_conditions"
    _save_figure(fig, base)
    frame.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def plot_nested_tuning_choices(cfg: dict[str, Any], decisions: pd.DataFrame | None = None) -> Path:
    if decisions is None:
        decisions = consolidate_tuning_decisions(cfg)
    if decisions.empty:
        raise ValueError("Nested tuning decisions are not available")
    frame = decisions.loc[decisions["stage"] == "core"].copy()
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    k_counts = (
        frame.groupby(["method", "selected_k"], as_index=False)
        .size()
        .rename(columns={"size": "outer_fold_count"})
    )
    sns.barplot(
        data=k_counts,
        x="method",
        y="outer_fold_count",
        hue="selected_k",
        ax=axes[0],
    )
    axes[0].set(title="Nested-CV selected k", xlabel="Feature selector", ylabel="Outer-fold count")
    axes[0].tick_params(axis="x", rotation=45)
    alpha_frame = frame.loc[
        frame["method"].isin(["hybrid", "adaptive_hybrid"]) & frame["selected_alpha"].notna()
    ].copy()
    if alpha_frame.empty:
        axes[1].set_axis_off()
    else:
        alpha_counts = (
            alpha_frame.groupby(["method", "selected_alpha"], as_index=False)
            .size()
            .rename(columns={"size": "outer_fold_count"})
        )
        sns.barplot(
            data=alpha_counts,
            x="method",
            y="outer_fold_count",
            hue="selected_alpha",
            ax=axes[1],
        )
        axes[1].set(title="Nested-CV selected alpha", xlabel="Hybrid selector", ylabel="Outer-fold count")
    base = Path(cfg["results_root"]) / "figures" / "nested_tuning_choices"
    _save_figure(fig, base)
    frame.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def plot_calibration(
    cfg: dict[str, Any],
    predictions: pd.DataFrame,
    dataset: str,
    stage: str = "core",
    bins: int = 10,
    model: str | None = None,
    sample_regime: str | int = "full",
    k: int | None = None,
) -> Path:
    set_plot_style()
    frame = predictions.loc[
        (predictions["dataset"] == dataset)
        & (predictions["stage"] == stage)
        & (predictions["sample_regime"].astype(str) == str(sample_regime))
        & predictions["condition"].isin(["real", "name_invariant"])
    ].copy()
    if frame.empty:
        raise ValueError(dataset)
    if model is None:
        model = "tabpfn" if "tabpfn" in set(frame["model"]) else str(frame["model"].iloc[0])
    frame = frame.loc[frame["model"] == model].copy()
    available_k = sorted(frame["k"].dropna().astype(int).unique().tolist())
    if k is None and available_k:
        preferred_k = int(cfg["feature_selection"]["primary_k"])
        k = preferred_k if preferred_k in available_k else available_k[0]
    if k is not None:
        frame = frame.loc[(frame["k"] == k) | (frame["method"] == "no_fs")].copy()
    if frame.empty:
        raise ValueError(f"No calibration rows for {dataset}/{stage}/{model}/k={k}")
    averaged = frame.groupby(["method", "sample_id"], as_index=False).agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--", color="black", label="Perfect")
    plot_rows = []
    for method, group in averaged.groupby("method"):
        group = group.sort_values("y_prob")
        unique_probabilities = int(group["y_prob"].nunique())
        if unique_probabilities < 2:
            continue
        group["bin"] = pd.qcut(group["y_prob"], q=min(bins, unique_probabilities), duplicates="drop")
        curve = group.groupby("bin", observed=True).agg(mean_prob=("y_prob", "mean"), observed=("y_true", "mean"))
        ax.plot(curve["mean_prob"], curve["observed"], marker="o", label=method)
        for rec in curve.reset_index(drop=True).to_dict("records"):
            plot_rows.append({"dataset": dataset, "method": method, **rec})
    k_label = "nested-selected k" if cfg.get("nested_tuning", {}).get("enabled", False) else f"k={k}"
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive rate",
        title=f"Calibration: {dataset} ({model}, {k_label})",
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    base = Path(cfg["results_root"]) / "figures" / f"{stage}_{dataset}_calibration"
    _save_figure(fig, base)
    pd.DataFrame(plot_rows).to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def plot_compatibility_interaction(cfg: dict[str, Any], compatibility: pd.DataFrame) -> Path:
    set_plot_style()
    gain_cols = [c for c in compatibility if c.startswith("gain_auroc_")]
    long = compatibility.melt(
        id_vars=["dataset", "sample_regime", "k", "method", "condition", "repeat"],
        value_vars=gain_cols,
        var_name="model",
        value_name="gain_auroc",
    )
    long["model"] = long["model"].str.replace("gain_auroc_", "", regex=False)
    summary = long.groupby(["method", "model"], as_index=False)["gain_auroc"].mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.pointplot(data=summary, x="method", y="gain_auroc", hue="model", dodge=0.25, ax=ax)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    baseline = (
        str(compatibility["baseline_method"].iloc[0])
        if "baseline_method" in compatibility and not compatibility.empty
        else "baseline"
    )
    ax.set(
        title="FS × classifier interaction",
        ylabel=f"AUROC gain over {baseline}",
        xlabel="Feature selector",
    )
    base = Path(cfg["results_root"]) / "figures" / "compatibility_interaction_auroc"
    _save_figure(fig, base)
    long.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def plot_performance_heatmap(
    cfg: dict[str, Any], repeat_metrics: pd.DataFrame, metric: str = "auroc", stage: str = "core"
) -> Path:
    set_plot_style()
    frame = repeat_metrics.loc[
        (repeat_metrics["stage"] == stage)
        & (repeat_metrics["model"] == "tabpfn")
        & repeat_metrics["condition"].isin(["real", "name_invariant"])
    ].copy()
    primary_k = int(cfg["feature_selection"]["primary_k"])
    frame = frame.loc[frame["sample_regime"].astype(str).str.lower() == "full"]
    frame = frame.loc[(frame["k"] == primary_k) | (frame["method"] == "no_fs")]
    if frame.empty:
        raise ValueError(f"No {stage} TabPFN metrics")
    table = frame.pivot_table(index="method", columns="dataset", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.65 * len(table))))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="viridis", vmin=0.5 if metric == "auroc" else None, ax=ax)
    ax.set(title=f"TabPFN {metric.upper()} summary", xlabel="Dataset", ylabel="Feature selector")
    base = Path(cfg["results_root"]) / "figures" / f"{stage}_{metric}_performance_heatmap"
    _save_figure(fig, base)
    table.to_csv(base.with_suffix(".csv"), encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def plot_stability_heatmap(cfg: dict[str, Any], stability: pd.DataFrame, metric: str = "mean_kuncheva") -> Path:
    set_plot_style()
    primary_k = int(cfg["feature_selection"]["primary_k"])
    frame = stability.loc[
        (stability["stage"] == "core")
        & stability["condition"].isin(["real", "name_invariant"])
        & stability[metric].notna()
        & stability["complete_outer_cv"].astype(bool)
        & ((stability["k"] == primary_k) | (stability["method"] == "no_fs"))
    ].copy()
    if frame.empty:
        raise ValueError("No repeated feature selections are available for stability plotting")
    table = frame.pivot_table(index="method", columns="dataset", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.65 * len(table))))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="mako", vmin=-1, vmax=1, ax=ax)
    ax.set(title=f"Selection stability: {metric}", xlabel="Dataset", ylabel="Feature selector")
    base = Path(cfg["results_root"]) / "figures" / f"selection_stability_{metric}"
    _save_figure(fig, base)
    table.to_csv(base.with_suffix(".csv"), encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def plot_resource_usage(cfg: dict[str, Any], repeat_metrics: pd.DataFrame, stage: str = "core") -> Path:
    set_plot_style()
    frame = repeat_metrics.loc[repeat_metrics["stage"] == stage].copy()
    primary_k = int(cfg["feature_selection"]["primary_k"])
    frame = frame.loc[
        frame["condition"].isin(["real", "name_invariant"])
        & ((frame["k"] == primary_k) | (frame["method"] == "no_fs"))
    ]
    if frame.empty:
        raise ValueError(f"No {stage} resource metrics")
    summary = frame.groupby(["method", "model"], as_index=False).agg(
        fit_predict_seconds=("fit_predict_seconds", "mean"),
        gpu_peak_memory_mb=("gpu_peak_memory_mb", "max"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=summary, x="method", y="fit_predict_seconds", hue="model", ax=axes[0])
    sns.barplot(data=summary, x="method", y="gpu_peak_memory_mb", hue="model", ax=axes[1])
    axes[0].set(title="Mean fold runtime", xlabel="Feature selector", ylabel="Seconds")
    axes[1].set(title="Maximum recorded GPU memory", xlabel="Feature selector", ylabel="MB")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    base = Path(cfg["results_root"]) / "figures" / f"{stage}_resource_usage"
    _save_figure(fig, base)
    summary.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def plot_hybrid_alpha_sensitivity(
    cfg: dict[str, Any], repeat_metrics: pd.DataFrame, metric: str = "auroc"
) -> Path:
    set_plot_style()
    primary_k = int(cfg["feature_selection"]["primary_k"])
    frame = repeat_metrics.loc[
        (repeat_metrics["stage"] == "core")
        & (repeat_metrics["method"] == "hybrid")
        & (repeat_metrics["condition"] == "real")
        & (repeat_metrics["model"] == "tabpfn")
        & (repeat_metrics["k"] == primary_k)
        & (repeat_metrics["sample_regime"].astype(str).str.lower() == "full")
    ].copy()
    if frame["alpha"].nunique() < 2:
        raise ValueError("Hybrid alpha sensitivity results are not available")
    summary = frame.groupby(["dataset", "alpha"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(data=summary, x="alpha", y=metric, hue="dataset", marker="o", ax=ax)
    ax.axvline(float(cfg["feature_selection"]["hybrid_alpha_primary"]), color="black", linestyle="--")
    ax.set(
        title=f"Hybrid alpha sensitivity ({metric.upper()}, full, k={primary_k})",
        xlabel="LLM weight alpha",
    )
    base = Path(cfg["results_root"]) / "figures" / f"hybrid_alpha_sensitivity_{metric}"
    _save_figure(fig, base)
    summary.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def llm_execution_plan(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for condition, runs in [("real", int(cfg["llm"]["real_runs"])), ("anonymous", int(cfg["llm"]["anonymous_runs"]))]:
        for run in range(1, runs + 1):
            requests, manifest = make_llm_inline_requests(cfg, condition, run)
            rows.append(
                {
                    "condition": condition,
                    "run_index": run,
                    "logical_requests": len(requests),
                    "features": sum(len(m["feature_ids"]) for m in manifest),
                    "model": cfg["llm"]["model"],
                }
            )
    return pd.DataFrame(rows)


def submit_all_llm_jobs(cfg: dict[str, Any], confirm: str) -> pd.DataFrame:
    if confirm != "SUBMIT_GEMINI_BATCH_JOBS":
        raise RuntimeError("Submission cancelled: exact confirmation text was not provided")
    records = []
    for condition, runs in [("real", int(cfg["llm"]["real_runs"])), ("anonymous", int(cfg["llm"]["anonymous_runs"]))]:
        for run in range(1, runs + 1):
            job_file = Path(cfg["results_root"]) / "llm_jobs" / f"{condition}_run{run}" / "job.json"
            if job_file.exists():
                records.append(json.loads(job_file.read_text(encoding="utf-8")))
            else:
                records.append(submit_llm_batch(cfg, condition, run))
    return pd.DataFrame(records)


def all_llm_job_statuses(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for condition, runs in [("real", int(cfg["llm"]["real_runs"])), ("anonymous", int(cfg["llm"]["anonymous_runs"]))]:
        for run in range(1, runs + 1):
            try:
                rows.append({"condition": condition, "run_index": run, **get_llm_job_status(cfg, condition, run)})
            except FileNotFoundError:
                rows.append({"condition": condition, "run_index": run, "state": "NOT_SUBMITTED", "terminal": False})
    return pd.DataFrame(rows)


def retrieve_and_prepare_all_llm_scores(cfg: dict[str, Any]) -> dict[str, Any]:
    retrieved = []
    for condition, runs in [("real", int(cfg["llm"]["real_runs"])), ("anonymous", int(cfg["llm"]["anonymous_runs"]))]:
        for run in range(1, runs + 1):
            path = Path(cfg["results_root"]) / "llm_raw" / f"{condition}_run{run}.csv"
            if path.exists():
                frame = pd.read_csv(path)
            else:
                frame = retrieve_llm_batch(cfg, condition, run)
            retrieved.append({"condition": condition, "run": run, "rows": len(frame)})
        aggregate_llm_scores(cfg, condition)
    permuted = build_permuted_llm_scores(cfg)
    return {"retrieved": retrieved, "permuted_files": len(permuted)}


def llm_score_cache_status(cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in cfg["datasets"]:
        expected_ids = set(load_dataset(cfg, dataset).registry["feature_id"].astype(str))
        for condition in all_name_conditions(cfg):
            path = Path(cfg["results_root"]) / "llm_scores" / dataset / f"{condition}.csv"
            exists = path.is_file()
            valid = False
            n_features = 0
            if exists:
                try:
                    score = pd.read_csv(path)
                    observed = set(score["feature_id"].astype(str))
                    n_features = len(score)
                    valid = observed == expected_ids and len(score) == len(expected_ids)
                except Exception:
                    valid = False
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "exists": exists,
                    "valid": valid,
                    "n_features": n_features,
                    "path": str(path),
                }
            )
    return pd.DataFrame(rows)


def reuse_llm_score_cache(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy compatible second-experiment LLM outputs into the isolated run.

    LLM scores depend only on task text and feature names, both unchanged in
    experiment 3. Reusing them saves API cost without reusing labels, folds,
    data-driven scores, selected features, predictions, or metrics.
    """
    before = llm_score_cache_status(cfg)
    if not before.empty and before["valid"].all():
        return {"source": "current_results", "copied_files": 0, "ready": True}
    source_root_value = cfg.get("llm", {}).get("reuse_scores_root")
    if not cfg.get("llm", {}).get("reuse_existing_scores", True) or not source_root_value:
        return {"source": None, "copied_files": 0, "ready": False}
    source_root = Path(source_root_value)
    if not source_root.is_dir():
        return {"source": str(source_root), "copied_files": 0, "ready": False}
    source_cfg_path = source_root / "resolved_experiment_config.json"
    if not source_cfg_path.is_file():
        return {
            "source": str(source_root),
            "copied_files": 0,
            "ready": False,
            "reason": "source configuration snapshot is missing",
        }
    source_cfg = json.loads(source_cfg_path.read_text(encoding="utf-8"))
    source_llm = source_cfg.get("llm", {})
    current_llm = cfg.get("llm", {})
    llm_keys = ["model", "thinking_level", "real_runs", "anonymous_runs", "run_seeds"]
    mismatches = {
        key: {"source": source_llm.get(key), "current": current_llm.get(key)}
        for key in llm_keys
        if source_llm.get(key) != current_llm.get(key)
    }
    source_tasks = {
        dataset: source_cfg.get("datasets", {}).get(dataset, {}).get("task")
        for dataset in cfg["datasets"]
    }
    current_tasks = {dataset: cfg["datasets"][dataset].get("task") for dataset in cfg["datasets"]}
    if source_tasks != current_tasks:
        mismatches["dataset_tasks"] = {"source": source_tasks, "current": current_tasks}
    if mismatches:
        return {
            "source": str(source_root),
            "copied_files": 0,
            "ready": False,
            "reason": "LLM model, run configuration, or task text differs",
            "mismatches": mismatches,
        }
    copied = 0
    for folder in ["llm_raw", "llm_scores"]:
        source_dir = source_root / folder
        target_dir = Path(cfg["results_root"]) / folder
        if not source_dir.is_dir():
            continue
        for source in source_dir.rglob("*"):
            if not source.is_file():
                continue
            target = target_dir / source.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
                copied += 1
    after = llm_score_cache_status(cfg)
    ready = bool(not after.empty and after["valid"].all())
    report = {
        "source": str(source_root),
        "copied_files": copied,
        "ready": ready,
        "validated_rows": int(after["valid"].sum()) if not after.empty else 0,
        "expected_rows": len(after),
    }
    atomic_write_json(Path(cfg["results_root"]) / "logs" / "llm_cache_reuse.json", report)
    return report


def fisher_discriminant_ratio(embedding: np.ndarray, y: Sequence[int]) -> float:
    z = np.asarray(embedding, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    z0, z1 = z[y_arr == 0], z[y_arr == 1]
    if len(z0) < 2 or len(z1) < 2:
        return float("nan")
    c0, c1 = z0.mean(axis=0), z1.mean(axis=0)
    between = float(np.sum((c1 - c0) ** 2))
    within = float(np.mean(np.sum((z0 - c0) ** 2, axis=1)) + np.mean(np.sum((z1 - c1) ** 2, axis=1)))
    return between / within if within > 0 else float("inf")


def build_embedding_grid(cfg: dict[str, Any]) -> pd.DataFrame:
    if not cfg.get("embedding", {}).get("enabled", True):
        return pd.DataFrame()
    if cfg["embedding"].get("outer_folds") == "all":
        folds = range(1, int(cfg["cv"]["outer_folds"]) + 1)
    else:
        folds = [1]
    repeats = (
        range(1, int(cfg["cv"]["outer_repeats"]) + 1)
        if cfg["embedding"].get("outer_repeats") == "all"
        else [1]
    )
    rows: list[dict[str, Any]] = []
    for dataset, repeat, fold, method in itertools.product(
        cfg["datasets"], repeats, folds, cfg["embedding"]["methods"]
    ):
        selector_seed = (
            int(cfg["feature_selection"]["random_seeds"][0])
            if method == "random"
            else int(cfg.get("seeds", {}).get("selector_tie_seed", 20260814))
        )
        rows.append(
            {
                "stage": "embedding",
                "dataset": dataset,
                "sample_regime": "full",
                "repeat": int(repeat),
                "fold": int(fold),
                "method": method,
                "condition": (
                    "real"
                    if method in {"text", "hybrid", "adaptive_hybrid"}
                    else "name_invariant"
                ),
                "k": int(cfg["feature_selection"]["primary_k"]),
                "alpha": float(cfg["feature_selection"]["hybrid_alpha_primary"]),
                "selector_seed": selector_seed,
                "model": "tabpfn",
            }
        )
    return pd.DataFrame(rows)


def run_embedding_case(cfg: dict[str, Any], row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    from tabpfn_extensions.embedding import TabPFNEmbedding

    if row["model"] != "tabpfn":
        raise ValueError("Embeddings are only extracted from local TabPFN")
    bundle = load_dataset(cfg, row["dataset"])
    outer_train, outer_test = get_outer_indices(cfg, bundle, int(row["repeat"]), int(row["fold"]))
    subset_seed = stable_int(
        "sample",
        int(cfg.get("seeds", {}).get("subset_seed", 20260813)),
        row["dataset"],
        row["repeat"],
        row["fold"],
    )
    train_idx = training_subset(bundle, outer_train, row["sample_regime"], subset_seed)
    selected = select_feature_ids(cfg, bundle, train_idx, row)
    reg = subset_registry(bundle.registry, selected)
    train, test, cat_indices = prepare_tabular_frames(
        bundle.X.iloc[train_idx][selected].reset_index(drop=True),
        bundle.X.iloc[outer_test][selected].reset_index(drop=True),
        reg,
    )
    seed = stable_int(
        "paired_model",
        int(cfg.get("seeds", {}).get("model_seed", 20260813)),
        row["repeat"],
        row["fold"],
    )
    seed_everything(seed)
    model = make_tabpfn_classifier(cfg, cat_indices, seed, no_fs=False)
    extractor = TabPFNEmbedding(
        n_fold=int(cfg["embedding"]["n_fold"]),
        model=model,
        shuffle=True,
        random_state=seed,
    )
    train_3d = extractor.fit_transform(train.to_numpy(), bundle.y.iloc[train_idx].to_numpy(dtype=int))
    test_3d = extractor.transform(test.to_numpy())
    train_embedding = np.mean(train_3d, axis=0)
    test_embedding = np.mean(test_3d, axis=0)
    y_test = bundle.y.iloc[outer_test].to_numpy(dtype=int)
    silhouette = (
        float(silhouette_score(test_embedding, y_test, metric="cosine"))
        if len(np.unique(y_test)) == 2 and min(np.bincount(y_test)) >= 2
        else float("nan")
    )
    fisher = fisher_discriminant_ratio(test_embedding, y_test)
    identity = {k: row.get(k) for k in ["dataset", "sample_regime", "repeat", "fold", "method", "condition", "k"]}
    base = Path(cfg["results_root"]) / "embeddings" / f"embedding_{stable_hash(identity)[:16]}"
    np.savez_compressed(
        base.with_suffix(".npz"),
        train_embedding=train_embedding,
        test_embedding=test_embedding,
        y_train=bundle.y.iloc[train_idx].to_numpy(dtype=int),
        y_test=y_test,
        train_sample_id=bundle.sample_ids.iloc[train_idx].to_numpy(),
        test_sample_id=bundle.sample_ids.iloc[outer_test].to_numpy(),
        selected_features=np.asarray(selected),
    )
    metrics = {
        **identity,
        "silhouette_cosine": silhouette,
        "fisher_ratio": fisher,
        "n_train": len(train_idx),
        "n_test": len(outer_test),
        "embedding_dim": test_embedding.shape[1],
        "created_at": utc_now(),
    }
    atomic_write_json(base.with_suffix(".json"), metrics)
    del extractor, model, train_3d, test_3d
    gc.collect()
    return base.with_suffix(".npz"), metrics


def consolidate_embedding_metrics(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(cfg["results_root"]) / "embeddings").glob("embedding_*.json"))]
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(Path(cfg["results_root"]) / "metrics" / "embedding_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def plot_embedding_case(cfg: dict[str, Any], npz_path: str | Path) -> Path:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    train_z, test_z = data["train_embedding"], data["test_embedding"]
    y_train, y_test = data["y_train"], data["y_test"]
    projection_seed = int(cfg.get("seeds", {}).get("model_seed", 20260813))
    pca = PCA(n_components=2, random_state=projection_seed)
    pca_train = pca.fit_transform(train_z)
    pca_test = pca.transform(test_z)
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(int(cfg["embedding"]["umap_neighbors"]), max(2, len(train_z) - 1)),
            min_dist=float(cfg["embedding"]["umap_min_dist"]),
            metric="cosine",
            random_state=projection_seed,
        )
        umap_train = reducer.fit_transform(train_z)
        umap_test = reducer.transform(test_z)
    except Exception as exc:
        warnings.warn(f"UMAP unavailable; PCA coordinates reused: {exc}")
        umap_train, umap_test = pca_train.copy(), pca_test.copy()
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tr, te, title in [
        (axes[0], pca_train, pca_test, "PCA of TabPFN embedding"),
        (axes[1], umap_train, umap_test, "UMAP of TabPFN embedding"),
    ]:
        for cls, color in [(0, "#2878B5"), (1, "#D1495B")]:
            ax.scatter(tr[y_train == cls, 0], tr[y_train == cls, 1], s=22, alpha=0.25, color=color, marker="o")
            ax.scatter(te[y_test == cls, 0], te[y_test == cls, 1], s=55, alpha=0.9, color=color, marker="x", label=f"test class {cls}")
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    base = Path(cfg["results_root"]) / "figures" / npz_path.stem
    _save_figure(fig, base)
    plot_data = pd.concat(
        [
            pd.DataFrame({"split": "train", "y": y_train, "pca1": pca_train[:, 0], "pca2": pca_train[:, 1], "umap1": umap_train[:, 0], "umap2": umap_train[:, 1]}),
            pd.DataFrame({"split": "test", "y": y_test, "pca1": pca_test[:, 0], "pca2": pca_test[:, 1], "umap1": umap_test[:, 0], "umap2": umap_test[:, 1]}),
        ],
        ignore_index=True,
    )
    plot_data.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    plt.close(fig)
    return base.with_suffix(".png")


def generate_standard_figures(cfg: dict[str, Any]) -> dict[str, Any]:
    predictions = consolidate_predictions(cfg)
    repeat_metrics = aggregate_repeat_metrics(cfg, predictions)
    fold_metrics = aggregate_fold_metrics(cfg, predictions)
    fold_variability = summarize_fold_variability(cfg, fold_metrics)
    summary = summarize_metrics(cfg, repeat_metrics)
    stability = compute_selection_stability(cfg)
    frequency = compute_selection_frequency(cfg)
    primary_metrics = primary_core_oof_metrics(cfg, predictions)
    bootstrap = paired_bootstrap_auroc(cfg, predictions)
    compatibility = compatibility_table(cfg, repeat_metrics)
    semantic = semantic_gain_table(cfg, repeat_metrics)
    tuning_decisions = consolidate_tuning_decisions(cfg)
    completion = experiment_completion_report(cfg)
    outputs: dict[str, Any] = {
        "prediction_rows": len(predictions),
        "repeat_metric_rows": len(repeat_metrics),
        "fold_metric_rows": len(fold_metrics),
        "fold_variability_rows": len(fold_variability),
        "summary_rows": len(summary),
        "stability_rows": len(stability),
        "selection_frequency_rows": len(frequency),
        "primary_oof_metric_rows": len(primary_metrics),
        "paired_bootstrap_rows": len(bootstrap),
        "compatibility_rows": len(compatibility),
        "semantic_rows": len(semantic),
        "nested_tuning_decision_rows": len(tuning_decisions),
        "completion": completion.to_dict("records"),
        "figures": [],
        "errors": [],
    }
    jobs = [
        (plot_metric_vs_sample_size, (cfg, repeat_metrics, "auroc", "core")),
        (plot_performance_heatmap, (cfg, repeat_metrics, "auroc", "core")),
        (plot_stability_heatmap, (cfg, stability, "mean_kuncheva")),
        (plot_stability_heatmap, (cfg, stability, "mean_jaccard")),
        (plot_resource_usage, (cfg, repeat_metrics, "core")),
        (plot_nested_tuning_choices, (cfg, tuning_decisions)),
    ]
    if (repeat_metrics["stage"] == "semantic").any():
        jobs.append((plot_name_condition_effect, (cfg, repeat_metrics, "auroc")))
    if not compatibility.empty:
        jobs.append((plot_compatibility_interaction, (cfg, compatibility)))
    if not bootstrap.empty:
        jobs.append((plot_paired_bootstrap_auroc, (cfg, bootstrap)))
    for dataset in cfg["datasets"]:
        jobs.append((plot_calibration, (cfg, predictions, dataset, "core", int(cfg["metrics"]["ece_bins"]))))
    for fn, args in jobs:
        try:
            outputs["figures"].append(str(fn(*args)))
        except Exception as exc:
            outputs["errors"].append({"function": fn.__name__, "error": repr(exc)})
    atomic_write_json(Path(cfg["results_root"]) / "logs" / "figure_generation_report.json", outputs)
    return outputs


__all__ = [
    "PIPELINE_VERSION",
    "load_config",
    "init_result_dirs",
    "environment_report",
    "validate_all_data",
    "semantic_mapping_report",
    "load_dataset",
    "seed_everything",
    "ensure_gemini_api_key",
    "get_llm_job_status",
    "wait_for_llm_job",
    "configured_outer_assignments",
    "llm_execution_plan",
    "submit_all_llm_jobs",
    "all_llm_job_statuses",
    "retrieve_and_prepare_all_llm_scores",
    "llm_score_cache_status",
    "reuse_llm_score_cache",
    "build_experiment_grid",
    "build_hybrid_alpha_grid",
    "build_hybrid_k_grid",
    "build_embedding_grid",
    "experiment_execution_plan",
    "expected_execution_counts",
    "experiment_completion_report",
    "grid_readiness",
    "run_grid",
    "consolidate_predictions",
    "aggregate_repeat_metrics",
    "aggregate_fold_metrics",
    "summarize_fold_variability",
    "summarize_metrics",
    "compute_selection_stability",
    "compute_selection_frequency",
    "consolidate_tuning_decisions",
    "compute_llm_run_stability",
    "primary_core_oof_predictions",
    "primary_core_oof_metrics",
    "paired_bootstrap_auroc",
    "compatibility_table",
    "semantic_gain_table",
    "run_embedding_case",
    "consolidate_embedding_metrics",
    "plot_embedding_case",
    "plot_performance_heatmap",
    "plot_stability_heatmap",
    "plot_resource_usage",
    "plot_hybrid_alpha_sensitivity",
    "plot_paired_bootstrap_auroc",
    "plot_nested_tuning_choices",
    "generate_standard_figures",
    "cross_fitted_data_scores",
    "mutual_information_scores",
]
