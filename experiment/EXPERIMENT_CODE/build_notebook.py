"""Generate the clean, output-free notebook for the reduced third experiment."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# LLMFS × TabPFN 3차 경량 실험 — 1×5 CV + Inner Holdout

1·2차 코드는 수정하지 않습니다. 3차는 계산량을 줄이기 위해 반복 outer CV를 `1×5`로
축소하고, 각 outer-train 내부에서 stratified holdout 한 번으로 설정을 선택합니다.

- 후보 k: `32/64/128`
- Hybrid alpha: `0.25/0.5/0.75`
- 주 비교: Adaptive Hybrid, 기존 Hybrid, Text, Data-only, MI, LASSO,
  Random 3 seeds, CatBoost-RFE, NoFS
- 공식 평가: 5개 outer-test fold를 합친 pooled OOF
- 통계: paired bootstrap 5,000회, 95% CI, Holm 보정

기본 실행 범위는 **Core TabPFN 330건**입니다. 의미 검증·모델 compatibility·embedding은
이번 경량 실행에서 제외했습니다. 모든 실행 셀은 이미 활성화되어 있습니다.
""",
        "title",
    ),
    markdown(
        """## 0. 환경과 프로젝트 불러오기

Python 3.11, CUDA PyTorch, TabPFN 8.1.0과 로컬 v2.5 checkpoint를 사용합니다.
노트북을 `3차` 폴더에서 열고 `모두 실행`을 누르세요.
""",
        "environment-heading",
    ),
    code("""import sys
print(sys.executable)
""", "python-path"),
    code(
        """from pathlib import Path
import importlib
import json
import os
import sys
import pandas as pd

candidates = [
    Path.cwd().resolve(),
    Path.home() / 'Desktop' / '2026 캡스톤_2' / '2차 실험_학회 포스터' / '3차',
]
PROJECT_ROOT = next((p for p in candidates if (p / 'EXPERIMENT_CODE').is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('3차/EXPERIMENT_CODE 폴더를 찾지 못했습니다.')

CODE_ROOT = PROJECT_ROOT / 'EXPERIMENT_CODE'
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import llmfs_pipeline as exp
exp = importlib.reload(exp)
CFG = exp.load_config(CODE_ROOT / 'experiment_config.json')
PATHS = exp.init_result_dirs(CFG)
exp.seed_everything(int(CFG['seeds']['global_seed']))

print('Pipeline:', exp.PIPELINE_VERSION)
print('Project:', PROJECT_ROOT)
print('Data:', CFG['data_root'])
print('Results:', CFG['results_root'])
""",
        "load-project",
    ),
    code(
        """env = exp.environment_report(CFG, require_cuda=True)
display(env)
assert env['cuda_available'] is True
assert env['tabpfn_checkpoint']['is_file'] is True
print('CUDA / local TabPFN checkpoint: PASS')
""",
        "strict-environment",
    ),
    markdown(
        """## 1. 데이터·1×5 outer CV·내부 holdout 검사

소표본화는 outer-train에만 적용합니다. Inner holdout은 outer-train의 약 75%로 특징을
선택·학습하고 나머지 약 25%로 k·alpha를 고릅니다. Outer-test는 설정 선택에 사용하지 않습니다.
""",
        "data-heading",
    ),
    code(
        """data_summary = exp.validate_all_data(CFG)
mapping_report = exp.semantic_mapping_report(CFG, raise_on_failure=True)
display(data_summary)
display(mapping_report)

assert set(data_summary['dataset']) == {'colon', 'golub', 'metabric'}
assert (data_summary['outer_repeats'] == 1).all()
assert (data_summary['outer_folds'] == 5).all()
assert mapping_report['passed'].all()
assert CFG['nested_tuning']['strategy'] == 'stratified_holdout'
assert int(CFG['nested_tuning']['inner_folds']) == 1
assert CFG['nested_tuning']['k_grid'] == [32, 64, 128]
assert CFG['nested_tuning']['alpha_grid'] == [0.25, 0.5, 0.75]
assert CFG['execution']['enabled_stages'] == ['core']
print('Data / 1×5 outer CV / one inner holdout: PASS')
""",
        "validate-data",
    ),
    code(
        """execution_plan = exp.experiment_execution_plan(CFG)
display(execution_plan)

core_grid = exp.build_experiment_grid(CFG, 'core')
assert len(core_grid) == 330
assert int(execution_plan.loc[execution_plan['stage'] == 'TOTAL', 'model_runs'].iloc[0]) == 330
print('Core outer runs:', len(core_grid))
print('Expected inner Logistic fits: 1,260')
print('Paired fold-seed base:', CFG['seeds']['model_seed'])
""",
        "execution-plan",
    ),
    markdown(
        """## 2. Gemini score 준비

특징 이름과 task가 같은 2차 LLM 점수를 feature ID와 설정 기준으로 검증해 재사용합니다.
검증 실패 시에만 API key를 입력하고 기존 Gemini Batch를 실행합니다.
""",
        "gemini-heading",
    ),
    code(
        """RUN_LLM_PREPARE = True
if RUN_LLM_PREPARE:
    llm_cache = exp.reuse_llm_score_cache(CFG)
    display(llm_cache)
    if not llm_cache['ready']:
        display(exp.llm_execution_plan(CFG))
        _gemini_key = exp.ensure_gemini_api_key()
        display(exp.submit_all_llm_jobs(CFG, confirm='SUBMIT_GEMINI_BATCH_JOBS'))
        for condition, runs in [('real', int(CFG['llm']['real_runs'])), ('anonymous', int(CFG['llm']['anonymous_runs']))]:
            for run_index in range(1, runs + 1):
                status = exp.get_llm_job_status(CFG, condition, run_index)
                if status['state'] != 'JOB_STATE_SUCCEEDED':
                    status = exp.wait_for_llm_job(CFG, condition, run_index, timeout_hours=24.0)
                if status['state'] != 'JOB_STATE_SUCCEEDED':
                    raise RuntimeError(f'Gemini Batch failed: {condition}/run{run_index}: {status}')
        display(exp.retrieve_and_prepare_all_llm_scores(CFG))

llm_status = exp.llm_score_cache_status(CFG)
display(llm_status)
assert llm_status['valid'].all()
print('All LLM scores: READY')
""",
        "gemini-prepare",
    ),
    markdown("""## 3. 실행 전 smoke test""", "smoke-heading"),
    code(
        """RUN_SMOKE = True
if RUN_SMOKE:
    smoke_status = exp.run_grid(CFG, exp.build_experiment_grid(CFG, 'smoke'))
    display(smoke_status)
    assert (smoke_status['status'] == 'completed').all()
""",
        "run-smoke",
    ),
    markdown(
        """## 4. 주 분석 — Core TabPFN 330건

모든 방법은 동일한 inner holdout과 k 후보를 사용합니다. Hybrid 두 방식만 alpha 후보를
추가합니다. Inner holdout으로 선택된 설정을 outer-train 전체에 다시 적용하고 outer-test를
TabPFN으로 예측합니다.
""",
        "core-heading",
    ),
    code(
        """RUN_CORE = True
if RUN_CORE:
    core_status = exp.run_grid(CFG, core_grid)
    display(core_status['status'].value_counts(dropna=False))
    assert (core_status['status'] == 'completed').all()
""",
        "run-core",
    ),
    markdown(
        """## 5. OOF 성능·설정 선택·안정성·통계

Random은 세 selector seed의 확률을 표본별 평균합니다. Adaptive Hybrid를 기준으로 모든
비교 방법과 paired bootstrap을 수행합니다.
""",
        "metrics-heading",
    ),
    code(
        """RUN_METRICS = True
if RUN_METRICS:
    predictions = exp.consolidate_predictions(CFG)
    repeat_metrics = exp.aggregate_repeat_metrics(CFG, predictions)
    fold_metrics = exp.aggregate_fold_metrics(CFG, predictions)
    fold_variability = exp.summarize_fold_variability(CFG, fold_metrics)
    metric_summary = exp.summarize_metrics(CFG, repeat_metrics)
    primary_oof = exp.primary_core_oof_metrics(CFG, predictions)
    tuning_decisions = exp.consolidate_tuning_decisions(CFG)
    stability = exp.compute_selection_stability(CFG)
    selection_frequency = exp.compute_selection_frequency(CFG)
    bootstrap = exp.paired_bootstrap_auroc(CFG, predictions, reference_method='adaptive_hybrid')

    display(primary_oof.sort_values(['dataset', 'sample_regime', 'auroc'], ascending=[True, True, False]))
    display(tuning_decisions.loc[tuning_decisions['stage'] == 'core'].head(30))
    display(bootstrap.sort_values(['dataset', 'sample_regime', 'delta_auroc'], ascending=[True, True, False]))
    assert len(primary_oof) == 54
    assert len(bootstrap) == 48
    complete = stability.loc[(stability['stage'] == 'core') & stability['complete_outer_cv']]
    assert (complete['n_pairs'] == 10).all()
""",
        "aggregate-metrics",
    ),
    markdown(
        """## 6. 표·시각화 생성 및 완료 검사

공식 성능, calibration, 선택 안정성, inner holdout이 선택한 k·alpha, paired-bootstrap CI와
자원 사용량을 PNG·SVG·PDF·CSV로 저장합니다.
""",
        "figures-heading",
    ),
    code(
        """RUN_FIGURES = True
if RUN_FIGURES:
    figure_report = exp.generate_standard_figures(CFG)
    display(figure_report)
    if figure_report['errors']:
        print('Figure warnings:', figure_report['errors'])
    completion = exp.experiment_completion_report(CFG)
    display(completion)
    assert completion.loc[completion['stage'] == 'TOTAL', 'complete'].iloc[0]
""",
        "generate-figures",
    ),
    markdown(
        """## 7. 최종 출력 위치

- 공식 성능: `RESULTS_LIGHT_1X5_HOLDOUT/metrics/primary_core_oof_metrics.csv`
- inner 선택 결과: `nested_tuning_decisions.csv`
- paired bootstrap: `paired_bootstrap_auroc.csv`
- 안정성: `selection_stability.csv`, `selection_frequency.csv`
- 그림: `RESULTS_LIGHT_1X5_HOLDOUT/figures`
- 오류와 실행 로그: `RESULTS_LIGHT_1X5_HOLDOUT/logs`
""",
        "results-heading",
    ),
    code(
        """print('Experiment output root:', CFG['results_root'])
print('Primary metrics:', PATHS['metrics'] / 'primary_core_oof_metrics.csv')
print('Nested decisions:', PATHS['metrics'] / 'nested_tuning_decisions.csv')
print('Paired bootstrap:', PATHS['metrics'] / 'paired_bootstrap_auroc.csv')
print('Figures:', PATHS['figures'])
""",
        "results-paths",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3.11.9", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parent.parent / "LLMFS_TabPFN_Experiment.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output)
