# LLMFS × TabPFN

LLM의 특징 의미 점수와 데이터 기반 특징 선택을 결합하여, 소표본 분류에서 TabPFN의 예측 성능을 비교하는 연구 프로젝트입니다. Colon, Golub, METABRIC 데이터셋을 대상으로 **Adaptive Quota Hybrid**와 기존 특징 선택 방법을 평가합니다.

실험 코드, 실행 노트북, 저장된 실험 결과와 KDMS 포스터를 포함합니다. 원본 학습 데이터와 TabPFN 모델 체크포인트는 이 저장소에 포함되어 있지 않습니다.

## 주요 방법

- **Text**: LLM 기반 특징 점수로 선택합니다.
- **Hybrid**: Text와 Data 순위를 가중 결합합니다.
- **Adaptive Quota Hybrid**: `round(alpha × k)`개의 Text quota와 나머지 Data quota를 유지하며, Gemini 반복 일치도와 데이터 점수 불확실성을 반영합니다. 중복으로 부족한 특징은 adaptive consensus 순위로 채웁니다.
- **비교 방법**: Data-only, Mutual Information, LASSO, Random(3 seeds), CatBoost-RFE, NoFS.
- **최종 분류기**: 로컬 체크포인트를 사용하는 TabPFN v2.5.

## 현재 실험 설계

| 항목 | 설정 |
|---|---|
| 데이터 및 outer-train 표본 조건 | Colon full/24, Golub full/24, METABRIC full/64 |
| Outer 평가 | Stratified 1회 × 5-fold CV |
| Inner 선택 | Outer-train 내부 stratified holdout, 약 75%/25% |
| 특징 수 후보 k | 32, 64, 128 |
| Hybrid alpha 후보 | 0.25, 0.50, 0.75 |
| Inner 평가기 | Logistic Regression, C=1.0 |
| 선택 기준 | 최고 AUROC의 0.005 이내 → log loss → 작은 k → alpha 0.5 근접 |
| Core 규모 | Outer TabPFN 330건, Inner 후보 평가 1,260건 |
| 통계 비교 | Paired bootstrap 5,000회, 95% CI, Holm 보정 |

Random은 특징 선택 seed 3개의 표본별 OOF 확률을 평균합니다. 소표본 조건은 평가 대상 전체 표본 수가 아니라 outer-train의 학습 표본 제한입니다. 작은 holdout에서는 설정 선택의 분산이 클 수 있습니다. 기본 실행 범위는 core이며, 의미 검증·모델 호환성·embedding 확장 함수는 코드에 남아 있습니다.

## 저장된 결과

아래는 포함된 `primary_core_oof_metrics.csv`에서 읽은 pooled OOF AUROC입니다. 이번 저장소 정리 과정에서 모델을 다시 학습한 결과는 아닙니다.

| 데이터 | 학습 표본 조건 | Adaptive Hybrid | Hybrid | Text | NoFS |
|---|---|---:|---:|---:|---:|
| colon | full | 0.8989 | 0.8477 | 0.8443 | 0.8756 |
| colon | 24 | 0.8523 | 0.7943 | 0.8136 | 0.8295 |
| golub | full | 0.9915 | 0.9957 | 0.9949 | 0.9932 |
| golub | 24 | 0.9932 | 0.9957 | 0.9804 | 0.9898 |
| metabric | full | 0.7570 | 0.7588 | 0.7638 | 0.7531 |
| metabric | 64 | 0.6470 | 0.6615 | 0.6146 | 0.6385 |

개별 조건의 수치만으로 모든 조건에서의 우월성이나 통계적 유의성을 단정하지 않습니다. 전체 방법 및 통계 비교는 아래 파일을 확인하세요.

- [공식 OOF 성능](experiment/RESULTS_LIGHT_1X5_HOLDOUT/metrics/primary_core_oof_metrics.csv)
- [표본별 OOF 예측](experiment/RESULTS_LIGHT_1X5_HOLDOUT/metrics/primary_core_oof_predictions.csv)
- [Paired bootstrap 비교](experiment/RESULTS_LIGHT_1X5_HOLDOUT/metrics/paired_bootstrap_auroc.csv)
- [특징 선택 안정성](experiment/RESULTS_LIGHT_1X5_HOLDOUT/metrics/selection_stability.csv)
- [결과 그림](experiment/RESULTS_LIGHT_1X5_HOLDOUT/figures)
- [KDMS 포스터](KDMS_포스터_최종본.pdf)

## 폴더 구성

```text
LLMFS_TabPFN/
├── README.md
├── KDMS_포스터_최종본.pdf
└── experiment/
    ├── LLMFS_TabPFN_Experiment.ipynb
    ├── EXPERIMENT_CODE/
    │   ├── llmfs_pipeline.py
    │   ├── experiment_config.json
    │   ├── requirements.txt
    │   ├── build_notebook.py
    │   └── README.md
    ├── RESULTS_LIGHT_1X5_HOLDOUT/   # 현재 경량 실험 결과
    └── RESULTS_NESTED_3X5/         # 이전 설계의 개발 결과/캐시
```

## 실행 준비

Python 3.11과 CUDA 사용이 가능한 GPU 환경을 준비합니다. 기존 코드 안내의 기준 커널은 Python 3.11.9입니다. CUDA에 맞는 PyTorch는 별도로 설치한 뒤 나머지 의존성을 설치합니다.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r experiment/EXPERIMENT_CODE/requirements.txt
```

`experiment/EXPERIMENT_CODE/experiment_config.json`의 다음 항목을 자신의 환경에 맞게 수정하세요. 상대 경로는 설정 파일이 있는 디렉터리를 기준으로 해석됩니다.

| 설정 | 준비 사항 |
|---|---|
| `data_root` | 전처리된 Colon/Golub/METABRIC 데이터 경로. 기본 `../../DATA`는 저장소 루트의 DATA를 가리키며 현재 미포함입니다. |
| `models.tabpfn.checkpoint` | 별도로 준비한 TabPFN v2.5 checkpoint 경로. 기본값은 기존 로컬 환경의 경로입니다. |
| `llm.reuse_scores_root` | 재사용할 LLM 캐시 경로. 기본값의 `2차` 폴더는 미포함이므로 사용 가능한 캐시 위치로 변경해야 합니다. |
| `results_root` | 결과 저장 위치. 기본값은 `../RESULTS_LIGHT_1X5_HOLDOUT`입니다. |
| `llm.model` | 기존 설정은 `gemini-3.5-flash-lite`이며, 새 API 실행 전 계정에서 해당 모델을 사용할 수 있는지 확인합니다. |

각 데이터셋 폴더에는 최소한 다음 파일이 필요합니다. 실제 컬럼 규격은 `llmfs_pipeline.py`의 `load_dataset`, `semantic_mapping_report`를 참고하세요.

```text
DATA/<colon|golub|metabric>/
├── data/X.csv                     # sample_id 및 특징 컬럼
├── data/y.csv                     # sample_id, target
├── metadata/feature_registry.csv  # 특징 순서, 유형, semantic_name 등
├── splits/outer_folds.csv
└── name_maps/                     # real.csv, anonymous.csv, permutation 매핑
```

X/y의 sample_id 순서와 X/registry의 feature_id 순서가 일치해야 합니다. 캐시는 모델·프롬프트·특징 이름 등의 검증을 통과해야 재사용됩니다. 검증 실패 시 노트북이 Gemini API 키를 요청하고 API 작업을 제출하므로 비용이 발생할 수 있습니다. 키는 저장소에 기록하지 마세요.

## 실행

노트북이 현재 작업 디렉터리의 `EXPERIMENT_CODE`를 찾으므로 `experiment` 디렉터리에서 Jupyter를 실행합니다.

```bash
cd experiment
jupyter lab LLMFS_TabPFN_Experiment.ipynb
```

준비한 Python/CUDA 커널을 선택한 뒤 셀을 순서대로 실행합니다. LLM 준비, smoke, core, 지표, 그림 플래그가 기본적으로 활성화되어 있습니다. `resume=true` 설정은 일치하는 완료 결과를 재사용합니다. 새로운 독립 실험은 `results_root`를 별도 디렉터리로 지정하세요.

자세한 설계는 [실험 코드 README](experiment/EXPERIMENT_CODE/README.md)를 참고하세요. 저장된 결과를 읽는 데는 원본 데이터나 GPU가 필요하지 않지만, 전체 재실행에는 위 외부 자원이 필요합니다.
