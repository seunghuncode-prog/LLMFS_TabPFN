# LLMFS × TabPFN 3차 경량 실험

실행 노트북: `../LLMFS_TabPFN_Experiment.ipynb`

1·2차 코드와 결과는 수정하지 않는다. 3차 출력은 전부
`../RESULTS_LIGHT_1X5_HOLDOUT`에 저장한다. 이전 3×5 개발 검사의 캐시와 섞이지 않는
독립 결과 폴더다.

## 축소된 실험 설계

- 데이터: Colon full/24, Golub full/24, METABRIC full/64.
- Outer 평가: stratified 1회 × 5-fold CV. 다섯 test fold를 모두 사용한다.
- Inner 선택: 각 outer-train에서 stratified holdout 1회.
- Inner train/validation 비율: 약 75%/25%.
- 후보 k: 32, 64, 128.
- Hybrid 후보 alpha: 0.25, 0.50, 0.75.
- Inner 공통 평가기: 고정 `C=1.0` Logistic Regression.
- 선택 규칙: 최고 inner AUROC의 0.005 이내 후보 → log loss → 작은 k → alpha 0.5 근접.
- Outer 모델: 로컬 checkpoint를 사용하는 TabPFN v2.5.
- 비교: Text, 기존 weighted-rank Hybrid, Adaptive Quota Hybrid, Data-only, MI,
  LASSO, Random 3 seeds, CatBoost-RFE, NoFS.
- 같은 outer fold의 모든 방법은 동일 모델 seed를 공유한다.
- Random 3개 seed는 특징 집합만 다르며 공식 OOF 확률은 표본별 평균한다.
- paired bootstrap 5,000회, 95% CI와 Holm 보정을 적용한다.

Inner holdout은 outer-test와 완전히 분리되어 있으므로 테스트 누수 없이 k와 alpha를 고른다.
단, 소표본 outer-train 24명에서는 validation이 약 6명뿐이라 inner 3-fold보다 설정 선택의
분산이 크다. 이는 계산량을 크게 줄이는 대신 감수하는 한계다.

## Adaptive Quota Hybrid

1. `round(alpha × k)`개를 Text quota로 보존한다.
2. 나머지를 Data quota로 보존한다.
3. Gemini 3회 일치도와 data-score holdout 불확실성으로 Text priority를 조정한다.
4. 두 quota가 겹쳐 부족한 자리는 adaptive consensus 순위로 채운다.

기존 weighted-rank Hybrid도 동일한 holdout과 후보 grid로 선택해 직접 비교한다.

## 실행 규모

Core 실행 한정:

- 비-Random 8개 방법: `8 × 6조건 × 5fold = 240`
- Random 3 seeds: `3 × 6조건 × 5fold = 90`
- Outer TabPFN 합계: **330건**
- Inner holdout Logistic 후보 평가: **1,260건**
- Smoke test: 2건

이전 3차 설계의 outer 3,315건에서 330건으로 약 90% 감소했다. Inner 후보 평가도
약 45,360건에서 1,260건으로 약 97% 감소했다.

이번 경량 실행에서는 의미 검증, TabPFN/CatBoost/Logistic compatibility, embedding을
기본 실행 범위에서 제외했다. 관련 함수는 코드에 유지되어 있으므로 Core 결과를 확인한 뒤
별도 확장할 수 있다.

## Gemini 비용

2차와 LLM 모델·프롬프트·특징 이름이 같으므로 검증된 `llm_raw`와 `llm_scores`만 복사한다.
label, fold, data score, 선택 특징과 예측은 재사용하지 않는다. 캐시 검증을 통과하면 API
추가 비용은 0이며, 실패할 때만 노트북이 API key를 요청한다.

## 결과 파일

- `metrics/primary_core_oof_metrics.csv`: 공식 pooled OOF 성능.
- `metrics/primary_core_oof_predictions.csv`: 공식 표본별 OOF 확률.
- `metrics/nested_tuning_decisions.csv`: fold별 선택 k, alpha와 holdout 성능.
- `metrics/paired_bootstrap_auroc.csv`: Adaptive Hybrid 기준 paired bootstrap.
- `metrics/selection_stability.csv`: 5-fold의 10개 특징 집합 쌍 안정성.
- `metrics/selection_frequency.csv`: 특징별 선택 빈도.
- `figures`: 성능, calibration, 안정성, 튜닝 선택, bootstrap, 자원 사용량 시각화.

## 실행 방법

1. CUDA가 활성화된 Python 3.11.9 커널을 선택한다.
2. `../LLMFS_TabPFN_Experiment.ipynb`를 연다.
3. 위에서부터 `모두 실행`한다.
4. 모든 실행 플래그는 이미 `True`이며 추가 수정은 필요 없다.
5. 중단 후 다시 실행하면 `resume=true`로 완료된 결과를 재사용한다.
