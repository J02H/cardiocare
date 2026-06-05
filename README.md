# CardioCare — 심장병 예측 종단간(End-to-End) 머신러닝 시스템

UCI Heart Disease 데이터셋의 일상적 임상 측정값으로부터 심장병 발병 가능성을
예측하는, 재현 가능한 종단간 머신러닝 시스템입니다.

> ⚕️ **알리되, 결정하지 않는다(inform, not decide).** CardioCare는 심장 전문의의
> 판단을 돕는 **보조** 의사결정 지원 도구이며, 절대 단독으로 진단·치료를 결정하지
> 않습니다. 모든 출력은 임상의에게 제시되는 확률값이고, 최종 책임은 임상의에게
> 있습니다. 시스템의 모든 설계, 특히 **거짓 음성(False Negative)** 최소화를 위한
> recall 우선 모델 선택은 이 원칙에서 비롯됩니다.

---

## 1. 프로젝트 개요

| 단계 | 내용 | 코드 |
|---|---|---|
| EDA·전처리 | 클래스 분포, 결측/중복/이상치, 누수 없는 sklearn `Pipeline` | `notebooks/01_eda_preprocessing.ipynb`, `src/preprocessing.py` |
| 학습·추적 | 3개 이상 모델 계열, 5-겹 CV, GridSearch 튜닝, MLflow 기록, recall 우선 선택 | `src/train.py` |
| 추론 | CSV 입력으로 predict + predict_proba, 로깅 포함 | `src/inference.py` |
| 모니터링·드리프트 | KS 검정 드리프트 탐지, 정상 vs 드리프트 균형 정확도 비교, 시계열 그래프 | `src/monitor.py` |
| 테스트 | unittest 7개(shape, 확률, 범위 검증, 결정론 등) | `tests/test_pipeline.py` |
| 패키징·CI | Docker 이미지 + GitHub Actions | `Dockerfile`, `.github/workflows/ci.yml` |
| 보고서 | 6쪽 PDF, 모든 루브릭 항목 포함 | `report.pdf` |

---

## 2. 폴더 구조

```
.
├── data/
│   ├── heart_disease_uci.csv   # 원본 UCI/Kaggle 데이터(약 920행)
│   ├── convert_uci.py          # 원본 -> 표준 스키마 변환 스크립트
│   ├── heart.csv               # 변환 완료 데이터(학습 입력)
│   ├── sample_input.csv        # 추론용 예시 8행 (target 열 없음)
│   └── download_data.py        # 네트워크에서 UCI 직접 받는 보조 경로
├── notebooks/
│   └── 01_eda_preprocessing.ipynb   # 실행 완료, 출력 포함
├── src/
│   ├── preprocessing.py        # 로드/이진화/검증 + sklearn Pipeline 빌더
│   ├── train.py                # 학습, CV, 튜닝, MLflow, 모델 선택
│   ├── inference.py            # CLI 배치 추론 + 로깅
│   └── monitor.py              # KS 드리프트 탐지 + 성능 비교
├── tests/
│   └── test_pipeline.py        # unittest 스위트
├── mlruns/                     # MLflow 아티팩트 (mlflow.db 함께 참조)
├── models/                     # final_model.joblib (학습 후 생성)
├── outputs/                    # 비교표, 선택 근거, 드리프트 리포트, 그림
├── logs/                       # inference.log
├── tools/                      # 노트북·보고서 생성 헬퍼(재현용)
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
├── report.pdf
└── README.md
```

---

## 3. 설치

**Python 3.10 이상**이 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate        # 윈도우: .venv\Scripts\activate
pip install -r requirements.txt
```

> **모듈 임포트.** `src/` 스크립트는 자기 디렉터리를 `sys.path`에 추가하므로
> `python src/train.py`가 그대로 동작합니다. 직접 패키지로 임포트할 때는
> 저장소 루트에서 `PYTHONPATH=.`를 사용하세요:
> ```bash
> PYTHONPATH=. python -m unittest discover -s tests
> ```

---

## 4. 데이터 준비

본 프로젝트는 **UCI Heart Disease 통합 데이터셋**(Kaggle 배포본
`heart_disease_uci.csv`, 약 920행)을 사용합니다. 원본은 문자열 라벨과
`thalch`/`num` 등 다른 컬럼명을 쓰므로, 표준 코드(정수)와 프로젝트 스키마로
변환하는 스크립트를 제공합니다. 원본 `data/heart_disease_uci.csv`가 함께
포함되어 있어 다음 한 줄로 변환을 재현할 수 있습니다:

```bash
python data/convert_uci.py
```

변환 결과인 `data/heart.csv`(14개 컬럼: age, sex, cp, ..., thal, target)도 이미
포함되어 있어, 변환 없이 바로 학습할 수도 있습니다.

원본 데이터 출처:
- Kaggle: <https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data>
- UCI 공식: <https://archive.ics.uci.edu/dataset/45/heart+disease>

> 다중 클래스 타깃(`num`, 0~4)은 파이프라인에서 자동으로 이진화됩니다
> (`0`=정상, `>0`=심장병). `data/download_data.py`는 네트워크 환경에서 UCI를
> 내려받는 대체 경로도 제공합니다.

---

## 5. 전체 재현 명령 (한 번에)

```bash
pip install -r requirements.txt
python data/download_data.py                                   # 데이터 준비
python src/train.py --data data/heart.csv                      # 학습 + MLflow + 선택
mlflow ui --backend-store-uri sqlite:///mlflow.db              # run 확인(브라우저)
python src/inference.py --input data/sample_input.csv \
       --model models/final_model.joblib \
       --output outputs/predictions.csv                        # 배치 추론
python src/monitor.py --data data/heart.csv \
       --model models/final_model.joblib                       # 드리프트 탐지
python -m unittest discover -s tests                           # 테스트
docker build -t cardiocare:1.0 .                               # 이미지 빌드
docker run --rm cardiocare:1.0                                 # 컨테이너 추론
```

---

## 6. 학습

```bash
python src/train.py --data data/heart.csv
# 선택: 파이프라인 내부의 RF 기반 특성 선택 활성화
python src/train.py --data data/heart.csv --use-feature-selection
```

**Logistic Regression, SVC, Random Forest** 세 계열을 학습하고, 선두 계열에 대해
**5-겹 교차검증**과 **GridSearchCV** 튜닝 run을 추가합니다. 산출물:

- `models/final_model.joblib` — 선택된 파이프라인
- `outputs/model_comparison.csv` — 전체 지표
- `outputs/final_model_rationale.txt` — recall 우선 선택 근거

모든 run은 **파라미터, 지표(균형 정확도·정밀도·재현율·F1·혼동행렬), 학습된 모델
아티팩트, `model_family` 태그**를 MLflow에 기록합니다.

### 실제 학습 결과 (UCI Heart Disease 통합 데이터, 약 920행)

| 모델 | 균형정확도 | 정밀도 | 재현율 | F1 | 거짓음성(FN) |
|---|---|---|---|---|---|
| **svc (최종 선택)** | 0.849 | 0.850 | 0.892 | 0.871 | 11 |
| svc_tuned | 0.849 | 0.850 | 0.892 | 0.871 | 11 |
| random_forest | 0.844 | 0.849 | 0.882 | 0.865 | 12 |
| logistic_regression | 0.817 | 0.829 | 0.853 | 0.841 | 15 |

거짓 음성이 가장 위험한 오류이므로 **재현율을 우선**한 임상 종합 점수
(재현율 0.5 + 균형 정확도 0.3 + F1 0.2)로 SVC를 최종 선택했습니다. SVC는
재현율 0.892, 거짓 음성 11건으로 가장 우수했습니다.

---

## 7. MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# 브라우저에서 http://127.0.0.1:5000 접속
```

**CardioCare** 실험에 4개 run(3개 계열 + 1개 튜닝)이 보이며, 각 run에 지표와 모델
아티팩트가 기록돼 있습니다.

> 참고: MLflow 3.x에서 파일 저장소(`./mlruns` 직접 사용)가 폐기되어, 추적
> 백엔드로 SQLite(`mlflow.db`)를 사용하고 아티팩트는 `mlruns/`에 저장합니다.

---

## 8. 추론

```bash
python src/inference.py \
  --input data/sample_input.csv \
  --model models/final_model.joblib \
  --output outputs/predictions.csv
```

`outputs/predictions.csv`에 `prediction`, `prob_no_disease`, `prob_disease`를
출력합니다. 모든 활동은 `logs/inference.log`에 기록됩니다. target 열이 없는
`sample_input.csv`만으로도 동작합니다.

---

## 9. 모니터링·드리프트

```bash
python src/monitor.py --data data/heart.csv --model models/final_model.joblib
```

- 계측된 추론 정보(타임스탬프, 모델 버전, 입력 shape, 예측값, 균형 정확도)를
  `logs/inference.log`에 기록.
- 연속형 특성을 이동(chol +30 및 분산 ×1.3, trestbps +15, oldpeak +0.5)시키고
  특성별 `scipy.stats.ks_2samp`를 수행해 `p < 0.05`를 플래그.
- 정상 vs 드리프트 균형 정확도 비교.
- `outputs/drift_report.csv`, `outputs/drift_summary.txt`,
  `outputs/performance_over_time.png` 생성.

### 실제 드리프트 결과 (UCI Heart Disease 통합 데이터)

| 특성 | KS 통계량 | p-value | 플래그 |
|---|---|---|---|
| age | 0.047 | 8.79e-01 | 정상 |
| trestbps | 0.398 | 2.91e-20 | **드리프트** |
| chol | 0.246 | 4.06e-08 | **드리프트** |
| thalach | 0.058 | 7.06e-01 | 정상 |
| oldpeak | 0.479 | 2.34e-29 | **드리프트** |

균형 정확도: 정상 **0.8485** → 드리프트 **0.8376** (감소폭 **0.0109**). 입력
드리프트와 성능 저하의 연관성이 확인됩니다.

---

## 10. 테스트

```bash
python -m unittest discover -s tests
```

(1) 예측 shape = 입력 행 수, (2) `predict_proba` ∈ [0,1] 및 행 합 ≈ 1,
(3) 임상 범위 검증(예: `chol ∈ [0,600]`), (4) 고정 시드 결정론, 그리고 전처리기
fit/transform, 타깃 이진화, `sample_input.csv` 스모크 테스트까지 **총 7개**를
포함합니다. **실제 실행 결과 7개 전부 통과(OK)**했습니다.

---

## 11. Docker

```bash
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

이미지는 고정 의존성을 설치하고 코드·`models/`·`data/`를 복사한 뒤,
`data/sample_input.csv`에 대한 추론을 기본 명령으로 실행합니다. 비밀값은 포함되지
않습니다.

---

## 12. CI (GitHub Actions)

`.github/workflows/ci.yml`은 모든 `push`와 `pull_request`에서 실행됩니다:
Python 3.10 설정 → 의존성 설치 → 데이터 준비 → 학습 → `unittest` → 추론 →
드리프트 모니터링. **main 브랜치를 green으로 유지**하는 것이 채점 대상입니다.

---

## 13. 윤리적 주의사항

- **알리되, 결정하지 않는다:** CardioCare는 임상의를 보조할 뿐 대체하지 않으며,
  진료를 거부·지연시키는 데 사용해서는 안 됩니다.
- **Human-in-the-loop:** 임상의가 모든 출력을 검토하고, 재학습 데이터의 라벨은
  확진된 임상 결과에서만 가져옵니다. 모델이 자신의 예측으로 학습하는 **폭주
  피드백 루프**를 방지합니다.
- **불확실성·공정성:** 출력은 확률로 제시되며 하위 집단 공정성을 모니터링합니다.
  데이터셋은 작고 오래되어 일반화에 한계가 있습니다.

---

## 14. AI 도구 사용 공개

AI 보조 도구(ChatGPT / Copilot / Claude 등)는 **보일러플레이트 작성과 디버깅
보조에만** 사용했습니다. 지표 선택, recall 우선 선택, 드리프트 방법론, 서빙 전략
등 모든 설계 결정과 최종 코드는 본인이 검토·이해했으며, 제출 코드 전체에 대해
본인이 책임집니다. 자세한 내용은 `report.pdf` 부록 A 참조.

---

## 15. 재현성 메모

- 모든 랜덤 시드 고정(`random_state=42`).
- 의존성 버전을 `requirements.txt`에 고정.
- 경로는 `pathlib.Path`로 처리하며, 데이터/모델이 없을 때 친절한 에러 메시지 출력.
- 전체 파이프라인이 README만으로 재현됩니다:
  `pip install` → `train.py` → `docker build` → `python -m unittest`.
