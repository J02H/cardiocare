"""UCI Heart Disease 데이터 준비 보조 스크립트.

기본 데이터(data/heart.csv)는 이미 저장소에 포함되어 있다. 이 스크립트는
네트워크가 연결된 환경에서 원본 UCI 데이터를 새로 받아 표준 스키마로
변환하고 싶을 때 사용하는 보조 경로다.

우선순위:
1. data/heart.csv 가 이미 있으면 아무 것도 하지 않는다(멱등).
2. ucimlrepo 로 UCI 원본(id=45)을 받아 표준 스키마로 변환한다.

원본 출처:
- UCI: https://archive.ics.uci.edu/dataset/45/heart+disease
- Kaggle: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
  (Kaggle 배포본 heart_disease_uci.csv 를 받은 경우에는 download_data.py 대신
   `python data/convert_uci.py --input heart_disease_uci.csv` 를 사용한다.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
HEART_CSV = DATA_DIR / "heart.csv"

# 표준 코드 매핑 (convert_uci.py 와 동일 규칙)
SEX_MAP = {"Female": 0, "Male": 1, 0: 0, 1: 1}
CP_MAP = {"typical angina": 1, "atypical angina": 2, "non-anginal": 3,
          "asymptomatic": 4, 1: 1, 2: 2, 3: 3, 4: 4}
FBS_MAP = {False: 0, True: 1, "FALSE": 0, "TRUE": 1, 0: 0, 1: 1}
RESTECG_MAP = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2,
               0: 0, 1: 1, 2: 2}
EXANG_MAP = {False: 0, True: 1, "FALSE": 0, "TRUE": 1, 0: 0, 1: 1}
SLOPE_MAP = {"upsloping": 1, "flat": 2, "downsloping": 3, 1: 1, 2: 2, 3: 3}
THAL_MAP = {"normal": 3, "fixed defect": 6, "reversable defect": 7,
            3: 3, 6: 6, 7: 7}


def _to_standard(df: pd.DataFrame) -> pd.DataFrame:
    rename = {"thalch": "thalach", "num": "target"}
    df = df.rename(columns=rename)
    out = pd.DataFrame()
    out["age"] = df["age"]
    out["sex"] = df["sex"].map(SEX_MAP)
    out["cp"] = df["cp"].map(CP_MAP)
    out["trestbps"] = df["trestbps"]
    out["chol"] = df["chol"]
    out["fbs"] = df["fbs"].map(FBS_MAP)
    out["restecg"] = df["restecg"].map(RESTECG_MAP)
    out["thalach"] = df["thalach"]
    out["exang"] = df["exang"].map(EXANG_MAP)
    out["oldpeak"] = df["oldpeak"]
    out["slope"] = df["slope"].map(SLOPE_MAP)
    out["ca"] = df["ca"]
    out["thal"] = df["thal"].map(THAL_MAP)
    out["target"] = df["target"]
    return out


def main() -> None:
    if HEART_CSV.exists():
        print(f"[download] {HEART_CSV} 이미 존재 - 아무 작업도 하지 않음.")
        return
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
    except Exception:
        raise SystemExit(
            "data/heart.csv 가 없고 ucimlrepo 도 설치돼 있지 않습니다.\n"
            "  pip install ucimlrepo  후 다시 실행하거나,\n"
            "  Kaggle 의 heart_disease_uci.csv 를 받아\n"
            "  python data/convert_uci.py --input heart_disease_uci.csv 를 실행하세요."
        )
    ds = fetch_ucirepo(id=45)
    df = pd.concat([ds.data.features.copy(), ds.data.targets.copy()], axis=1)
    out = _to_standard(df)
    out.to_csv(HEART_CSV, index=False)
    print(f"[download] UCI 원본 -> {HEART_CSV} ({out.shape})")


if __name__ == "__main__":
    main()
