"""Kaggle/UCI 'heart_disease_uci.csv'(원본 통합본)를 본 프로젝트의 표준
heart.csv 스키마로 변환한다.

원본은 문자열 라벨(Male/Female, 'typical angina' 등)과 thalch/num 컬럼명,
그리고 id/dataset 부가 컬럼을 가진다. 이를 표준 UCI 코드(정수)와
age,sex,cp,trestbps,chol,fbs,restecg,thalch->thalach,exang,oldpeak,slope,ca,
thal,target 스키마로 매핑한다.

사용법:
    python data/convert_uci.py --input <원본csv> --output data/heart.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# 표준 UCI 코드 매핑 (원본 문자열 -> 정수 코드)
SEX_MAP = {"Female": 0, "Male": 1}
CP_MAP = {  # chest pain type 1..4 (UCI 원전 코드)
    "typical angina": 1,
    "atypical angina": 2,
    "non-anginal": 3,
    "asymptomatic": 4,
}
FBS_MAP = {False: 0, True: 1, "FALSE": 0, "TRUE": 1}
RESTECG_MAP = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
EXANG_MAP = {False: 0, True: 1, "FALSE": 0, "TRUE": 1}
SLOPE_MAP = {"upsloping": 1, "flat": 2, "downsloping": 3}
THAL_MAP = {"normal": 3, "fixed defect": 6, "reversable defect": 7}


def convert(in_path: Path, out_path: Path) -> None:
    df = pd.read_csv(in_path)

    out = pd.DataFrame()
    out["age"] = df["age"]
    out["sex"] = df["sex"].map(SEX_MAP)
    out["cp"] = df["cp"].map(CP_MAP)
    out["trestbps"] = df["trestbps"]
    out["chol"] = df["chol"]
    out["fbs"] = df["fbs"].map(FBS_MAP)
    out["restecg"] = df["restecg"].map(RESTECG_MAP)
    # 원본은 thalch, 프로젝트 스키마는 thalach
    out["thalach"] = df["thalch"]
    out["exang"] = df["exang"].map(EXANG_MAP)
    out["oldpeak"] = df["oldpeak"]
    out["slope"] = df["slope"].map(SLOPE_MAP)
    out["ca"] = df["ca"]
    out["thal"] = df["thal"].map(THAL_MAP)
    # 타깃: num(0..4) 그대로 둔다. 파이프라인의 binarize_target이 >0 -> 1 처리.
    out["target"] = df["num"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[convert] {in_path.name} -> {out_path} ({out.shape})")
    print(f"[convert] target 분포(원본 num):")
    print(out["target"].value_counts().sort_index().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default=str(Path(__file__).resolve().parent /
                               "heart_disease_uci.csv"),
                    help="원본 heart_disease_uci.csv 경로")
    ap.add_argument("--output",
                    default=str(Path(__file__).resolve().parent / "heart.csv"))
    args = ap.parse_args()
    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
