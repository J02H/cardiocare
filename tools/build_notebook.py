"""Build notebooks/01_eda_preprocessing.ipynb programmatically."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []

def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# CardioCare — EDA & Preprocessing (Task 1 / §5.1)

**Goal.** Understand the UCI Heart Disease data, then derive a *reusable*
preprocessing pipeline (no one-off cells). Every preprocessing decision below is
implemented in `src/preprocessing.py` so it re-applies identically to new data
inside an sklearn `Pipeline` — which also guarantees **no data leakage** (all
learnable transforms fit only on the training fold).

> **Ethical framing.** CardioCare is an *assistive* tool: it **informs, it does
> not decide**. Outputs are decision-support for a cardiologist, never an
> autonomous diagnosis.""")

code("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import (load_heart_data, binarize_target, clean_frame,
                           validate_input_ranges, build_preprocessor,
                           build_model_pipeline, split_X_y,
                           NUMERIC_FEATURES, CATEGORICAL_FEATURES)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
RANDOM_STATE = 42""")

md("## 1. Load the data")
code("""df_raw = load_heart_data(ROOT / "data" / "heart.csv")
print("shape:", df_raw.shape)
df_raw.head()""")

code("""df_raw.info()""")
code("""df_raw.describe()""")

md("""## 2. Target class distribution

The raw target is multiclass (0 = no disease, 1–4 = increasing severity). We
binarise it (`>0 -> 1`). We then inspect the class balance, because it dictates
**which metric we trust**.""")
code("""df = binarize_target(df_raw)
dist = df["target"].value_counts(normalize=True).sort_index()
print(dist)
ax = dist.plot(kind="bar", color=["#2980b9", "#c0392b"])
ax.set_xticklabels(["0 = no disease", "1 = disease"], rotation=0)
ax.set_ylabel("proportion"); ax.set_title("Target class distribution")
plt.tight_layout(); plt.show()""")

md("""**Why this matters for metric choice.** The classes are only mildly
imbalanced, but plain *accuracy* would still reward a model that simply favours
the majority class. Because a **false negative** (missing a real heart-disease
case) is the dangerous error in cardiology, we evaluate with
**balanced accuracy, precision, recall, F1 and the confusion matrix**, and we
ultimately select the final model with a **recall-first** criterion.""")

md("## 3. Missing values — per column and total")
code("""miss = df.isna().sum()
print("Missing per column:")
print(miss[miss > 0] if miss.any() else "None")
print("\\nTotal missing cells:", int(df.isna().sum().sum()))""")

md("## 4. Duplicate rows and empty columns")
code("""print("Exact duplicate rows:", int(df.duplicated().sum()))
empty_cols = [c for c in df.columns if df[c].isna().all()]
print("Fully-empty columns:", empty_cols or "None")""")

md("""## 5. Outliers on continuous features (boxplots + IQR)

We inspect `age, trestbps, chol, thalach, oldpeak`.""")
code("""cont = NUMERIC_FEATURES
fig, axes = plt.subplots(1, len(cont), figsize=(16, 4))
for ax, feat in zip(axes, cont):
    sns.boxplot(y=df[feat], ax=ax, color="#16a085")
    ax.set_title(feat)
plt.tight_layout(); plt.show()

def iqr_outliers(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum()), round(lo, 2), round(hi, 2)

print("IQR-based outlier counts:")
for feat in cont:
    n, lo, hi = iqr_outliers(df[feat])
    print(f"  {feat:9s}: {n:3d} outside [{lo}, {hi}]")""")

md("## 6. Clinical range validation")
code("""validate_input_ranges(df)""")

md("""## 7. Correlation snapshot (continuous features vs target)""")
code("""corr = df[cont + ["target"]].apply(pd.to_numeric, errors="coerce").corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation (continuous features + target)")
plt.tight_layout(); plt.show()""")

md("""## 8. What the EDA told us → preprocessing decisions

| Observation (EDA) | Decision (in `src/preprocessing.py`) |
|---|---|
| A few missing values in `ca`, `thal`, `chol` | **Impute, don't drop columns** — they are clinically informative. Numeric → **median** (robust to outliers); categorical → **most_frequent**. |
| Continuous features on very different scales (`chol` ~200s vs `oldpeak` ~0–6) | **StandardScaler** on numeric features (essential for LogisticRegression / SVC). |
| Coded categoricals (`cp`, `thal`, `slope`, `ca`, …) are **not ordinal-safe** | **OneHotEncoder(handle_unknown="ignore")** so unseen categories at inference don't crash. |
| Real but plausible outliers (high `chol`, `oldpeak`) | **Keep** them — they carry genuine clinical risk signal. We *validate* impossible values (range checks) instead of clipping real extremes. |
| 1 exact duplicate row, possible empty columns | Drop duplicates & all-NaN columns in `clean_frame()` (dataset hygiene, learns nothing → no leakage). |
| Mild class imbalance + costly false negatives | Evaluate with balanced accuracy / recall / F1 / confusion matrix; `class_weight="balanced"`; **recall-first** final selection. |

**Leakage guard.** All imputers, the scaler, the encoder, and the optional
feature selector live *inside* the `Pipeline` and are fit **after**
`train_test_split`, on the training fold only.""")

md("## 9. The reusable pipeline (proof it fits & transforms)")
code("""df_clean = clean_frame(binarize_target(load_heart_data(ROOT / "data" / "heart.csv")))
X, y = split_X_y(df_clean)

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

pre = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
Xt = pre.fit_transform(X_tr)            # fit on TRAIN ONLY -> no leakage
print("transformed train shape:", Xt.shape)

from sklearn.linear_model import LogisticRegression
pipe = build_model_pipeline(
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                       class_weight="balanced"),
    NUMERIC_FEATURES, CATEGORICAL_FEATURES)
pipe.fit(X_tr, y_tr)
print("held-out accuracy:", round(pipe.score(X_te, y_te), 3))""")

md("""**Next:** `src/train.py` trains ≥3 model families, logs everything to
MLflow, runs 5-fold CV + hyperparameter tuning, and selects the final model on a
recall-first clinical criterion. See `README.md`.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
out = ROOT / "notebooks" / "01_eda_preprocessing.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
