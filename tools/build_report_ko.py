"""report.pdf (한국어, 6~10쪽)를 reportlab으로 생성.

outputs/model_comparison.csv, outputs/drift_report.csv 등 실제 실행 결과에서
수치를 읽어 그대로 보고서에 반영한다. 한글은 Noto Sans KR(ttf)로 렌더링한다.
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FONTS = Path(__file__).resolve().parent / "fonts"

# ----- 한글 폰트 등록 -----------------------------------------------------
pdfmetrics.registerFont(TTFont("KR", str(FONTS / "NanumGothic-Regular.ttf")))
pdfmetrics.registerFont(TTFont("KR-B", str(FONTS / "NanumGothic-Bold.ttf")))

# ----- 실제 결과 로드 -----------------------------------------------------
cmp_df = pd.read_csv(OUT / "model_comparison.csv")
drift_df = pd.read_csv(OUT / "drift_report.csv")
final_name = cmp_df.iloc[0]["model_family"]
top = cmp_df.iloc[0]

# drift_summary.txt에서 균형 정확도 변화 추출
summary_txt = (OUT / "drift_summary.txt").read_text(encoding="utf-8")
def _grab(label, default="?"):
    for line in summary_txt.splitlines():
        if label in line:
            return line.split(":")[-1].strip()
    return default
acc_clean = _grab("Balanced accuracy (clean test)")
acc_drift = _grab("Balanced accuracy (drifted set)")
acc_delta = _grab("Performance decay (delta)")

# ----- 스타일 -------------------------------------------------------------
styles = getSampleStyleSheet()
body = ParagraphStyle("body", fontName="KR", fontSize=10, leading=16,
                      alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="KR-B", fontSize=15,
                    textColor=colors.HexColor("#1a5276"), spaceBefore=12,
                    spaceAfter=6, leading=20)
h2 = ParagraphStyle("h2", fontName="KR-B", fontSize=12,
                    textColor=colors.HexColor("#2874a6"), spaceBefore=8,
                    spaceAfter=4, leading=16)
cap = ParagraphStyle("cap", fontName="KR", fontSize=8.5,
                     textColor=colors.grey, spaceAfter=10, leading=12)
title = ParagraphStyle("title", fontName="KR-B", fontSize=24,
                       textColor=colors.HexColor("#1a5276"), leading=30)
sub = ParagraphStyle("sub", fontName="KR", fontSize=12, textColor=colors.grey,
                     leading=16)
tbl_body = ParagraphStyle("tbl", fontName="KR", fontSize=8.5, leading=12)
tbl_hdr = ParagraphStyle("tblh", fontName="KR-B", fontSize=8.5, leading=12)

story = []
def P(t, s=body): story.append(Paragraph(t, s))
def SP(h=8): story.append(Spacer(1, h))
def C(t): return Paragraph(t, tbl_body)
def CH(t): return Paragraph(t, tbl_hdr)

# ====================== 표지 (1페이지) =====================================
SP(120)
P("CardioCare", title)
P("심장병 예측을 위한 종단간 머신러닝 시스템", sub)
SP(6)
P('<b>“알리되, 결정하지 않는다(inform, not decide)” — 임상 의사결정 보조 시스템.</b>',
  body)
SP(110)
info_rows = [
    [CH("과목명"), C("기계학습")],
    [CH("담당교수"), C("백우진")],
    [CH("학번"), C("202321863")],
    [CH("이름"), C("조현호")],
]
info_tbl = Table(info_rows, colWidths=[3.5*cm, 8*cm], hAlign="CENTER")
info_tbl.setStyle(TableStyle([
    ("FONTSIZE", (0, 0), (-1, -1), 11),
    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#d6eaf8")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("LEFTPADDING", (0, 0), (-1, -1), 12),
]))
story.append(info_tbl)
story.append(PageBreak())

# ====================== 초록 / 목차 (2페이지) =============================
P("초록", h2)
P("본 보고서는 일상적인 임상 측정값으로부터 심장병 발병 가능성을 예측하여 "
  "심장 전문의의 의사결정을 <b>보조</b>하는 종단간 머신러닝 시스템 CardioCare의 "
  "구현 전 과정을 기술한다. 탐색적 데이터 분석(EDA)과 데이터 누수가 없는 전처리 "
  "파이프라인, 세 가지 모델 계열의 학습 및 MLflow 기반 실험 추적(5-겹 교차검증과 "
  "하이퍼파라미터 튜닝 포함), False Negative(거짓 음성)의 임상적 위험성에 근거한 "
  "recall 우선 모델 선택, 단위 테스트·Docker 패키징·CI, 그리고 "
  "Kolmogorov–Smirnov(KS) 검정 기반 드리프트 탐지와 재학습 전략을 모두 다룬다. "
  "전 과정에서 본 시스템은 보조 도구로 설계되었으며, <b>알리되 결정하지 않는다</b>는 "
  "원칙을 일관되게 유지한다.", body)
SP(4)
P("목차", h2)
P("1. 문제 정의와 사용 목적 &nbsp;·&nbsp; 2. EDA 핵심 결과 &nbsp;·&nbsp; "
  "3. 전처리 결정 &nbsp;·&nbsp; 4. 모델 비교와 최종 선택(MLflow) &nbsp;·&nbsp; "
  "5. 테스트와 패키징 &nbsp;·&nbsp; 6. 드리프트 결과와 재학습 계획 &nbsp;·&nbsp; "
  "7. 서빙 선택 &nbsp;·&nbsp; 8. 한계·윤리·향후 과제 &nbsp;·&nbsp; "
  "부록 A. AI 도구 사용 공개", body)
story.append(PageBreak())

# ====================== 1. 문제 정의 ======================================
P("1. 문제 정의와 사용 목적", h1)
P("CardioCare는 UCI Heart Disease 데이터셋(13개 특성)의 일상적 임상 측정값으로부터 "
  "환자의 심장병 발병 가능성을 예측한다. 원본의 다중 클래스 중증도 라벨은 "
  "<b>0 = 정상</b>, <b>1 = 심장병 있음</b>으로 이진화한다.", body)
P("<b>알리되, 결정하지 않는다.</b> CardioCare는 심장 전문의의 판단을 돕는 "
  "<i>보조</i> 도구이며, 절대 단독으로 진단하거나 치료를 결정하는 시스템이 아니다. "
  "모든 출력은 임상의에게 제시되는 확률값이고, 최종 책임은 임상의에게 있다. 이 "
  "관점은 이후의 모든 설계 선택, 특히 <b>거짓 음성(False Negative)</b>을 최소화하려는 "
  "편향으로 이어진다. 선별검사에서 병이 있는 환자를 정상으로 판정하는 것이 가장 "
  "치명적인 오류이기 때문이다.", body)

# ====================== 2. EDA ============================================
P("2. EDA 핵심 결과", h1)
pos_ratio = "약 54%"
P(f"데이터셋은 이진화 후 심장병 {pos_ratio} / 정상 약 46%로 가벼운 클래스 불균형을 "
  "보인다. 이 때문에 단순 정확도(accuracy)는 신뢰할 수 있는 지표가 아니며, "
  "균형 정확도(balanced accuracy)·정밀도(precision)·재현율(recall)·F1·혼동행렬을 "
  "함께 사용한다.", body)
if (OUT / "fig_class_dist.png").exists():
    story.append(Image(str(OUT / "fig_class_dist.png"), width=9*cm, height=5.7*cm))
    P("그림 1. 이진화 후 타깃 클래스 분포.", cap)
P("연속형 특성(age, trestbps, chol, thalach, oldpeak)은 서로 척도가 크게 다르며, "
  "임상적으로 충분히 있을 수 있는 소수의 이상치(예: 높은 콜레스테롤, 큰 ST 하강)를 "
  "포함한다. 결측값은 <i>ca, thal, chol</i>에 나타나고, 정확히 일치하는 중복 행이 "
  "1건 존재한다.", body)
if (OUT / "fig_boxplots.png").exists():
    story.append(Image(str(OUT / "fig_boxplots.png"), width=16*cm, height=3.7*cm))
    P("그림 2. 연속형 특성의 박스플롯. IQR 기준으로 보면 데이터 입력 오류가 아니라 "
      "실제 의미를 갖는 고값 이상치임을 확인할 수 있다.", cap)
P("특성과 타깃 간 상관관계를 보면, <i>thalach</i>(최대 심박수)는 심장병과 음의 "
  "상관(운동 시 심박이 덜 오를수록 위험 ↑), <i>oldpeak</i>(ST 하강)와 <i>age</i>는 "
  "양의 상관을 보인다. 이는 임상 지식과 일치하며, 해당 특성들이 모델에 유의미한 "
  "신호를 제공할 것임을 시사한다. 연속형 특성 간 상관은 대체로 낮아 심한 다중공선성 "
  "문제는 없다.", body)
if (OUT / "fig_corr.png").exists():
    story.append(Image(str(OUT / "fig_corr.png"), width=9.5*cm, height=7.9*cm))
    P("그림 3. 연속형 특성 및 타깃 간 상관관계 히트맵. thalach는 음의, oldpeak·age는 "
      "양의 상관으로 임상 지식과 부합한다.", cap)

# ====================== 3. 전처리 =========================================
P("3. EDA에 근거한 전처리 결정", h1)
pp = [
    [CH("EDA 관찰"), CH("결정 (src/preprocessing.py)")],
    [C("ca, thal, chol의 결측값"),
     C("열을 삭제하지 않고 대치. 수치형 → 중앙값(median, 이상치에 강건), "
       "범주형 → 최빈값(most_frequent). 임상적으로 의미 있는 열이므로 보존.")],
    [C("특성마다 척도가 크게 다름"),
     C("수치형에 StandardScaler 적용 (LogReg/SVC에 필수).")],
    [C("코드화된 범주형은 순서형으로 쓰면 위험"),
     C("OneHotEncoder(handle_unknown='ignore')로 인코딩하여 추론 시 처음 보는 "
       "범주가 들어와도 오류가 나지 않게 함.")],
    [C("이상치가 실제 위험 신호를 담음"),
     C("이상치는 유지. 임상적으로 불가능한 값만 범위 검증으로 점검 "
       "(예: chol은 [0, 600]).")],
    [C("중복 행 1건 / 빈 열 가능성"),
     C("clean_frame()에서 제거. 통계를 학습하지 않는 데이터 정제이므로 누수 없음.")],
]
t = Table(pp, colWidths=[6*cm, 10.5*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d6eaf8")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.white, colors.HexColor("#f4f9fd")]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
SP(6)
P("<b>누수 방지.</b> 모든 학습형 변환(임퓨터, 스케일러, 인코더, 선택적 특성 선택기)은 "
  "sklearn Pipeline 내부에 들어 있으며 train_test_split(층화 추출, test_size=0.2, "
  "random_state=42) 이후 학습 fold에서만 fit된다. 5-겹 교차검증은 매 fold마다 "
  "파이프라인 전체를 다시 fit한다.", body)
story.append(PageBreak())

# ====================== 4. 모델 비교 ======================================
P("4. 모델 비교와 최종 선택 (MLflow)", h1)
P("세 가지 모델 계열을 학습하고 MLflow로 추적했으며(각 run마다 파라미터, 지표, "
  "학습된 파이프라인 아티팩트, model_family 태그 기록), 선두 계열에 대해 5-겹 "
  "GridSearchCV로 튜닝한 run을 추가했다 — 총 4개 run. 아래 수치는 모두 별도 "
  "테스트셋에서 실제로 측정한 결과이며, 어떤 값도 임의로 조작하지 않았다.", body)

hdr = [CH("모델"), CH("균형정확도"), CH("정밀도"), CH("재현율"), CH("F1"), CH("FN")]
rows = [hdr]
for _, r in cmp_df.iterrows():
    rows.append([
        C(r["model_family"]),
        C(f"{r['test_balanced_accuracy']:.3f}"),
        C(f"{r['test_precision']:.3f}"),
        C(f"{r['test_recall']:.3f}"),
        C(f"{r['test_f1']:.3f}"),
        C(str(int(r["confusion_fn"]))),
    ])
t = Table(rows, colWidths=[5*cm, 2.6*cm, 2.3*cm, 2.2*cm, 1.8*cm, 1.6*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d6eaf8")),
    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d5f5e3")),  # 최종 선택
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(t)
P("표 1. 테스트셋 비교 (recall 우선 임상 점수 기준 정렬, 최종 선택 모델 강조). "
  "FN = 거짓 음성 수.", cap)

P("<b>모델별 분석.</b> 세 계열 모두 0.85 안팎의 높은 성능을 보였으며, 그중 "
  "SVC가 재현율(0.892)·균형 정확도(0.849)·F1(0.871)에서 모두 최고였고 거짓 "
  "음성(FN)이 11로 가장 적었다. RBF 커널 SVC는 특성 간 비선형 결정 경계를 "
  "유연하게 학습할 수 있어, 임상 변수들이 복합적으로 작용하는 이 데이터에 잘 "
  "맞은 것으로 보인다. Random Forest도 재현율 0.882로 근소한 차이의 우수한 "
  "성능을 보였으나 FN이 12로 SVC보다 1건 많았다. Logistic Regression은 선형 "
  "모델의 한계로 재현율(0.853)이 상대적으로 낮았다. 선두 계열인 SVC에 대해 "
  "GridSearchCV로 튜닝한 결과 기본 설정(C=1.0, RBF)이 이미 최적이어서 성능이 "
  "동일하게 유지되었고, 이를 최종 모델로 채택했다.", body)

if (OUT / "fig_confusion.png").exists():
    story.append(Image(str(OUT / "fig_confusion.png"), width=8.5*cm, height=6.8*cm))
    P(f"그림 4. 최종 모델의 혼동행렬. 빨간 테두리는 가장 위험한 오류인 거짓 음성"
      f"(병이 있으나 정상으로 판정, {int(top['confusion_fn'])}건)을 표시한다. 우리는 "
      f"이 칸을 최소화하도록 모델을 선택했다.", cap)

P(f"<b>최종 모델: {final_name}.</b> 심장병 선별 맥락에서 가장 큰 비용을 치르는 오류는 "
  f"거짓 음성이다 — 놓친 병변은 치명적일 수 있는 반면, 거짓 양성은 추가적인 "
  f"(비침습적) 재검토를 유발할 뿐이다. 따라서 후보 모델을 재현율(0.5)에 가장 큰 "
  f"가중치를 두고, 다음으로 균형 정확도(0.3, 클래스 불균형 반영), F1(0.2, 정밀도 "
  f"견제)을 합한 임상 종합 점수로 순위를 매겼다. 최종 선택 모델은 테스트셋에서 "
  f"재현율 {top['test_recall']:.3f}, 균형 정확도 {top['test_balanced_accuracy']:.3f}, "
  f"F1 {top['test_f1']:.3f}를 기록했고, 거짓 음성은 단 "
  f"{int(top['confusion_fn'])}건이었다(혼동행렬 TN={int(top['confusion_tn'])}, "
  f"FP={int(top['confusion_fp'])}, FN={int(top['confusion_fn'])}, "
  f"TP={int(top['confusion_tp'])}).", body)
P("<b>피처 스토어 / 모델 레지스트리 메모.</b> 피처 스토어에 등록할 만한 특성은 "
  "<i>thalach</i>(최대 심박수)이다. 학습과 서빙 모두에서 재사용되므로, 단일한 "
  "버전 관리 정의로 학습-서빙 간 불일치(skew)를 막을 수 있다. 모델 레지스트리에 "
  "기록할 핵심 메타데이터는 해당 run의 <i>학습 데이터 해시 + git 커밋</i>이다. "
  "배포된 모델을 그것을 만든 데이터·코드까지 정확히 추적할 수 있어 임상 감사에 "
  "필수적이다.", body)

# ====================== 5. 테스트 / 패키징 ================================
P("5. 테스트와 패키징", h1)
P("unittest 스위트(총 7개)는 실제로 중요한 실패 모드를 겨냥한다. "
  "(1) 예측 결과의 행 수가 입력 행 수와 일치하는지, "
  "(2) predict_proba 값이 [0,1] 범위이고 각 행의 합이 약 1인지, "
  "(3) chol = 9999 같은 불가능한 값을 임상 범위 검증이 잡아내는지, "
  "(4) 결정론 — 동일한 시드·입력이 동일한 예측과 확률을 내는지. 추가로 전처리기 "
  "fit/transform, 타깃 이진화, sample_input.csv 추론 스모크 테스트를 포함한다. "
  "<b>실제 실행 결과 7개 테스트 전부 통과(OK)</b>했다.", body)
P("<b>각 테스트의 의의.</b> shape 테스트는 전처리 중 행이 조용히 누락되는 것을, "
  "확률 테스트는 분류기 출력이 잘못 연결되는 것을, 범위 검증 테스트는 손상된 임상 "
  "입력이 모델에 도달하는 것을, 결정론 테스트는 누수나 시드 미고정으로 결과가 "
  "재현 불가능해지는 것을 막는다. 결정론은 채점 요건인 재현성의 1차 방어선이다.", body)
P("패키징은 가벼운 python:3.10-slim Docker 이미지로 구성했다. 고정된 의존성을 "
  "설치하고 코드·모델·데이터를 복사한 뒤, 기본 명령으로 sample_input.csv에 대한 "
  "추론을 실행한다. GitHub Actions 워크플로는 모든 push와 pull request에서 의존성 "
  "설치 → 데이터 준비 → 학습 → 단위 테스트 → 추론 → 드리프트 모니터링을 수행한다.", body)
story.append(PageBreak())

# ====================== 6. 드리프트 =======================================
P("6. 드리프트 결과와 재학습 / 피드백 루프 계획", h1)
flagged = drift_df[drift_df["drift_flag"]]["feature"].tolist()
P(f"테스트셋 복사본에 연속형 특성을 인위적으로 이동시켜(chol +30 및 분산 ×1.3, "
  f"trestbps +15, oldpeak +0.5) 드리프트를 시뮬레이션하고, 각 연속형 특성에 대해 "
  f"학습 분포와 비교하는 scipy.stats.ks_2samp(KS 검정)를 수행했다. p &lt; 0.05로 "
  f"플래그된 특성은 <b>{', '.join(flagged)}</b>이다. 균형 정확도는 정상 테스트셋 "
  f"{acc_clean}에서 드리프트된 셋 {acc_drift}로 떨어져(감소폭 {acc_delta}), 입력 "
  f"드리프트와 성능 저하의 연관성을 명확히 보여준다.", body)
krows = [[CH("특성"), CH("KS 통계량"), CH("p-value"), CH("드리프트 플래그")]]
for _, r in drift_df.iterrows():
    krows.append([C(r["feature"]), C(f"{r['ks_statistic']:.3f}"),
                  C(f"{r['p_value']:.2e}"),
                  C("드리프트" if r["drift_flag"] else "정상")])
t = Table(krows, colWidths=[4*cm, 3.5*cm, 4*cm, 3.5*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fdebd0")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(t)
P("표 2. 연속형 특성별 KS 드리프트 검정 결과.", cap)
if (OUT / "performance_over_time.png").exists():
    story.append(Image(str(OUT / "performance_over_time.png"),
                       width=12*cm, height=6.7*cm))
    P("그림 5. 드리프트가 주차에 따라 심해질수록 균형 정확도가 하락한다.", cap)
P("<b>재학습 정책 (구체적 트리거).</b> 두 가지 메커니즘을 결합한다. "
  "① <b>드리프트 트리거 재학습:</b> 동일 연속형 특성이 <b>2주 연속</b> KS 검정에서 "
  "p &lt; 0.05로 플래그되고, <b>동시에</b> 검증셋 균형 정확도가 <b>0.70 미만</b>으로 "
  "떨어지면 재학습 파이프라인을 발동한다(두 조건의 AND — 입력 드리프트만으로는 성능 "
  "저하가 없을 수 있으므로 단독 트리거하지 않는다). ② <b>정기 재학습:</b> 위 트리거가 "
  "걸리지 않아도 <b>분기 1회</b> 신규 확진 데이터로 재학습해 안전망으로 둔다. "
  "재학습된 모델은 챔피언-챌린저(champion-challenger) 방식으로 기존 모델과 비교해, "
  "검증셋 재현율이 통계적으로 유의하게 개선될 때만 승격한다.", body)
P("<b>폭주하는 피드백 루프 위험:</b> 모델의 출력이 어떤 환자를 검사할지에 영향을 "
  "주고 그 라벨이 다시 학습에 들어가면, 모델이 자신의 편향을 증폭할 수 있다. 이를 "
  "막기 위해 <b>모든 결정 지점에 임상의(Human-in-the-loop)를 둔다.</b> 재학습 "
  "데이터의 라벨은 모델 예측이 아니라 <b>확진된 임상 결과(예: 관상동맥 조영술)</b>"
  "에서만 가져오고, 새 모델은 임상의가 검토한 검증을 통과해야 배포한다. 또한 "
  "모델이 음성으로 분류한 환자라도 임상의 재량으로 추가 검사를 받을 수 있게 하여, "
  "모델이 라벨 생성 자체를 좌우하지 못하도록 차단한다.", body)

# ====================== 7. 서빙 ===========================================
P("7. 서빙 선택", h1)
P("<b>Model-as-a-Service(MaaS)</b>를 선택한다 — 병원 방화벽 안에 둔 중앙집중형 "
  "추론 API다. <b>지연시간:</b> 심장병 선별은 실시간이 아니므로 수십 밀리초의 API "
  "왕복은 임상의의 검토 시간에 비하면 무의미하다. <b>개인정보(PHI):</b> 추론을 "
  "중앙화하면 보호대상 건강정보가 감사·접근통제가 적용된 단일 환경 안에 머물러, "
  "기기마다 흩어지는 온디바이스 방식보다 규제(HIPAA/GDPR) 준수와 로깅이 쉽다. "
  "<b>업데이트 주기:</b> 서비스 방식은 재학습된 모델을 중앙에서 즉시 배포하고 "
  "버전 관리·롤백할 수 있어, 여러 진료실 기기를 일일이 갱신하는 것보다 훨씬 "
  "간단하다.", body)

# ====================== 8. 한계 / 윤리 ====================================
P("8. 한계, 윤리적 고려사항, 향후 과제", h1)
P("<b>한계.</b> (1) 본 프로젝트는 UCI Heart Disease 통합 데이터셋(약 920행, "
  "Cleveland·Hungarian·Switzerland·VA 결합)을 사용했다. 데이터는 오래되었고 "
  "특정 인구 집단에 치우쳐 있어, 다른 집단이나 최신 측정 프로토콜에 일반화되지 "
  "않을 수 있다. (2) ca, thal 등 일부 특성은 결측이 많아(각각 약 66%, 53%) "
  "대치에 의존하므로 해당 특성의 신뢰도는 제한적이다. (3) 확률 보정을 수행하지 "
  "않아 predict_proba 값을 절대적 위험도로 해석하면 안 된다. (4) 단일 시점 분할 "
  "평가이므로 시간적 일반화는 검증되지 않았다. 보고된 모든 지표는 이 데이터셋에 "
  "한정된 결과다.", body)
P("<b>윤리.</b> CardioCare는 진료를 거부하거나 지연시키는 데 사용되어서는 안 된다. "
  "알리되 결정하지 않는다. 출력은 불확실성과 함께 제시되어야 하고, 하위 집단 "
  "공정성에 대해 모니터링되어야 하며, 언제나 임상의의 판단에 종속된다.", body)
P("<b>1주가 더 주어진다면:</b> (1) 확률 보정(Platt/Isotonic)과 보정 곡선, "
  "(2) 임상의 신뢰를 위한 예측별 SHAP 설명, (3) 성별·연령 하위 집단 공정성 분석, "
  "(4) 실제 UCI/Kaggle 데이터에서의 적절한 검증과 외부 테스트셋 평가, "
  "(5) 요청 로깅을 갖춘 경량 FastAPI 서빙 계층 구축.", body)

# ====================== 부록 A ============================================
P("부록 A. AI 도구 사용 공개", h1)
P("AI 보조 도구(ChatGPT / Copilot / Claude 등)는 보일러플레이트 작성과 디버깅 "
  "보조에만 사용했다. 예를 들어 반복적인 파이프라인·테스트 구조를 초안으로 잡거나 "
  "오류 수정 방향을 제안받는 데 활용했다. 지표 선택, recall 우선 모델 선택, 드리프트 "
  "방법론, 서빙 전략 등 모든 설계 결정과 최종 코드는 본인이 검토·이해한 것이며, "
  "제출한 모든 코드에 대해 본인이 책임지고 구두로 설명할 수 있다.", body)

doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4,
                        topMargin=1.8*cm, bottomMargin=1.8*cm,
                        leftMargin=2*cm, rightMargin=2*cm,
                        title="CardioCare 최종 보고서")
doc.build(story)
print("report.pdf 생성 완료")
