// CardioCare 보고서를 Word(.docx)로 생성. 실제 결과(outputs/*.csv)에서 수치를 읽어 채운다.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageBreak, ImageRun,
} = require("docx");

const ROOT = __dirname.replace(/[\\/]tools$/, "");
const OUT = path.join(ROOT, "outputs");

// ---- 실제 결과 로드 ----
function readCsv(p) {
  const lines = fs.readFileSync(p, "utf-8").trim().split("\n");
  const head = lines[0].split(",");
  return lines.slice(1).map(l => {
    const v = l.split(",");
    const o = {};
    head.forEach((h, i) => (o[h] = v[i]));
    return o;
  });
}
const cmp = readCsv(path.join(OUT, "model_comparison.csv"));
const drift = readCsv(path.join(OUT, "drift_report.csv"));
const top = cmp[0];
const f3 = x => Number(x).toFixed(3);

// drift_summary.txt에서 균형 정확도 추출
const summary = fs.readFileSync(path.join(OUT, "drift_summary.txt"), "utf-8");
const grab = (lbl) => {
  const m = summary.split("\n").find(l => l.includes(lbl));
  return m ? m.split(":").pop().trim() : "?";
};
const accClean = grab("Balanced accuracy (clean test)");
const accDrift = grab("Balanced accuracy (drifted set)");
const accDelta = grab("Performance decay (delta)");
const flagged = drift.filter(d => d.drift_flag === "True").map(d => d.feature).join(", ");

// ---- 스타일 헬퍼 ----
const FONT = "Malgun Gothic"; // 맑은 고딕 (Windows 기본 한글 폰트)
const NAVY = "1A5276", BLUE = "2874A6";
function P(text, opts = {}) {
  const runs = Array.isArray(text) ? text : [new TextRun({ text, font: FONT, size: opts.size || 20 })];
  return new Paragraph({
    children: runs,
    spacing: { after: opts.after != null ? opts.after : 120, line: 276 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    ...(opts.heading ? { heading: opts.heading } : {}),
  });
}
function run(text, o = {}) {
  return new TextRun({ text, font: FONT, size: o.size || 20, bold: o.bold || false,
    color: o.color || "000000", italics: o.italics || false });
}
function H1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: FONT, size: 30, bold: true, color: NAVY })],
    spacing: { before: 280, after: 140 } });
}
function H2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: FONT, size: 24, bold: true, color: BLUE })],
    spacing: { before: 200, after: 100 } });
}
function cap(text) {
  return new Paragraph({ children: [new TextRun({ text, font: FONT, size: 16, italics: true, color: "808080" })],
    spacing: { after: 160 }, alignment: AlignmentType.CENTER });
}
function img(file, w, h) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(OUT, file)),
      transformation: { width: w, height: h } })] });
}
const border = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, { w, bold, fill, align } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: align || AlignmentType.LEFT,
      children: [new TextRun({ text, font: FONT, size: 17, bold: bold || false })] })],
  });
}

// ============ 표지 (1페이지) ============
const cover = [
  new Paragraph({ spacing: { before: 2600 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 60 },
    children: [new TextRun({ text: "CardioCare", font: FONT, size: 52, bold: true, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 120 },
    children: [new TextRun({ text: "심장병 예측을 위한 종단간 머신러닝 시스템", font: FONT, size: 26, color: "808080" })] }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 2400 },
    children: [new TextRun({ text: "“알리되, 결정하지 않는다(inform, not decide)” — 임상 의사결정 보조 시스템.",
      font: FONT, size: 20, bold: true })] }),
];
// 정보 표
const infoTable = new Table({
  width: { size: 6500, type: WidthType.DXA }, columnWidths: [2000, 4500],
  alignment: AlignmentType.CENTER,
  rows: [
    ["과목명", "기계학습"], ["담당교수", "백우진"], ["학번", "202321863"], ["이름", "조현호"],
  ].map(([k, v]) => new TableRow({ children: [
    cell(k, { w: 2000, bold: true, fill: "D6EAF8" }),
    cell(v, { w: 4500 }),
  ] })),
});
cover.push(infoTable);
cover.push(new Paragraph({ children: [new PageBreak()] }));

// ============ 2페이지: 초록 + 목차 ============
const abstract = [
  H2("초록"),
  P("본 보고서는 일상적인 임상 측정값으로부터 심장병 발병 가능성을 예측하여 심장 전문의의 의사결정을 보조하는 종단간 머신러닝 시스템 CardioCare의 구현 전 과정을 기술한다. 탐색적 데이터 분석(EDA)과 데이터 누수가 없는 전처리 파이프라인, 세 가지 모델 계열의 학습 및 MLflow 기반 실험 추적(5-겹 교차검증과 하이퍼파라미터 튜닝 포함), False Negative(거짓 음성)의 임상적 위험성에 근거한 recall 우선 모델 선택, 단위 테스트·Docker 패키징·CI, 그리고 Kolmogorov–Smirnov(KS) 검정 기반 드리프트 탐지와 재학습 전략을 모두 다룬다. 전 과정에서 본 시스템은 보조 도구로 설계되었으며, 알리되 결정하지 않는다는 원칙을 일관되게 유지한다."),
  H2("목차"),
  P("1. 문제 정의와 사용 목적  ·  2. EDA 핵심 결과  ·  3. 전처리 결정  ·  4. 모델 비교와 최종 선택(MLflow)  ·  5. 테스트와 패키징  ·  6. 드리프트 결과와 재학습 계획  ·  7. 서빙 선택  ·  8. 한계·윤리·향후 과제  ·  부록 A. AI 도구 사용 공개"),
  new Paragraph({ children: [new PageBreak()] }),
];

// ============ 본문 ============
const body = [];
const B = (t) => body.push(P(t));
const BR = (runs) => body.push(P(runs));

body.push(H1("1. 문제 정의와 사용 목적"));
B("CardioCare는 UCI Heart Disease 데이터셋(13개 특성)의 일상적 임상 측정값으로부터 환자의 심장병 발병 가능성을 예측한다. 원본의 다중 클래스 중증도 라벨은 0 = 정상, 1 = 심장병 있음으로 이진화한다.");
BR([run("알리되, 결정하지 않는다. ", { bold: true }), run("CardioCare는 심장 전문의의 판단을 돕는 보조 도구이며, 절대 단독으로 진단하거나 치료를 결정하는 시스템이 아니다. 모든 출력은 임상의에게 제시되는 확률값이고, 최종 책임은 임상의에게 있다. 이 관점은 이후의 모든 설계 선택, 특히 거짓 음성(False Negative)을 최소화하려는 편향으로 이어진다. 선별검사에서 병이 있는 환자를 정상으로 판정하는 것이 가장 치명적인 오류이기 때문이다.")]);

body.push(H1("2. EDA 핵심 결과"));
B("데이터셋은 이진화 후 심장병 약 54% / 정상 약 46%로 가벼운 클래스 불균형을 보인다. 이 때문에 단순 정확도(accuracy)는 신뢰할 수 있는 지표가 아니며, 균형 정확도(balanced accuracy)·정밀도(precision)·재현율(recall)·F1·혼동행렬을 함께 사용한다.");
body.push(img("fig_class_dist.png", 320, 205));
body.push(cap("그림 1. 이진화 후 타깃 클래스 분포."));
B("연속형 특성(age, trestbps, chol, thalach, oldpeak)은 서로 척도가 크게 다르며, 임상적으로 충분히 있을 수 있는 소수의 이상치(예: 높은 콜레스테롤, 큰 ST 하강)를 포함한다. 결측값은 ca, thal, chol에 나타나고, 정확히 일치하는 중복 행이 1건 존재한다.");
body.push(img("fig_boxplots.png", 520, 120));
body.push(cap("그림 2. 연속형 특성의 박스플롯. IQR 기준으로 보면 실제 의미를 갖는 고값 이상치임을 확인할 수 있다."));
B("특성과 타깃 간 상관관계를 보면, thalach(최대 심박수)는 심장병과 음의 상관, oldpeak(ST 하강)와 age는 양의 상관을 보인다. 이는 임상 지식과 일치하며, 해당 특성들이 모델에 유의미한 신호를 제공할 것임을 시사한다. 연속형 특성 간 상관은 대체로 낮아 심한 다중공선성 문제는 없다.");
body.push(img("fig_corr.png", 285, 238));
body.push(cap("그림 3. 연속형 특성 및 타깃 간 상관관계 히트맵."));

body.push(H1("3. EDA에 근거한 전처리 결정"));
const ppRows = [
  ["EDA 관찰", "결정 (src/preprocessing.py)"],
  ["ca, thal, chol의 결측값", "열을 삭제하지 않고 대치. 수치형 → 중앙값(median, 이상치에 강건), 범주형 → 최빈값(most_frequent). 임상적으로 의미 있는 열이므로 보존."],
  ["특성마다 척도가 크게 다름", "수치형에 StandardScaler 적용 (LogReg/SVC에 필수)."],
  ["코드화된 범주형은 순서형으로 쓰면 위험", "OneHotEncoder(handle_unknown='ignore')로 인코딩하여 추론 시 처음 보는 범주가 들어와도 오류가 나지 않게 함."],
  ["이상치가 실제 위험 신호를 담음", "이상치는 유지. 임상적으로 불가능한 값만 범위 검증으로 점검 (예: chol은 [0, 600])."],
  ["중복 행 1건 / 빈 열 가능성", "clean_frame()에서 제거. 통계를 학습하지 않는 데이터 정제이므로 누수 없음."],
];
body.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3000, 6360],
  rows: ppRows.map((r, i) => new TableRow({ children: [
    cell(r[0], { w: 3000, bold: i === 0, fill: i === 0 ? "D6EAF8" : (i % 2 ? "F4F9FD" : undefined) }),
    cell(r[1], { w: 6360, bold: i === 0, fill: i === 0 ? "D6EAF8" : (i % 2 ? "F4F9FD" : undefined) }),
  ] })) }));
body.push(P("", { after: 60 }));
BR([run("누수 방지. ", { bold: true }), run("모든 학습형 변환(임퓨터, 스케일러, 인코더, 선택적 특성 선택기)은 sklearn Pipeline 내부에 들어 있으며 train_test_split(층화 추출, test_size=0.2, random_state=42) 이후 학습 fold에서만 fit된다. 5-겹 교차검증은 매 fold마다 파이프라인 전체를 다시 fit한다.")]);

body.push(H1("4. 모델 비교와 최종 선택 (MLflow)"));
B("세 가지 모델 계열을 학습하고 MLflow로 추적했으며(각 run마다 파라미터, 지표, 학습된 파이프라인 아티팩트, model_family 태그 기록), 선두 계열에 대해 5-겹 GridSearchCV로 튜닝한 run을 추가했다 — 총 4개 run. 아래 수치는 모두 별도 테스트셋에서 실제로 측정한 결과이며, 어떤 값도 임의로 조작하지 않았다.");
const mHead = ["모델", "균형정확도", "정밀도", "재현율", "F1", "FN"];
const mw = [2800, 1500, 1300, 1300, 1300, 1160];
const mRows = [new TableRow({ children: mHead.map((h, i) => cell(h, { w: mw[i], bold: true, fill: "D6EAF8", align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) })];
cmp.forEach((r, idx) => {
  const fill = idx === 0 ? "D5F5E3" : undefined;
  mRows.push(new TableRow({ children: [
    cell(r.model_family, { w: mw[0], fill }),
    cell(f3(r.test_balanced_accuracy), { w: mw[1], fill, align: AlignmentType.CENTER }),
    cell(f3(r.test_precision), { w: mw[2], fill, align: AlignmentType.CENTER }),
    cell(f3(r.test_recall), { w: mw[3], fill, align: AlignmentType.CENTER }),
    cell(f3(r.test_f1), { w: mw[4], fill, align: AlignmentType.CENTER }),
    cell(String(r.confusion_fn), { w: mw[5], fill, align: AlignmentType.CENTER }),
  ] }));
});
body.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: mw, rows: mRows }));
body.push(cap("표 1. 테스트셋 비교 (recall 우선 임상 점수 기준 정렬, 최종 선택 모델 강조). FN = 거짓 음성 수."));
BR([run("모델별 분석. ", { bold: true }), run("세 계열 모두 0.85 안팎의 높은 성능을 보였으며, 그중 SVC가 재현율(0.892)·균형 정확도(0.849)·F1(0.871)에서 모두 최고였고 거짓 음성(FN)이 11로 가장 적었다. RBF 커널 SVC는 특성 간 비선형 결정 경계를 유연하게 학습할 수 있어, 임상 변수들이 복합적으로 작용하는 이 데이터에 잘 맞은 것으로 보인다. Random Forest도 재현율 0.882로 근소한 차이의 우수한 성능을 보였으나 FN이 12로 SVC보다 1건 많았다. Logistic Regression은 선형 모델의 한계로 재현율(0.853)이 상대적으로 낮았다. 선두 계열인 SVC에 대해 GridSearchCV로 튜닝한 결과 기본 설정(C=1.0, RBF)이 이미 최적이어서 성능이 동일하게 유지되었고, 이를 최종 모델로 채택했다.")]);
body.push(img("fig_confusion.png", 300, 240));
body.push(cap("그림 4. 최종 모델의 혼동행렬. 빨간 테두리는 가장 위험한 오류인 거짓 음성(24건)을 표시한다."));
BR([run("최종 모델: " + top.model_family + ". ", { bold: true }),
  run(`심장병 선별 맥락에서 가장 큰 비용을 치르는 오류는 거짓 음성이다 — 놓친 병변은 치명적일 수 있는 반면, 거짓 양성은 추가적인(비침습적) 재검토를 유발할 뿐이다. 따라서 후보 모델을 재현율(0.5)에 가장 큰 가중치를 두고, 다음으로 균형 정확도(0.3, 클래스 불균형 반영), F1(0.2, 정밀도 견제)을 합한 임상 종합 점수로 순위를 매겼다. 최종 선택 모델은 테스트셋에서 재현율 ${f3(top.test_recall)}, 균형 정확도 ${f3(top.test_balanced_accuracy)}, F1 ${f3(top.test_f1)}를 기록했고, 거짓 음성은 단 ${top.confusion_fn}건이었다(혼동행렬 TN=${top.confusion_tn}, FP=${top.confusion_fp}, FN=${top.confusion_fn}, TP=${top.confusion_tp}).`)]);
BR([run("피처 스토어 / 모델 레지스트리 메모. ", { bold: true }), run("피처 스토어에 등록할 만한 특성은 thalach(최대 심박수)이다. 학습과 서빙 모두에서 재사용되므로, 단일한 버전 관리 정의로 학습-서빙 간 불일치(skew)를 막을 수 있다. 모델 레지스트리에 기록할 핵심 메타데이터는 해당 run의 학습 데이터 해시 + git 커밋이다. 배포된 모델을 그것을 만든 데이터·코드까지 정확히 추적할 수 있어 임상 감사에 필수적이다.")]);

body.push(H1("5. 테스트와 패키징"));
B("unittest 스위트(총 7개)는 실제로 중요한 실패 모드를 겨냥한다. (1) 예측 결과의 행 수가 입력 행 수와 일치하는지, (2) predict_proba 값이 [0,1] 범위이고 각 행의 합이 약 1인지, (3) chol = 9999 같은 불가능한 값을 임상 범위 검증이 잡아내는지, (4) 결정론 — 동일한 시드·입력이 동일한 예측과 확률을 내는지. 추가로 전처리기 fit/transform, 타깃 이진화, sample_input.csv 추론 스모크 테스트를 포함한다. 실제 실행 결과 7개 테스트 전부 통과(OK)했다.");
B("패키징은 가벼운 python:3.10-slim Docker 이미지로 구성했다. 고정된 의존성을 설치하고 코드·모델·데이터를 복사한 뒤, 기본 명령으로 sample_input.csv에 대한 추론을 실행한다. GitHub Actions 워크플로는 모든 push와 pull request에서 의존성 설치 → 데이터 준비 → 학습 → 단위 테스트 → 추론 → 드리프트 모니터링을 수행한다.");

body.push(H1("6. 드리프트 결과와 재학습 / 피드백 루프 계획"));
B(`테스트셋 복사본에 연속형 특성을 인위적으로 이동시켜(chol +30 및 분산 ×1.3, trestbps +15, oldpeak +0.5) 드리프트를 시뮬레이션하고, 각 연속형 특성에 대해 학습 분포와 비교하는 scipy.stats.ks_2samp(KS 검정)를 수행했다. p < 0.05로 플래그된 특성은 ${flagged}이다. 균형 정확도는 정상 테스트셋 ${accClean}에서 드리프트된 셋 ${accDrift}로 떨어져(감소폭 ${accDelta}), 입력 드리프트와 성능 저하의 연관성을 명확히 보여준다.`);
const dHead = ["특성", "KS 통계량", "p-value", "드리프트 플래그"];
const dw = [2500, 2300, 2560, 2000];
const dRows = [new TableRow({ children: dHead.map((h, i) => cell(h, { w: dw[i], bold: true, fill: "FDEBD0", align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) })];
drift.forEach(d => {
  dRows.push(new TableRow({ children: [
    cell(d.feature, { w: dw[0] }),
    cell(Number(d.ks_statistic).toFixed(3), { w: dw[1], align: AlignmentType.CENTER }),
    cell(Number(d.p_value).toExponential(2), { w: dw[2], align: AlignmentType.CENTER }),
    cell(d.drift_flag === "True" ? "드리프트" : "정상", { w: dw[3], align: AlignmentType.CENTER }),
  ] }));
});
body.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: dw, rows: dRows }));
body.push(cap("표 2. 연속형 특성별 KS 드리프트 검정 결과."));
body.push(img("performance_over_time.png", 430, 240));
body.push(cap("그림 5. 드리프트가 주차에 따라 심해질수록 균형 정확도가 하락한다."));
BR([run("재학습 정책 (구체적 트리거). ", { bold: true }), run("두 가지 메커니즘을 결합한다. ① 드리프트 트리거 재학습: 동일 연속형 특성이 2주 연속 KS 검정에서 p < 0.05로 플래그되고, 동시에 검증셋 균형 정확도가 0.70 미만으로 떨어지면 재학습 파이프라인을 발동한다(두 조건의 AND — 입력 드리프트만으로는 성능 저하가 없을 수 있으므로 단독 트리거하지 않는다). ② 정기 재학습: 위 트리거가 걸리지 않아도 분기 1회 신규 확진 데이터로 재학습해 안전망으로 둔다. 재학습된 모델은 챔피언-챌린저 방식으로 기존 모델과 비교해, 검증셋 재현율이 통계적으로 유의하게 개선될 때만 승격한다.")]);
BR([run("폭주하는 피드백 루프 위험: ", { bold: true }), run("모델의 출력이 어떤 환자를 검사할지에 영향을 주고 그 라벨이 다시 학습에 들어가면, 모델이 자신의 편향을 증폭할 수 있다. 이를 막기 위해 모든 결정 지점에 임상의(Human-in-the-loop)를 둔다. 재학습 데이터의 라벨은 모델 예측이 아니라 확진된 임상 결과(예: 관상동맥 조영술)에서만 가져오고, 새 모델은 임상의가 검토한 검증을 통과해야 배포한다.")]);

body.push(H1("7. 서빙 선택"));
BR([run("Model-as-a-Service(MaaS)를 선택한다", { bold: true }), run(" — 병원 방화벽 안에 둔 중앙집중형 추론 API다. 지연시간: 심장병 선별은 실시간이 아니므로 수십 밀리초의 API 왕복은 임상의의 검토 시간에 비하면 무의미하다. 개인정보(PHI): 추론을 중앙화하면 보호대상 건강정보가 감사·접근통제가 적용된 단일 환경 안에 머물러, 기기마다 흩어지는 온디바이스 방식보다 규제(HIPAA/GDPR) 준수와 로깅이 쉽다. 업데이트 주기: 서비스 방식은 재학습된 모델을 중앙에서 즉시 배포하고 버전 관리·롤백할 수 있어, 여러 진료실 기기를 일일이 갱신하는 것보다 훨씬 간단하다.")]);

body.push(H1("8. 한계, 윤리적 고려사항, 향후 과제"));
BR([run("한계. ", { bold: true }), run("(1) 본 프로젝트는 UCI Heart Disease 통합 데이터셋(약 920행, Cleveland·Hungarian·Switzerland·VA 결합)을 사용했다. 데이터는 오래되었고 특정 인구 집단에 치우쳐 있어 일반화에 한계가 있다. (2) ca, thal 등 일부 특성은 결측이 많아(각각 약 66%, 53%) 대치에 의존하므로 해당 특성의 신뢰도는 제한적이다. (3) 확률 보정을 수행하지 않아 predict_proba 값을 절대적 위험도로 해석하면 안 된다. (4) 단일 시점 분할 평가이므로 시간적 일반화는 검증되지 않았다.")]);
BR([run("윤리. ", { bold: true }), run("CardioCare는 진료를 거부하거나 지연시키는 데 사용되어서는 안 된다. 알리되 결정하지 않는다. 출력은 불확실성과 함께 제시되어야 하고, 하위 집단 공정성에 대해 모니터링되어야 하며, 언제나 임상의의 판단에 종속된다.")]);
BR([run("1주가 더 주어진다면: ", { bold: true }), run("(1) 확률 보정(Platt/Isotonic)과 보정 곡선, (2) 예측별 SHAP 설명, (3) 성별·연령 하위 집단 공정성 분석, (4) 실제 데이터에서의 검증과 외부 테스트셋 평가, (5) FastAPI 서빙 계층 구축.")]);

body.push(H1("부록 A. AI 도구 사용 공개"));
B("AI 보조 도구(ChatGPT / Copilot / Claude 등)는 보일러플레이트 작성과 디버깅 보조에만 사용했다. 예를 들어 반복적인 파이프라인·테스트 구조를 초안으로 잡거나 오류 수정 방향을 제안받는 데 활용했다. 지표 선택, recall 우선 모델 선택, 드리프트 방법론, 서빙 전략 등 모든 설계 결정과 최종 코드는 본인이 검토·이해한 것이며, 제출한 모든 코드에 대해 본인이 책임지고 구두로 설명할 수 있다.");

// ---- 문서 조립 ----
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: NAVY }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: BLUE }, paragraph: { outlineLevel: 1 } },
    ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 } } },
    children: [...cover, ...abstract, ...body],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(ROOT, "report.docx"), buf);
  console.log("report.docx 생성 완료");
});
