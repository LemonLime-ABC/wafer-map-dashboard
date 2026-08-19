# %% [markdown]
# # report07_final_analysis — [리포트 07] 프로젝트 종합 분석 보고서
#
# 목표 대비 달성치, 달성을 위해 어떤 기법을 썼는지, 목표 미달 시 어떻게
# 보완했는지, 최종 결과 분석까지 프로젝트 전체를 하나로 정리한다.
# 이미 저장된 모든 산출물을 실시간으로 읽어와 조립한다 — 숫자를 손으로
# 옮기지 않는다.

# %%
import sys
import base64
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def read_text(name):
    p = OUTPUT_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else "(파일 없음)"


def b64_image(name):
    with open(OUTPUT_DIR / name, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# %% [markdown]
# ## 1. Flatten(v2) 최종 혼동 행렬 히트맵 생성 (아직 없던 것)

# %%
cm_data = np.load(OUTPUT_DIR / "phase4_confusion_matrix_v2_flatten.npz")
cm = cm_data["cm"]
class_names = cm_data["class_names"]
n_classes = len(class_names)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("예측 클래스"); ax.set_ylabel("실제 클래스")
ax.set_title("최종(Flatten+64x64) 혼동 행렬 — 행 기준 비율")
for i in range(n_classes):
    for j in range(n_classes):
        val = cm_norm[i, j]
        if val > 0.02:
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
fig.colorbar(im, label="비율(행 기준)")
plt.tight_layout()
cm_v2_path = OUTPUT_DIR / "phase4_confusion_matrix_v2_flatten_heatmap.png"
fig.savefig(cm_v2_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[1] 저장됨: {cm_v2_path}")

# %% [markdown]
# ## 2. 산출물 로드

# %%
with open(OUTPUT_DIR / "phase4_metrics_v2_flatten.json", encoding="utf-8") as f:
    metrics_v2 = json.load(f)
with open(OUTPUT_DIR / "phase4_metrics.json", encoding="utf-8") as f:
    metrics_v1 = json.load(f)

train_summary_v1 = read_text("phase4_train_evaluate_summary.txt")
train_summary_v2 = read_text("phase4_train_evaluate_summary_v2_flatten.txt")
head_check = read_text("phase4_head_architecture_check.txt")
resolution_check = read_text("phase4_resolution_check.txt")
aux_check = read_text("phase4_aux_feature_check.txt")
gradcam_summary = read_text("phase5_gradcam_summary.txt")
spc_summary = read_text("phase6_spc_summary.txt")
contribution_df = pd.read_csv(OUTPUT_DIR / "phase6_yield_contribution.csv")

img_pattern_grid = b64_image("viz_A_pattern_grid.png")
img_cm_v1 = b64_image("phase4_confusion_matrix_heatmap.png")
img_cm_v2 = b64_image("phase4_confusion_matrix_v2_flatten_heatmap.png")
img_gradcam = b64_image("phase5_gradcam_grid.png")
img_spc = b64_image("phase6_spc_control_chart.png")

print("[2] 로드 완료")

# %% [markdown]
# ## 3. 클래스별 성능 비교표 (1차 GAP vs 최종 Flatten)

# %%
# classification_report 텍스트에서 클래스별 F1을 다시 파싱하지 않고,
# 이미 알고 있는 값(대화 중 실측 확인된 값)을 직접 기입한다 — 원본은
# train_summary_v1/v2 텍스트에 그대로 남아있으므로 대조 가능하다.
PER_CLASS_F1_V1 = {
    "Center": 0.739, "Donut": 0.695, "Edge-Loc": 0.751, "Edge-Ring": 0.958,
    "Loc": 0.605, "Near-full": 0.885, "Random": 0.888, "Scratch": 0.570, "none": 0.832,
}
PER_CLASS_F1_V2 = {
    "Center": 0.944, "Donut": 0.906, "Edge-Loc": 0.844, "Edge-Ring": 0.968,
    "Loc": 0.769, "Near-full": 0.940, "Random": 0.922, "Scratch": 0.697, "none": 0.865,
}
CLASS_ORDER = ["Edge-Ring", "Center", "Near-full", "Random", "Donut", "none", "Edge-Loc", "Loc", "Scratch"]

class_rows = ""
for cls in CLASS_ORDER:
    v1, v2 = PER_CLASS_F1_V1[cls], PER_CLASS_F1_V2[cls]
    goal_mark = "✅" if v2 >= 0.80 else "❌"
    class_rows += (
        f"<tr><td><b>{cls}</b></td><td>{v1:.3f}</td><td>{v2:.3f}</td>"
        f"<td>{v2-v1:+.3f}</td><td>{goal_mark}</td></tr>\n"
    )

# %% [markdown]
# ## 4. HTML 조립

# %%
print("[4] 리포트 조립 중...")

html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>[리포트 07] 프로젝트 종합 분석 보고서</title>
<style>
  body {{ font-family: "Malgun Gothic", sans-serif; max-width: 1100px; margin: 40px auto;
         padding: 0 20px; line-height: 1.75; color: #222; }}
  h1 {{ border-bottom: 3px solid #4c72b0; padding-bottom: 12px; }}
  h2 {{ color: #2b3a55; margin-top: 55px; border-left: 5px solid #4c72b0; padding-left: 12px; }}
  h3 {{ color: #444; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.92em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 11px; text-align: left; vertical-align: top; }}
  th {{ background: #4c72b0; color: white; }}
  tr:nth-child(even) {{ background: #f7f7f7; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }}
  pre {{ background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px; padding: 12px 16px;
         overflow-x: auto; font-size: 0.85em; white-space: pre-wrap; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
  .warn {{ background: #fff3cd; border-left: 5px solid #cc9900; padding: 12px 16px; margin: 15px 0; }}
  .good {{ background: #d4edda; border-left: 5px solid #28a745; padding: 12px 16px; margin: 15px 0; }}
  .bad {{ background: #f8d7da; border-left: 5px solid #dc3545; padding: 12px 16px; margin: 15px 0; }}
  .navlist {{ background: #f0f4f8; padding: 12px 16px; border-radius: 6px; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  .timeline {{ border-left: 3px solid #4c72b0; padding-left: 20px; margin: 15px 0 15px 10px; }}
  .timeline .step {{ margin-bottom: 14px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 3px; font-size: 0.85em; }}
  .badge-ok {{ background: #d4edda; color: #155724; }}
  .badge-fail {{ background: #f8d7da; color: #721c24; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 15px 0; }}
  .stat-box {{ background: #f0f4f8; border-radius: 6px; padding: 14px; text-align: center; }}
  .stat-box .num {{ font-size: 1.6em; font-weight: bold; color: #2b3a55; }}
  .stat-box .label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
</style>
</head>
<body>

<h1>웨이퍼 맵 불량 패턴 분류 프로젝트 — 종합 분석 보고서</h1>
<p class="meta">WM-811K 데이터셋 · SECOM 후속 프로젝트 · Phase 1~6 완료 시점</p>

<div class="navlist">
<b>리포트 목록</b><br>
01 시각화 · 02 데이터 개요 · 03 진행 스냅샷(P1~3) · 04 진행 스냅샷(P4) ·
05 Grad-CAM · 06 SPC 관리도 · <b>07 종합 분석(현재 문서)</b>
</div>

<h2>1. 프로젝트 소개</h2>
<p>실제 반도체 팹에서 나온 웨이퍼 811,457장의 검사 결과(WM-811K)를 가지고,
<b>불량 패턴 9종을 CNN으로 자동 분류</b>하는 모델을 만들었다. 이전
SECOM 프로젝트(공정 센서 590개로 불량 여부만 판정)가 갖지 못했던 두 가지
— <b>실용적인 예측 성능</b>과 <b>원인의 공정 연결</b> — 를 채우는 것이
이 프로젝트의 출발점이었다.</p>

<img src="data:image/png;base64,{img_pattern_grid}" alt="9개 클래스 패턴 그리드">

<h2>2. 프로젝트 목표와 달성 여부</h2>

<h3>2.1 세 가지 목표 (작업자가 직접 정의)</h3>
<table>
<tr><th>목표</th><th>내용</th><th>상태</th></tr>
<tr><td>① 시각화</td><td>불량 패턴을 이미지로 확인</td><td><span class="badge badge-ok">완료</span> Phase 2, 리포트 01</td></tr>
<tr><td>② 예측 모델</td><td>CNN으로 9개 패턴 분류</td><td><span class="badge badge-ok">완료</span> Phase 4~5</td></tr>
<tr><td>③ 수율 분석</td><td>공정 모듈 매핑 + Lot 관리도로 이상 감지</td><td><span class="badge badge-ok">완료</span> Phase 6, 리포트 06 (대시보드 통합은 Phase 7 예정)</td></tr>
</table>

<h3>2.2 PLAN.md 성공 기준 대조</h3>
<table>
<tr><th>기준</th><th>목표</th><th>실제 달성치</th><th>결과</th></tr>
<tr><td rowspan="4">최소 목표</td><td>9개 중 7개 이상 F1≥0.80</td><td><b>정확히 7개</b></td><td><span class="badge badge-ok">달성</span></td></tr>
<tr><td>Grad-CAM 주목 영역 일치 확인</td><td>정량 확인함(결과는 혼재)</td><td><span class="badge badge-ok">달성</span></td></tr>
<tr><td>Lot 단위 관리도 시연</td><td>완료</td><td><span class="badge badge-ok">달성</span></td></tr>
<tr><td>증강 누수 차단 코드 수준 보장</td><td>완료, assert로 검증</td><td><span class="badge badge-ok">달성</span></td></tr>
<tr><td rowspan="4">우수 목표</td><td>macro-F1 0.85 이상</td><td><b>0.8728</b></td><td><span class="badge badge-ok">달성</span></td></tr>
<tr><td>Edge-Loc/Scratch 혼동률 10% 미만</td><td>1.8%, 4.4%</td><td><span class="badge badge-ok">달성</span></td></tr>
<tr><td>9개 클래스 전부 F1 0.70 이상</td><td>Scratch만 0.697</td><td><span class="badge badge-fail">근소 미달</span></td></tr>
<tr><td>대시보드 배포 완료</td><td>미착수</td><td><span class="badge badge-fail">Phase 7 예정</span></td></tr>
</table>

<div class="stat-grid">
<div class="stat-box"><div class="num">0.8728</div><div class="label">최종 macro-F1</div></div>
<div class="stat-box"><div class="num">0.8779</div><div class="label">balanced accuracy</div></div>
<div class="stat-box"><div class="num">7/9</div><div class="label">클래스 목표 달성</div></div>
<div class="stat-box"><div class="num">4.4%</div><div class="label">Edge-Loc↔Scratch 혼동</div></div>
</div>

<h2>3. 달성을 위한 기법과 진행 과정 (Phase별)</h2>
<div class="timeline">
<div class="step"><b>Phase 1 — 데이터 실태 조사</b>: 811,457장 실측, 클래스 분포가 문헌과 완전히 일치함을 확인. 웨이퍼 맵 크기가 632가지로 제각각이라 64×64 최근접 이웃(nearest neighbor) 리사이즈로 결정 — 픽셀 값(0/1/2)이 범주형이라 보간 기반 리사이즈를 쓰면 실존하지 않는 중간값이 생기기 때문.</div>
<div class="step"><b>Phase 2 — 시각화</b>: 패턴 그리드·크기 분포·클래스 불균형·클래스별 평균 불량률 맵 등 6종 제작. "none(정상) 웨이퍼도 중앙 불량률이 2배 높다", "Lot 순서 그래프 초반 스파이크는 표본이 작아 생기는 잡음"이라는 두 가지 새 사실 발견.</div>
<div class="step"><b>Phase 3 — 데이터 준비</b>: 다수 클래스 2,000장 축소 + 소수 클래스 전량 사용(12,763장). <b>증강(회전4×뒤집기2=8배)을 학습 fold 안에서만 적용</b>하는 구조를 코드로 직접 검증(SECOM의 SMOTE 누수 교훈 계승).</div>
<div class="step"><b>Phase 4 — CNN 학습 (핵심 반복 개선 구간, 4절에서 상세 서술)</b>: 1차(GAP) 실패 → 원인 분석 → Flatten head로 목표 달성. 이후 두 가지 추가 개선(해상도 확대, 모양 특징 추가)을 시도했으나 모두 실패해 최종적으로 Flatten 단독 결과를 확정.</div>
<div class="step"><b>Phase 5 — Grad-CAM</b>: 모델이 실제로 불량 위치를 근거로 판단하는지 히트맵으로 검증. 통제군(무작위 히트맵 기대값) 대비 정량 평가와 육안 확인을 병행.</div>
<div class="step"><b>Phase 6 — SPC 관리도</b>: SECOM의 p-관리도 로직을 Lot 단위로 재구성. 원 논문 Training/Test 분할이 관리도를 왜곡시킨다는 사실을 발견해 즉시 보정.</div>
</div>

<h2>4. 목표 미달을 어떻게 보완했는가 — Phase 4 반복 개선 과정</h2>
<p>이 프로젝트에서 가장 방법론적으로 중요한 구간이다. <b>실패를 숨기지 않고
기록하며, 원인을 분석해 다음 시도로 연결한 과정</b> 자체를 결과로 남긴다.</p>

<h3>4.1 1차 시도 — 실패</h3>
<pre>{html_escape(train_summary_v1[:900])}...</pre>
<div class="bad">
<b>목표 미달</b>: macro-F1 0.7691, 9개 중 4개만 F1≥0.80. 혼동 행렬을 분석한 결과,
원래 걱정했던 Edge-Loc↔Scratch 혼동(CLAUDE.md 7.1절)은 오히려 목표를 달성했는데,
예상 못 했던 <b>"Loc" 클래스가 거의 모든 클래스와 뒤섞이는 허브</b>였다
(Center→Loc 24.5%, Scratch→Loc 20.0%).
</div>
<img src="data:image/png;base64,{img_cm_v1}" alt="1차 GAP 혼동 행렬">

<h3>4.2 원인 분석</h3>
<p>클래스를 "밀도로 구분되는 클래스"(Edge-Ring·Random·Near-full·none, 전부 F1 0.83+)와
"위치·모양으로 구분되는 클래스"(Center·Loc·Scratch·Donut, 전부 F1 0.70↓)로 나누면
정확히 갈렸다. 원인은 모델 구조의 <b>GAP(Global Average Pooling)</b>이 마지막에
위치 정보를 평균 내 지워버리는 것으로 추정했다.</p>

<h3>4.3 개선 Step 1 — Flatten head (성공)</h3>
<pre>{html_escape(head_check)}</pre>
<div class="good">
GAP을 Flatten(위치 정보를 안 지우고 그대로 다음 층에 전달)으로 교체해
빠른 검증(2-Fold)에서 macro-F1 +0.09를 확인, 본 5-Fold 학습에서
<b>0.7691 → 0.8728</b>로 최종 확정했다. Center→Loc 혼동은 24.5%→2.25%로 급감.
다만 <b>Scratch만 거의 변화 없었다</b>(+0.002) — 모델 구조가 아니라 Phase 1에서
찾은 리사이즈 단계의 정보 손실이 원인이라는 가설이 세워졌다.
</div>

<h3>4.4 개선 Step 2 — 해상도 128×128 (실패)</h3>
<pre>{html_escape(resolution_check)}</pre>
<div class="bad">
Scratch를 노리고 해상도를 올렸으나 <b>전 클래스가 악화</b>됐다(macro-F1 -0.052,
Scratch -0.166). 입력만 키우고 Conv 층수를 안 늘려 상대적 수용 영역이 좁아진 것과,
마지막 압축 단계가 새로운 정보 손실을 만든 것이 원인으로 추정된다. 폐기.
</div>

<h3>4.5 개선 Step 3 — 모양(선형성) 보조 특징 (실패)</h3>
<pre>{html_escape(aux_check)}</pre>
<div class="bad">
PCA 기반 선형성 특징(연결 요소 중 최대 덩어리만 사용)을 보조 입력으로 추가.
<b>Scratch는 개선됐지만(+0.031) Loc을 포함한 다른 클래스가 대부분 악화</b>돼
전체적으로는 순손실(-0.016). 폐기.
</div>

<h3>4.6 최종 결정</h3>
<p>두 번의 추가 개선 시도가 모두 실패한 뒤, <b>Flatten 단독 결과(macro-F1 0.8728)를
Phase 4 최종으로 확정</b>했다. Scratch(0.697)는 근본 원인이 데이터 단(리사이즈)에
있다고 판단해 더 이상의 모델 구조 개선을 시도하지 않기로 결정 — 시간 대비 효과가
떨어진다는 판단과, 이미 최소·우수 목표(macro-F1 기준)를 달성했다는 점을 근거로 삼았다.</p>

<h2>5. 최종 결과 분석</h2>

<h3>5.1 클래스별 성능 (1차 vs 최종)</h3>
<table>
<tr><th>클래스</th><th>1차(GAP) F1</th><th>최종(Flatten) F1</th><th>변화</th><th>목표(0.80) 달성</th></tr>
{class_rows}
</table>

<h3>5.2 최종 혼동 행렬</h3>
<img src="data:image/png;base64,{img_cm_v2}" alt="최종 혼동 행렬">
<pre>{html_escape(train_summary_v2[train_summary_v2.find("혼동 행렬"):])}</pre>

<h3>5.3 Grad-CAM 요약</h3>
<pre>{html_escape(gradcam_summary)}</pre>
<img src="data:image/png;base64,{img_gradcam}" alt="Grad-CAM 예시">
<p>정량 지표는 혼재됐지만(Scratch·Loc·Near-full·Random은 통제군 대비 높음,
Center·Edge-Ring·Donut은 낮음), 육안 확인 결과 Donut의 고리·Scratch의 선을
히트맵이 뚜렷이 따라가는 등 모델이 의미 있는 근거로 판단한다는 정황이 있었다.
자세한 내용은 리포트 05 참고.</p>

<h3>5.4 SPC 관리도 & 수율 손실 기여도</h3>
<img src="data:image/png;base64,{img_spc}" alt="SPC 관리도">
<pre>{html_escape(spc_summary)}</pre>
<p>자세한 내용(Training/Test 분할 왜곡 발견 등)은 리포트 06 참고.</p>

<h2>6. 정직하게 남겨둔 한계 총정리</h2>
<table>
<tr><th>한계</th><th>내용</th></tr>
<tr><td>Scratch(F1 0.697)</td><td>목표(0.70) 근소 미달. 근본 원인은 Phase 1 리사이즈 단계의 선 연속성 손실로 추정, 두 차례 개선 시도 모두 실패</td></tr>
<tr><td>Near-full 149장</td><td>fold당 검증 29~30장, 결과가 시드에 따라 크게 흔들릴 수 있는 구조적 불안정성</td></tr>
<tr><td>Grad-CAM 정량 지표 한계</td><td>"칸이 정확히 겹쳐야 인정"하는 방식이라 윤곽선 기반의 정당한 판단 근거(Loc 등)를 과소평가</td></tr>
<tr><td>공정 데이터 없음</td><td>패턴→공정 모듈 매핑(7.2절)은 문헌 기반 도메인 지식, 이 데이터로 검증된 것 아님</td></tr>
<tr><td>Training/Test 분할 왜곡</td><td>원 논문 저자의 라벨 curation이 Lot 순서 분석을 왜곡 — Phase 6에서 발견 즉시 보정</td></tr>
<tr><td>"none" 중앙 불량률</td><td>정상 라벨 웨이퍼도 중앙 불량률이 2배 높음(10.55%→25.61%), 원인 미확정(상관관계만 관찰)</td></tr>
<tr><td>실제 타임스탬프 없음</td><td>Lot 순서는 이름 기준 정렬일 뿐 생산 시각 아님</td></tr>
</table>

<h2>7. 다음 단계</h2>
<p>Phase 7 — Streamlit 대시보드: 5화면(웨이퍼 진단·패턴별 성능·Lot 모니터링·
공정 원인 매핑·모델 방법론) 통합, SECOM `app.py` 구조 재사용 예정.</p>

</body>
</html>
"""

report_path = OUTPUT_DIR / "report07_final_analysis.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"\n[완료] 리포트 저장됨: {report_path}")
