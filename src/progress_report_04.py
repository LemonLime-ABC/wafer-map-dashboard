# %% [markdown]
# # progress_report_04 — [리포트 04] 진행 상황 스냅샷 (Phase 4 개선 진행 중)
#
# 리포트 03(Phase 1~3 완료 시점) 이후 진행된 내용을 담는다:
# - Phase 4 1차 학습 결과 (GAP 구조) — PLAN.md 목표 미달, 원인 분석
# - 개선 Step 1(Flatten head) 확인 결과 — 채택 확정
# - 개선 Step 2(128x128 해상도) — 진행 상황에 따라 결과 유무가 다름
#
# 이 스크립트는 무거운 연산(2GB pickle 로드, 811K장 리사이즈)을 하지
# 않는다 — 이미 저장된 요약 파일과 npz만 가볍게 읽어서 조립한다.
# (지금 5-Fold/해상도 비교 학습이 배경에서 돌고 있을 수 있어, CPU를
# 많이 쓰는 작업은 피한다)

# %%
import sys
import base64
import io
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def read_text(name: str) -> str | None:
    p = OUTPUT_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else None


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# %% [markdown]
# ## 1. 저장된 요약 파일 로드

# %%
phase4_summary = read_text("phase4_train_evaluate_summary.txt")
input_rep_check = read_text("phase4_input_representation_check.txt")
head_check = read_text("phase4_head_architecture_check.txt")
resolution_check = read_text("phase4_resolution_check.txt")  # 아직 없을 수 있음(진행 중)

with open(OUTPUT_DIR / "phase4_metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

print(f"[1] Phase4 1차 결과: macro-F1={metrics['oof_macro_f1']:.4f}")
print(f"    해상도 비교 결과 파일 존재 여부: {resolution_check is not None}")

# %% [markdown]
# ## 2. 1차 학습 혼동 행렬 히트맵 생성

# %%
cm_data = np.load(OUTPUT_DIR / "phase4_confusion_matrix.npz")
cm = cm_data["cm"]
class_names = cm_data["class_names"]
n_classes = len(class_names)

cm_norm = cm / cm.sum(axis=1, keepdims=True)  # 행(실제 클래스) 기준 비율로 정규화

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("예측 클래스")
ax.set_ylabel("실제 클래스")
ax.set_title("Phase 4 1차(GAP) 혼동 행렬 — 행 기준 비율")

for i in range(n_classes):
    for j in range(n_classes):
        val = cm_norm[i, j]
        if val > 0.02:  # 너무 작은 값은 숫자를 안 찍어 가독성 확보
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

fig.colorbar(im, label="비율 (행 기준)")
plt.tight_layout()

cm_path = OUTPUT_DIR / "phase4_confusion_matrix_heatmap.png"
fig.savefig(cm_path, dpi=120, bbox_inches="tight")
plt.close(fig)
with open(cm_path, "rb") as f:
    img_cm = base64.b64encode(f.read()).decode("ascii")
print(f"[2] 혼동 행렬 히트맵 저장: {cm_path}")

# %% [markdown]
# ## 3. HTML 조립

# %%
print("[3] 리포트 조립 중...")

resolution_section = ""
if resolution_check:
    resolution_section = f"""
    <h3>해상도 비교 결과</h3>
    <pre>{html_escape(resolution_check)}</pre>
    """
else:
    resolution_section = """
    <div class="status-progress">
    아직 실행 중입니다. 이 리포트를 만든 시점엔 결과가 없어 포함하지
    못했다 — 완료되면 리포트 05에서 결과와 함께 보고한다.
    </div>
    """

html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>[리포트 04] 진행 상황 스냅샷 — Phase 4 개선 중</title>
<style>
  body {{ font-family: "Malgun Gothic", sans-serif; max-width: 1000px; margin: 40px auto;
         padding: 0 20px; line-height: 1.7; color: #222; }}
  h1 {{ border-bottom: 3px solid #4c72b0; padding-bottom: 10px; }}
  h2 {{ color: #2b3a55; margin-top: 55px; border-left: 5px solid #4c72b0; padding-left: 10px; }}
  h3 {{ color: #444; margin-top: 25px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.92em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #4c72b0; color: white; }}
  tr:nth-child(even) {{ background: #f7f7f7; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
  pre {{ background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px; padding: 12px 16px;
         overflow-x: auto; font-size: 0.85em; white-space: pre-wrap; }}
  .warn {{ background: #fff3cd; border-left: 5px solid #cc9900; padding: 12px 16px; margin: 15px 0; }}
  .bad {{ background: #f8d7da; border-left: 5px solid #dc3545; padding: 12px 16px; margin: 15px 0; }}
  .good {{ background: #d4edda; border-left: 5px solid #28a745; padding: 12px 16px; margin: 15px 0; }}
  .status-progress {{ background: #d1ecf1; border-left: 5px solid #0c5460; padding: 12px 16px; margin: 15px 0; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  .navlist {{ background: #f0f4f8; padding: 12px 16px; border-radius: 6px; }}
</style>
</head>
<body>

<h1>[리포트 04] 진행 상황 스냅샷 — Phase 4 개선 진행 중</h1>
<p class="meta">이전 리포트: 01(시각화) · 02(데이터 개요) · 03(Phase 1~3 스냅샷)</p>

<div class="navlist">
<b>리포트 목록</b><br>
01 — Phase 2 시각화 리포트<br>
02 — 데이터 개요 리포트<br>
03 — 진행 상황 스냅샷 (Phase 1~3 완료 시점)<br>
04 — 진행 상황 스냅샷 (현재 문서, Phase 4 개선 진행 중)
</div>

<h2>1. Phase 4 1차 학습 결과 — 목표 미달, 정직하게 보고</h2>
<p>PLAN.md 7절 최소 목표("9개 클래스 중 7개 이상 F1≥0.80")를 <b>달성하지 못했다</b>
(실제 4개). 포장하지 않고 그대로 기록한다.</p>
<pre>{html_escape(phase4_summary) if phase4_summary else "요약 파일 없음"}</pre>

<img src="data:image/png;base64,{img_cm}" alt="1차 학습 혼동 행렬 히트맵">

<div class="bad">
<b>예상이 빗나간 지점</b>: 원래 걱정했던 Edge-Loc↔Scratch 혼동(10% 미만)은
오히려 목표를 달성했다(5.0%, 7.5%). 대신 예상 못 했던 <b>"Loc" 클래스가
거의 모든 클래스와 뒤섞이는 허브</b>였다 — Center의 24.5%, Scratch의
20.0%, Donut의 14.8%가 Loc으로 오분류됐다.
</div>

<h2>2. 원인 분석 — GAP이 위치·모양 정보를 지움</h2>
<p>클래스를 "밀도로 구분되는 클래스"(Edge-Ring/Random/Near-full/none, 전부
F1 0.83 이상)와 "위치·모양으로 구분되는 클래스"(Center/Loc/Scratch/Donut,
전부 F1 0.70 이하)로 나누면 정확히 갈린다. 모델 구조(Conv 3단 →
<b>Global Average Pooling</b>)가 8×8까지 남아있던 위치 정보를 마지막에
평균 1개 값으로 뭉개버리는 게 원인으로 추정됐다.</p>

<h2>3. 개선 Step 1 — Flatten head <span class="good" style="display:inline-block;padding:2px 10px;">채택 확정</span></h2>
<p>GAP 대신 8×8 위치 정보를 그대로 펴서(Flatten) Dense층에 넘기는 구조로
빠르게(2-Fold×15epoch) 검증했다.</p>
<pre>{html_escape(head_check) if head_check else "요약 파일 없음"}</pre>
<div class="good">
전체 macro-F1 +0.09, 특히 가설이 지목한 Center(+0.16)·Donut(+0.22)·Loc(+0.18)이
크게 개선됨을 확인했다. <b>Scratch만 거의 변화 없음(+0.002)</b> — 이는
Scratch의 문제가 모델 구조가 아니라 리사이즈 단계(Phase 1에서 이미 발견한
선 연속성 손실)에 있다는 기존 가설과 일치한다.
</div>

<h2>4. 개선 Step 2 — 해상도 128×128 (Scratch 겨냥)</h2>
<p>Flatten head를 고정하고, 64×64와 128×128 해상도만 비교해 Scratch가
실제로 좋아지는지 확인 중이다. 같은 RNG_SEED로 뽑아 두 해상도 모두
정확히 같은 웨이퍼 샘플을 사용한다(해상도 효과만 분리해서 보기 위함).</p>
{resolution_section}

<h2>5. 입력 표현 참고 자료 (raw 채택 근거)</h2>
<pre>{html_escape(input_rep_check) if input_rep_check else "요약 파일 없음"}</pre>

<h2>다음 단계</h2>
<ol>
  <li>Step 2(해상도) 결과 확인 → Scratch 개선 여부에 따라 채택 여부 결정</li>
  <li>필요시 Step 3(앙상블: raw+onehot 모델 결합) 시도</li>
  <li>필요시 Step 4(모델 확장) 시도</li>
  <li>최종 구조 확정 후 5-Fold 본 학습 재실행 → PLAN.md 목표 재확인</li>
  <li>Phase 5(Grad-CAM) 진행</li>
</ol>

</body>
</html>
"""

report_path = OUTPUT_DIR / "report04_progress_snapshot_phase4.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"\n[완료] 리포트 저장됨: {report_path}")
