# %% [markdown]
# # 07_spc_control_chart — Phase 6 Lot 단위 SPC 관리도
#
# SECOM 프로젝트(`1. 반도체_수율_프로젝트/main2.py`)의 주간 p-관리도
# 로직을 그대로 가져와 집계 단위만 "주(week)"에서 "Lot 묶음"으로 바꾼다.
#
# ```
# SECOM:  p_bar = 전체 평균 불량률
#         ucl = p_bar + 3*sqrt(p_bar*(1-p_bar)/n)
#         binomtest(불량수, n, p_bar) 로 p값 산출
# WM    : p_bar = 패턴별 전체 발생률 (위와 동일한 식 그대로 사용)
# ```
#
# **집계 단위 결정 — 실측 기반**: Lot 하나당 라벨된 웨이퍼 수를 세어보니
# 전체 46,293개 Lot 중 라벨이 1장이라도 있는 Lot은 10,762개(23%)뿐이었다.
# Lot 단위 그대로 관리도를 그리면 대부분 표본이 너무 적어 판단이 안 된다.
# 그래서 Phase 2에서 썼던 "Lot 번호 범위로 균등 분할"(구간마다 표본 크기가
# 55~22,700장으로 들쭉날쭉했던 방식) 대신, **"라벨된 웨이퍼 수가 목표치에
# 도달할 때까지 연속된 Lot을 묶는" 방식**을 쓴다 — 이러면 매 구간의 표본
# 크기가 훨씬 고르게 맞춰져 관리한계선(UCL) 폭을 공정하게 비교할 수 있다.

# %%
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as sstats

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_wm811k  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_GROUP_SIZE = 1000  # 그룹당 목표 라벨 웨이퍼 수
PATTERNS = ["Center", "Donut", "Edge-Ring", "Edge-Loc", "Loc", "Random", "Scratch", "Near-full"]

# %% [markdown]
# ## 1. 데이터 로드 및 Lot 묶음 구성

# %%
df = load_wm811k()
labeled_all = df[df["failureType_clean"].notna()].reset_index(drop=True)
n_total_lots = df["lotName"].nunique()
n_labeled_lots = labeled_all["lotName"].nunique()

print(f"[1] 라벨된 웨이퍼(전체): {len(labeled_all):,}장")
print(f"    전체 Lot {n_total_lots:,}개 중 라벨 있는 Lot {n_labeled_lots:,}개 "
      f"({n_labeled_lots / n_total_lots * 100:.1f}%)")

# ------------------------------------------------------------------
# 중요: 원 연구자의 "Training" 라벨을 제외하고 "Test" 라벨만 쓴다.
# ------------------------------------------------------------------
# 실측 결과, Lot 순서(그룹 5~25)에 몰려 있던 극단적인 발생률 스파이크가
# 그 구간 웨이퍼의 100%가 원 연구자의 "Training" 라벨이라는 사실과
# 정확히 일치했다 — 즉 실제 생산 이상이 아니라, 원 연구자가 자기 모델
# 학습용으로 다양한 불량 유형을 일부러 골고루 뽑아 만든 "Training" 세트가
# Lot 순서상 우연히 뭉쳐 있어 생긴 착시였다. CLAUDE.md에 "Training/Test
# 태그는 재사용하지 않는다"고 이미 적어뒀지만, Lot 순서 분석에서 이렇게
# 극적으로 드러날 줄은 몰랐다 — 발견 즉시 반영한다.
# 자연스러운 생산 비율에 더 가까운 "Test" 라벨(118,595장)만 남긴다.
labeled = labeled_all[labeled_all["trainTestLabel_clean"] == "Test"].reset_index(drop=True)
print(f"[1] 'Training' 라벨 제외 -> 관리도에 쓸 웨이퍼: {len(labeled):,}장 "
      f"(제외된 Training: {len(labeled_all) - len(labeled):,}장)")

labeled["lot_num"] = labeled["lotName"].str.extract(r"lot(\d+)").astype(int)
labeled = labeled.sort_values("lot_num").reset_index(drop=True)

# 문법 설명: (누적 인덱스 // 목표 크기)로 그룹 번호 매기기
# 0,1,2,...,999번째 행은 그룹 0, 1000~1999번째는 그룹 1, ... 식으로
# Lot 번호 순서를 유지한 채 "라벨 웨이퍼 1,000장씩" 묶는다.
labeled["group_id"] = np.arange(len(labeled)) // TARGET_GROUP_SIZE

group_sizes = labeled.groupby("group_id", observed=True).size()
print(f"    그룹 {len(group_sizes)}개 생성, 그룹당 표본 크기: "
      f"중앙값 {group_sizes.median():.0f} (마지막 그룹만 더 작을 수 있음: {group_sizes.iloc[-1]})")

# %% [markdown]
# ## 2. 패턴별 p-관리도 계산 (SECOM 로직 재사용)

# %%
print("\n[2] 패턴별 p-관리도 계산 중...")

chart_data = {}  # pattern -> DataFrame(group_id, n, count, rate, ucl, z, pval, over_ucl)
alerts = []

for pattern in PATTERNS:
    is_pattern = (labeled["failureType_clean"] == pattern)
    p_bar = is_pattern.mean()  # 전체 기간 평균 발생률 = 중심선

    g = labeled.groupby("group_id", observed=True).agg(
        n=("failureType_clean", "size"),
        count=("failureType_clean", lambda s: (s == pattern).sum()),
    )
    g["rate"] = g["count"] / g["n"]
    # SECOM과 동일 공식: p_bar + 3*sqrt(p_bar*(1-p_bar)/n)
    g["ucl"] = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / g["n"])
    g["z"] = (g["rate"] - p_bar) / np.sqrt(p_bar * (1 - p_bar) / g["n"])
    g["pval"] = [
        sstats.binomtest(int(c), int(n), p_bar, alternative="greater").pvalue
        for c, n in zip(g["count"], g["n"])
    ]
    g["over_ucl"] = g["rate"] > g["ucl"]
    g["p_bar"] = p_bar

    chart_data[pattern] = g

    n_alerts = int(g["over_ucl"].sum())
    print(f"    {pattern:10s}: 전체 발생률(중심선) {p_bar*100:.3f}%, "
          f"관리한계 초과 구간 {n_alerts}/{len(g)}개")
    if n_alerts > 0:
        worst = g[g["over_ucl"]].sort_values("z", ascending=False).iloc[0]
        alerts.append((pattern, int(worst.name), worst["rate"], worst["ucl"], worst["z"], worst["pval"]))

# %% [markdown]
# ## 3. 관리도 시각화 (8개 패턴 + 구간별 표본 크기)

# %%
print("\n[3] 관리도 그림 생성 중...")

fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
axes_flat = axes.flatten()

for i, pattern in enumerate(PATTERNS):
    ax = axes_flat[i]
    g = chart_data[pattern]
    x = g.index.values

    ax.plot(x, g["rate"] * 100, marker="o", markersize=3, linewidth=1, color="#4c72b0", label="발생률")
    ax.plot(x, g["ucl"] * 100, linestyle="--", linewidth=1, color="#cc3333", label="UCL")
    ax.axhline(g["p_bar"].iloc[0] * 100, linestyle=":", linewidth=1, color="#888888", label="중심선(p_bar)")

    over = g[g["over_ucl"]]
    if len(over) > 0:
        ax.scatter(over.index, over["rate"] * 100, color="red", s=25, zorder=5, label="관리한계 초과")

    ax.set_title(f"{pattern} (중심선 {g['p_bar'].iloc[0]*100:.2f}%, 초과 {len(over)}구간)", fontsize=10)
    ax.set_ylabel("발생률(%)")
    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

for ax in axes_flat[6:]:
    ax.set_xlabel(f"Lot 묶음 순서 (그룹당 라벨 약 {TARGET_GROUP_SIZE}장 — 실제 생산 시각 아님)")

fig.suptitle("패턴별 Lot 묶음 단위 p-관리도", y=1.0)
plt.tight_layout()
chart_path = OUTPUT_DIR / "phase6_spc_control_chart.png"
fig.savefig(chart_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"    저장됨: {chart_path}")

# 구간별 표본 크기 (모든 패턴에 공통)
fig2, ax2 = plt.subplots(figsize=(12, 3.5))
ax2.bar(group_sizes.index, group_sizes.values, color="#888888")
ax2.set_xlabel(f"Lot 묶음 순서 (그룹당 목표 라벨 {TARGET_GROUP_SIZE}장)")
ax2.set_ylabel("그룹 내\n실제 라벨 수")
ax2.set_title("구간별 표본 크기 (설계상 대체로 균등 — 마지막 구간만 작음)")
plt.tight_layout()
size_path = OUTPUT_DIR / "phase6_group_sizes.png"
fig2.savefig(size_path, dpi=120, bbox_inches="tight")
plt.close(fig2)
print(f"    저장됨: {size_path}")

# %% [markdown]
# ## 4. 수율 손실 기여도 (PLAN.md 6.2절)
#
# 수율 손실 기여도 = 패턴 발생률 × 그 패턴의 평균 불량 다이 비율
# (dieSize는 검증된 값이므로 그대로 사용 — CLAUDE.md 4.4절)

# %%
print("\n[4] 수율 손실 기여도 계산 중...")

defect_counts = np.array([int(np.sum(m == 2)) for m in labeled["waferMap"]])
labeled["defect_rate"] = defect_counts / labeled["dieSize"].to_numpy()

n_labeled = len(labeled)
contribution_rows = []
for pattern in PATTERNS:
    subset = labeled[labeled["failureType_clean"] == pattern]
    occurrence_rate = len(subset) / n_labeled
    avg_defect_rate = subset["defect_rate"].mean()
    contribution = occurrence_rate * avg_defect_rate
    contribution_rows.append({
        "패턴": pattern,
        "발생률(%)": occurrence_rate * 100,
        "평균 불량다이비율(%)": avg_defect_rate * 100,
        "수율손실 기여도": contribution,
    })

contribution_df = pd.DataFrame(contribution_rows).sort_values("수율손실 기여도", ascending=False)
contribution_df["기여도 순위"] = range(1, len(contribution_df) + 1)
print(contribution_df.to_string(index=False))

# %% [markdown]
# ## 5. 결과 저장

# %%
report_lines = ["=== Phase 6 SPC 관리도 요약 ===", ""]
report_lines.append(f"전체 Lot {n_total_lots:,}개 중 라벨 있는 Lot {n_labeled_lots:,}개 ({n_labeled_lots/n_total_lots*100:.1f}%)")
report_lines.append(f"Lot 묶음 {len(group_sizes)}개 (그룹당 목표 라벨 {TARGET_GROUP_SIZE}장)")
report_lines.append("")
report_lines.append("--- 패턴별 관리한계 초과 구간 (가장 심한 것) ---")
for pattern, grp, rate, ucl, z, pval in sorted(alerts, key=lambda r: -r[4]):
    report_lines.append(
        f"{pattern:10s} 그룹#{grp}: 발생률 {rate*100:.2f}% (UCL {ucl*100:.2f}%), "
        f"Z={z:+.2f}, p={pval:.4f}"
    )
if not alerts:
    report_lines.append("(관리한계를 초과한 구간 없음)")

report_lines.append("")
report_lines.append("--- 수율 손실 기여도 (발생률 x 평균 불량다이비율) ---")
report_lines.append(contribution_df.to_string(index=False))

summary_path = OUTPUT_DIR / "phase6_spc_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

contribution_df.to_csv(OUTPUT_DIR / "phase6_yield_contribution.csv", index=False, encoding="utf-8-sig")

print(f"\n[완료] 요약 저장: {summary_path}")
print(f"[완료] 기여도 표 저장: {OUTPUT_DIR / 'phase6_yield_contribution.csv'}")

# 대시보드(Phase 7)가 정적 이미지 대신 인터랙티브 그래프를 그릴 수 있도록
# 패턴별 그룹 단위 원본 표(rate/ucl/z/pval)를 joblib으로 따로 저장해둔다.
import joblib  # noqa: E402
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
joblib.dump(
    {"chart_data": chart_data, "group_sizes": group_sizes, "contribution_df": contribution_df,
     "n_total_lots": n_total_lots, "n_labeled_lots": n_labeled_lots,
     "target_group_size": TARGET_GROUP_SIZE},
    ARTIFACTS_DIR / "spc_bundle.joblib",
)
print(f"[완료] 대시보드용 번들 저장: {ARTIFACTS_DIR / 'spc_bundle.joblib'}")
