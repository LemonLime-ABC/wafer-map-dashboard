# %% [markdown]
# # 03_visualization — Phase 2 시각화
#
# 이 스크립트가 만드는 것 6가지 (채팅에서 미리 정리한 계획 그대로):
# A. 패턴 그리드 (9클래스 x 6장, 범주형 히트맵)
# B. 웨이퍼 맵 크기 분포 (막대 + 산점도)
# C. 클래스 불균형 (로그 스케일 막대)
# D. 클래스별 평균 "불량률" 맵 (다이 있는 칸만 분모로 삼음 — 핵심 함정 회피)
# E. 웨이퍼별 불량률 vs dieSize 산점도 (D의 교훈을 직접 증명하는 그림)
# F. Lot 순서 구간별 불량 패턴 발생 비율 (실제 시간축 아님을 명시)
#
# 결과는 개별 PNG(outputs/)로도 저장되고, 전부 합쳐진
# outputs/02_visualization_report.html 하나로도 만들어진다.

# %%
import sys
import base64
import re
import ast
from pathlib import Path

# Windows 콘솔 코드페이지(cp949)가 표현 못 하는 유니코드 문자(—) 때문에
# print()가 죽지 않도록 출력 인코딩을 UTF-8로 강제한다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# 한글 폰트 (Phase 1에서 확인한 방법 재사용)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_wm811k  # noqa: E402
from resize_utils import resize_nearest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SIZE = 64
RNG_SEED = 7
N_SAMPLES = 6

# Edge-Ring/Edge-Loc/Scratch를 나란히 배치해 비교가 쉽게
CLASS_ORDER = ["none", "Center", "Donut", "Edge-Ring", "Edge-Loc",
               "Scratch", "Loc", "Random", "Near-full"]

rng = np.random.default_rng(RNG_SEED)

# 범주형 컬러맵: 0=배경, 1=정상, 2=불량 세 값을 항상 같은 색에 고정한다.
# (viridis 같은 연속형 컬러맵을 쓰면 1과 2가 비슷한 색으로 보여 오해를 준다)
DIE_CMAP = ListedColormap(["#f0f0f0", "#4c72b0", "#c44e52"])
DIE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], DIE_CMAP.N)


def save_and_encode(fig, name: str) -> str:
    """
    그림을 outputs/name.png로 저장하고, 동시에 base64 문자열로도 인코딩해 반환한다.

    문법 설명 - base64: 이미지는 원래 텍스트가 아니라 이진 데이터라서 HTML
    파일 안에 직접 넣을 수 없다. base64는 이진 데이터를 A-Z/a-z/0-9 같은
    안전한 텍스트 문자로 바꿔주는 인코딩이다. 이렇게 바꾼 문자열을
    "data:image/png;base64,문자열" 형태로 <img> 태그에 넣으면, 별도
    이미지 파일 없이도 HTML 파일 하나에 이미지가 통째로 들어간다.
    """
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    plt.close(fig)
    print(f"    저장됨: {path.name}")
    return b64


html_sections = []  # (제목, 설명 HTML, base64) 튜플을 순서대로 쌓는다

# %% [markdown]
# ## 0. 데이터 로드

# %%
df = load_wm811k()
labeled = df[df["failureType_clean"].notna()].reset_index(drop=True)
print(f"[0] 라벨된 데이터: {len(labeled):,}장, "
      f"클래스: {sorted(labeled['failureType_clean'].unique())}")
assert set(CLASS_ORDER) == set(labeled["failureType_clean"].unique()), \
    "CLASS_ORDER가 실제 데이터의 클래스 목록과 다릅니다"

# %% [markdown]
# ## A. 패턴 그리드

# %%
print("[A] 패턴 그리드 생성 중...")
fig_a, axes_a = plt.subplots(len(CLASS_ORDER), N_SAMPLES,
                              figsize=(N_SAMPLES * 1.5, len(CLASS_ORDER) * 1.5))

for row_i, cls in enumerate(CLASS_ORDER):
    subset = labeled[labeled["failureType_clean"] == cls]
    # replace=False: 같은 웨이퍼가 중복으로 뽑히지 않게 함
    sample_idx = rng.choice(len(subset), size=N_SAMPLES, replace=False)
    for col_i, idx in enumerate(sample_idx):
        ax = axes_a[row_i, col_i]
        wmap = subset.iloc[idx]["waferMap"]
        resized = resize_nearest(wmap, TARGET_SIZE)
        ax.imshow(resized, cmap=DIE_CMAP, norm=DIE_NORM)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col_i == 0:
            ax.set_ylabel(cls, rotation=0, labelpad=45, fontsize=11, ha="right", va="center")

fig_a.suptitle("클래스별 패턴 그리드 (64x64 리사이즈, 클래스당 6장 무작위 샘플)", y=1.0)
plt.tight_layout()
b64_a = save_and_encode(fig_a, "viz_A_pattern_grid")

html_sections.append((
    "A. 패턴 그리드",
    """
    <p>9개 클래스에서 각 6장씩 무작위로 뽑아 64x64로 통일해 나열했다.
    회색=다이 없음(배경), 파랑=정상 다이, 빨강=불량 다이로 <b>항상 같은 색</b>을
    쓰는 범주형 컬러맵을 사용했다 — 연속형 컬러맵(viridis 등)을 쓰면 정상과
    불량이 비슷한 색으로 보여 눈으로 비교할 때 오해를 준다.</p>
    <p><b>확인할 것</b>: Edge-Ring/Edge-Loc/Scratch를 나란히 배치했다.
    Edge-Loc과 Scratch가 실제로 얼마나 비슷해 보이는지 눈으로 비교해보자 —
    이게 이 프로젝트의 핵심 논점(CLAUDE.md 7.1절)이다.</p>
    """,
    b64_a,
))

# %% [markdown]
# ## B. 웨이퍼 맵 크기 분포 (막대 + 산점도)

# %%
print("[B] 크기 분포 그림 생성 중...")
shape_df = pd.read_csv(OUTPUT_DIR / "shape_distribution.csv")
# 문법 설명 - ast.literal_eval: CSV에는 "(32, 29)"처럼 튜플이 문자열로
# 저장돼 있다. literal_eval은 그 문자열을 실제 파이썬 튜플 (32, 29)로
# "안전하게" 복원한다 (eval()과 달리 숫자/문자열/튜플 등 순수 데이터만
# 허용해서, 악성 코드가 문자열에 섞여 있어도 실행되지 않는다).
shape_df["shape_tuple"] = shape_df["shape(H,W)"].apply(ast.literal_eval)
shape_df["H"] = shape_df["shape_tuple"].apply(lambda t: t[0])
shape_df["W"] = shape_df["shape_tuple"].apply(lambda t: t[1])

fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(13, 5))

top20 = shape_df.head(20)
ax_b1.bar(range(len(top20)), top20["count"])
ax_b1.set_xticks(range(len(top20)))
ax_b1.set_xticklabels([str(t) for t in top20["shape_tuple"]], rotation=60, ha="right")
ax_b1.set_ylabel("장수")
ax_b1.set_title("상위 20개 크기 (전체의 57.5%)")

sc = ax_b2.scatter(shape_df["W"], shape_df["H"],
                    s=np.clip(shape_df["count"] / 40, 4, 300),
                    c=np.log10(shape_df["count"] + 1), cmap="viridis", alpha=0.6)
ax_b2.set_xlabel("가로 (W)")
ax_b2.set_ylabel("세로 (H)")
ax_b2.set_title("전체 632가지 크기 분포 (점 크기/색 = 등장 빈도)")
fig_b.colorbar(sc, ax=ax_b2, label="log10(장수)")

plt.tight_layout()
b64_b = save_and_encode(fig_b, "viz_B_size_distribution")

html_sections.append((
    "B. 웨이퍼 맵 크기 분포",
    """
    <p>왼쪽 막대는 Phase 1에서 이미 확인한 상위 20개 크기(전체의 57.5%)다.
    오른쪽 산점도는 나머지 612가지를 포함한 <b>전체 632가지</b> 크기를
    가로x세로 평면에 점으로 찍은 것이다 — 막대그래프만으로는 안 보이던
    "정사각형에 가까운 크기가 많은지, 특정 종횡비로 몰려 있는지" 같은
    구조를 확인할 수 있다.</p>
    """,
    b64_b,
))

# %% [markdown]
# ## C. 클래스 불균형 (로그 스케일 막대)

# %%
print("[C] 클래스 불균형 그림 생성 중...")
class_df = pd.read_csv(OUTPUT_DIR / "class_distribution.csv")
class_df = class_df.sort_values("count", ascending=False)

fig_c, ax_c = plt.subplots(figsize=(8, 5))
ax_c.bar(class_df["class"], class_df["count"], color="#4c72b0")
ax_c.set_yscale("log")
ax_c.set_ylabel("장수 (로그 스케일)")
ax_c.set_title("클래스별 장수 — 라벨된 172,950장 기준")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
b64_c = save_and_encode(fig_c, "viz_C_class_imbalance")

html_sections.append((
    "C. 클래스 불균형",
    """
    <p>none(85.2%)부터 Near-full(0.09%)까지 <b>3자리수 차이</b>가 난다.
    이래서 정확도가 아니라 macro-F1을 주 지표로 쓴다(CLAUDE.md 2.2절 원칙 ①).
    선형 스케일로 그리면 none 외에는 막대가 안 보여서, 로그 스케일을 썼다.</p>
    """,
    b64_c,
))

# %% [markdown]
# ## D. 클래스별 평균 "불량률" 맵 (핵심 함정 회피)

# %%
print("[D] 전체 라벨 데이터 리사이즈 중... (약 2초)")
resized_all = np.stack([resize_nearest(m, TARGET_SIZE) for m in labeled["waferMap"]])
labels_all = labeled["failureType_clean"].to_numpy()
print(f"    완료: {resized_all.shape}")

fig_d, axes_d = plt.subplots(3, 3, figsize=(11, 11))
axes_d_flat = axes_d.flatten()

# 문법 설명 - mpl.colormaps[...]와 set_bad(): mpl.colormaps["YlOrRd"]는
# 컬러맵 객체를 하나 가져온다(가져올 때마다 독립된 사본이라 안전하게
# 수정 가능). set_bad("white")는 "값이 NaN인 칸은 흰색으로 칠해라"는
# 뜻이다 — 아래에서 "그 칸에 다이가 있던 샘플이 하나도 없는 경우"를
# 일부러 NaN으로 만들어 두는데, 그런 칸은 "불량률 0%"가 아니라
# "애초에 데이터가 없다"는 뜻이므로 색으로 구분해줘야 한다.
cmap_rate = mpl.colormaps["YlOrRd"]
cmap_rate.set_bad("white")

im = None
for i, cls in enumerate(CLASS_ORDER):
    mask = labels_all == cls
    stack = resized_all[mask]

    # ---- 핵심: 0(다이없음)은 분모에서 제외하고 계산 ----
    die_present = (stack == 1) | (stack == 2)   # 다이가 있는 칸
    defect = (stack == 2)                        # 그중 불량인 칸
    denom = die_present.sum(axis=0)               # 칸별 "다이 있던 샘플 수"
    numer = defect.sum(axis=0)                    # 칸별 "그중 불량이었던 수"

    rate = np.full(denom.shape, np.nan)
    valid = denom > 0
    rate[valid] = numer[valid] / denom[valid]

    ax = axes_d_flat[i]
    im = ax.imshow(rate, cmap=cmap_rate, vmin=0, vmax=1)
    ax.set_title(f"{cls} (n={mask.sum():,})", fontsize=10)
    ax.axis("off")

fig_d.colorbar(im, ax=axes_d, shrink=0.6, label="칸별 불량률 (다이 있던 샘플 중)")
fig_d.suptitle("클래스별 평균 불량률 맵 (흰색 = 그 위치에 다이가 있던 샘플이 없음)", y=0.93)
b64_d = save_and_encode(fig_d, "viz_D_class_average_rate_map")

html_sections.append((
    "D. 클래스별 평균 불량률 맵 — 정규화가 필요한 지점",
    """
    <p><b>이 데이터셋엔 CD(임계치수) 같은 연속 계측값 컬럼이 없다</b>
    (실제 컬럼은 waferMap/dieSize/lotName/waferIndex/라벨뿐, Phase 1에서 확인).
    하지만 구조적으로 똑같은 함정이 여기 있다: 0/1/2를 그냥 숫자로 보고
    산술평균을 내면, "불량 농도"가 아니라 "웨이퍼 원형 실루엣의 흐림"이
    나온다 — 웨이퍼 가장자리는 리사이즈 후에도 어떤 샘플은 다이가 있고
    어떤 샘플은 없는 칸이 섞이기 때문이다.</p>
    <p>그래서 각 칸마다 <code>불량률 = count(값==2) / count(값 in {1,2})</code>로
    계산했다 — <b>다이가 없는(0) 샘플은 그 칸의 분모에서 아예 제외</b>한다.
    흰색 칸은 "불량률 0%"가 아니라 "그 위치에 다이가 있던 샘플 자체가 없음"을
    뜻한다(원래부터 웨이퍼 원 바깥이었던 자리).</p>
    <p><b>추가로 발견한 것</b>: "none"(패턴 없음으로 라벨된 정상 웨이퍼,
    147,431장) 클래스의 전체 평균 불량률은 10.55%인데, 정중앙 부근만 보면
    25.61%로 두 배 넘게 높다. Loc 클래스도 전체 14.96% vs 중앙 28.18%로
    비슷하게 높다 — 반면 Edge-Ring은 전체 15.19% vs 중앙 15.22%로 차이가
    거의 없다. 즉 "Center 패턴"으로 분류되지 않은 웨이퍼들에서도 정중앙
    부근 다이가 유독 불량으로 잡히는 경향이 있다. 확정할 수는 없지만,
    웨이퍼 중앙에 정렬/모니터링용 기준 다이를 두고 공정 데이터 취합에서
    제외하거나 의도적으로 불량 처리하는 관행이 반도체 공정에 흔히
    있다는 점을 고려하면 그 흔적일 가능성이 있다. 이건 상관관계
    관찰이지 확정된 인과관계가 아니므로, 그렇게만 적어둔다
    (CLAUDE.md 8.4절 원칙).</p>
    """,
    b64_d,
))

# %% [markdown]
# ## E. 웨이퍼별 불량률 vs dieSize 산점도 — D의 교훈을 직접 증명

# %%
print("[E] 불량률 계산 중...")
defect_counts = np.array([int(np.sum(m == 2)) for m in labeled["waferMap"]])
labeled["defect_rate"] = defect_counts / labeled["dieSize"].to_numpy()

# 시각화용으로만 클래스당 최대 3,000장을 무작위 샘플링한다.
# (172,950개 점을 다 찍으면 렌더링이 느려지고 점이 겹쳐 오히려 안 보인다 —
#  통계 계산이 아니라 "눈으로 구조를 보는" 목적이므로 표본으로 충분하다)
CAP = 3000
fig_e, ax_e = plt.subplots(figsize=(9, 6))
for cls in CLASS_ORDER:
    subset = labeled[labeled["failureType_clean"] == cls]
    if len(subset) > CAP:
        subset = subset.sample(CAP, random_state=RNG_SEED)
    ax_e.scatter(subset["dieSize"], subset["defect_rate"], s=6, alpha=0.35, label=cls)

ax_e.set_xscale("log")
ax_e.set_xlabel("dieSize (웨이퍼당 전체 다이 수, 로그 스케일)")
ax_e.set_ylabel("웨이퍼별 불량률 (불량 다이 수 / 전체 다이 수)")
ax_e.set_title("불량률 vs 웨이퍼 크기(dieSize) — 클래스별")
ax_e.legend(fontsize=8, markerscale=3, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
b64_e = save_and_encode(fig_e, "viz_E_defect_rate_vs_diesize")

html_sections.append((
    "E. 불량률 vs dieSize 산점도",
    """
    <p>D에서 설명한 "분모를 어떻게 잡느냐가 결과를 바꾼다"는 교훈을
    직접 보여주는 그림이다. x축은 dieSize(웨이퍼 하나에 들어있는 전체
    다이 수, 로그 스케일), y축은 웨이퍼별 불량률(=불량 다이 수 / 전체
    다이 수)이다.</p>
    <p>만약 여기서 <b>불량 다이 개수(비율이 아닌 절대 개수)</b>를 그대로
    비교했다면, dieSize가 큰 웨이퍼가 단지 다이가 많다는 이유만으로
    "더 심각한 불량"처럼 보였을 것이다. 비율로 정규화해야 클래스 간
    비교가 공정해진다 — Phase 6 수율 기여도 계산에도 이 정의를 그대로 쓴다.</p>
    """,
    b64_e,
))

# %% [markdown]
# ## F. Lot 순서 구간별 불량 패턴 발생 비율

# %%
print("[F] Lot 순서 집계 중...")
# 문법 설명 - str.extract(정규식): lotName은 전부 "lot<숫자>" 형태임을
# 미리 확인했다(46,293개 전부 일치). 정규식 r"lot(\d+)"의 괄호 안이
# "숫자 부분만 뽑아라"는 뜻이고, extract는 그걸 새 컬럼으로 만들어준다.
labeled["lot_num"] = labeled["lotName"].str.extract(r"lot(\d+)").astype(int)

N_BINS = 50
bins = pd.cut(labeled["lot_num"], bins=N_BINS)

# 문법 설명 - groupby + unstack: (구간, 클래스)별 개수를 센 다음,
# 각 구간 안에서 비율로 바꾸고, unstack으로 "구간=행, 클래스=열"인
# 표 형태로 펼친다. 원시 개수 대신 비율을 쓰는 이유: Lot마다 웨이퍼
# 수가 달라서, 원시 개수로 그리면 큰 Lot이 많이 섞인 구간이 실제보다
# 부풀려 보인다.
counts = labeled.groupby([bins, "failureType_clean"], observed=True).size()
rate_table = counts.groupby(level=0, observed=True).apply(lambda s: s / s.sum()).unstack(fill_value=0)
# 구간별 "전체 라벨된 웨이퍼 수" — 이 숫자가 작을수록 비율이 통계적으로
# 불안정하다(적은 표본에서는 우연히 한 클래스가 몰려 비율이 크게 흔들릴 수 있음).
total_per_bin = counts.groupby(level=0, observed=True).sum()

# 위: 클래스별 발생 비율 / 아래: 구간별 표본 크기 — 두 패널을 x축 공유해서
# "표본이 적은 구간에서 비율이 요동친다"는 걸 한눈에 대조할 수 있게 한다.
fig_f, (ax_f1, ax_f2) = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True, height_ratios=[3, 1]
)

for cls in CLASS_ORDER:
    if cls == "none":
        continue  # none은 85%를 차지해 스케일을 압도하므로 제외 (나머지 8개가 관심 대상)
    if cls in rate_table.columns:
        ax_f1.plot(range(len(rate_table)), rate_table[cls], marker="o", markersize=3, label=cls)

ax_f1.set_ylabel("구간 내 발생 비율 (해당 구간 라벨된 웨이퍼 중, none 제외)")
ax_f1.set_title("Lot 순서에 따른 불량 패턴 발생 비율")
ax_f1.legend(fontsize=8, ncol=2)

ax_f2.bar(range(len(total_per_bin)), total_per_bin.values, color="#888888")
ax_f2.set_yscale("log")
ax_f2.set_ylabel("구간 내\n라벨된 웨이퍼 수\n(로그)", fontsize=8)
ax_f2.set_xlabel(f"Lot 순서 구간 (lot 이름의 숫자 기준 {N_BINS}구간 분할 — 주의: 실제 생산 시각 아님)")

plt.tight_layout()
b64_f = save_and_encode(fig_f, "viz_F_lot_order_trend")

html_sections.append((
    "F. Lot 순서별 불량 패턴 발생 비율",
    """
    <p><b>중요한 한계 1</b>: 이 데이터셋엔 실제 타임스탬프가 없다.
    lotName("lot1"~"lot46293")과 waferIndex뿐이다. lot 번호 순서가
    실제 생산 시간 순서와 같다는 보장이 없으므로, 이 그래프를 "시간에
    따른 추이"라고 부르면 안 된다 — "lot 이름을 숫자 순으로 정렬한
    구간"일 뿐이다. 상관관계를 인과관계처럼 서술하지 않는다는
    CLAUDE.md 8.4절 원칙과 같은 이유로, 제목에 이 한계를 그대로 남겨뒀다.</p>
    <p><b>중요한 한계 2 (실제로 발견됨)</b>: 아래 패널(구간별 라벨된 웨이퍼 수,
    로그 스케일)을 보면 초반 구간은 55~500장 수준인데 후반 구간은
    1만~2만 장 이상으로 <b>표본 크기가 40배 넘게 차이 난다</b>. 표본이
    적은 초반 구간의 급격한 스파이크(위 패널에서 특정 클래스가 80~90%까지
    치솟는 구간들)는 실제 공정 추세라기보다 <b>적은 표본에서 우연히 한
    클래스가 몰린 통계적 잡음일 가능성이 크다</b>. 반대로 표본이 훨씬 큰
    후반 구간은 비율이 작고 평평한데, 이건 "불량이 없다"는 뜻이 아니라
    "많은 표본에 걸쳐 8개 클래스로 고르게 나뉘어 개별 비중이 작아졌다"는
    뜻이다. 두 패널을 같이 보지 않고 위 패널만 봤다면 초반 스파이크를
    실제 추세로 오독했을 것이다 — Phase 6 SPC 관리도를 만들 때도 구간별
    표본 크기를 반드시 같이 표시해야 한다.</p>
    <p>none(85%)은 스케일을 압도해서 나머지 8개 패턴의 움직임을 가리므로
    제외했다. 각 구간의 분모(전체 라벨된 웨이퍼 수)에는 none이 포함된
    채로 비율을 계산했다 — 그래야 "그 구간에서 각 불량이 실제로 차지한
    비중"이 맞게 나온다.</p>
    """,
    b64_f,
))

# %% [markdown]
# ## 리포트 조립 — HTML 하나로 통합

# %%
print("[리포트] HTML 조립 중...")

sections_html = ""
for title, desc, b64img in html_sections:
    sections_html += f"""
    <section>
      <h2>{title}</h2>
      {desc}
      <img src="data:image/png;base64,{b64img}" alt="{title}">
    </section>
    """

html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>[리포트 01] Phase 2 시각화 리포트 — WM-811K</title>
<style>
  body {{ font-family: "Malgun Gothic", sans-serif; max-width: 1000px; margin: 40px auto;
         padding: 0 20px; line-height: 1.6; color: #222; }}
  h1 {{ border-bottom: 3px solid #4c72b0; padding-bottom: 10px; }}
  h2 {{ color: #2b3a55; margin-top: 50px; border-left: 5px solid #4c72b0; padding-left: 10px; }}
  section {{ margin-bottom: 40px; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }}
  .meta {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>[리포트 01] WM-811K 웨이퍼 맵 — Phase 2 시각화 리포트</h1>
<p class="meta">라벨된 웨이퍼 {len(labeled):,}장 기준 (전체 811,457장 중).
데이터 출처: Kaggle qingyi/wm811k-wafer-map (LSWMD.pkl)</p>
{sections_html}
</body>
</html>
"""

report_path = OUTPUT_DIR / "report01_visualization.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"\n[완료] 리포트 저장됨: {report_path}")
print(f"       더블클릭하면 브라우저에서 바로 열립니다 (인터넷 연결 불필요)")
