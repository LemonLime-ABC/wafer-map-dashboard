# %% [markdown]
# # 13_unlabeled_samples — "즉석 판정" 화면용 미라벨 웨이퍼와 기준값 준비
#
# 새 화면(화면 6)은 사용자가 웨이퍼 맵 파일을 끌어다 넣으면 그 자리에서
# 모델을 돌려 판정한다. 그러려면 두 가지가 필요하다.
#
# 1. **시연용 입력** — 원본 81만 장 중 라벨이 없는 638,507장은 원 연구자도,
#    우리 모델도 답을 모르는 웨이퍼다. "학습시켜놓고 결과만 보여준 것"이 아니라
#    진짜 처음 보는 데이터를 넣는다는 걸 보여주려면 여기서 뽑아야 한다.
#    배포된 앱에는 2GB 원본이 없으므로, 일부를 뽑아 작은 파일로 들고 간다.
#
# 2. **입력 검사 기준값** — softmax 분류기는 무엇을 넣든 9개 중 하나를 반드시
#    골라낸다. 웨이퍼 맵이 아닌 이상한 격자를 넣어도 "Loc 83%" 같은 답이 나온다.
#    그래서 "학습 데이터와 얼마나 다른 입력인가"를 경고할 기준을
#    라벨된 172,950장에서 실측해 둔다.

# %%
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_wm811k   # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
SAMPLES = PROJECT_ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)

RNG_SEED = 42
N_POOL = 1000          # 앱에 들고 갈 미라벨 웨이퍼 수 (값이 0/1/2라 압축하면 수백 KB)
COMPRESS = ("zlib", 9)

# %%
df = load_wm811k()
labeled = df[df["failureType_clean"].notna()]
unlabeled = df[df["failureType_clean"].isna()]
print(f"[1] 라벨 {len(labeled):,}장 / 미라벨 {len(unlabeled):,}장")


def map_stats(m: np.ndarray):
    """웨이퍼 맵 한 장의 요약값: 배경(0) 비율, 불량 다이 비율(불량/전체 다이)."""
    n0 = (m == 0).sum()
    n1 = (m == 1).sum()
    n2 = (m == 2).sum()
    dies = n1 + n2
    return n0 / m.size, (n2 / dies if dies else np.nan), int(dies)


# %% [markdown]
# ## 기준값 — 라벨된 172,950장에서 실측
#
# 분위수(0.5%~99.5%)로 잡는 이유: 최솟값·최댓값은 이상치 한 장에 끌려가서
# 기준으로 쓰기 어렵다. 99%가 들어가는 범위 밖이면 "학습 때 거의 못 본 입력"이라고
# 말할 수 있다.

# %%
st_lab = np.array([map_stats(m) for m in labeled["waferMap"]], dtype=float)
shapes = np.array([m.shape for m in labeled["waferMap"]])
ref = {
    "n_ref": int(len(labeled)),
    "zero_frac_q": np.nanpercentile(st_lab[:, 0], [0.5, 99.5]).tolist(),
    "zero_frac_min": float(np.nanmin(st_lab[:, 0])),
    "defect_ratio_q": np.nanpercentile(st_lab[:, 1], [0.5, 99.5]).tolist(),
    "side_min": int(shapes.min()),
    "side_max": int(shapes.max()),
}
print("[2] 입력 검사 기준값(라벨된 전체 기준)")
for k, v in ref.items():
    print(f"    {k}: {v}")

# %% [markdown]
# ## 시연용 미라벨 웨이퍼 1,000장 — 무작위 추출
#
# 불량이 뚜렷한 것만 골라오면 "잘 되는 것만 보여준다"는 반박을 받는다.
# 그래서 앱에 들고 가는 풀(pool)은 **무작위**로 뽑고, 불량 비율로 좁히는 건
# 화면에서 사용자가 직접 하게 한다.

# %%
rng = np.random.default_rng(RNG_SEED)
has_die = unlabeled["waferMap"].apply(lambda m: ((m == 1) | (m == 2)).any())
cand = unlabeled[has_die]
pick = cand.iloc[rng.choice(len(cand), size=N_POOL, replace=False)]

# 미라벨이라는 사실을 한 번 더 확인 — 학습 샘플(12,763장)은 라벨된 데이터에서만
# 뽑았으므로, 여기 들어온 웨이퍼는 어떤 fold 모델도 본 적이 없다.
assert pick["failureType_clean"].isna().all()

pool_stats = np.array([map_stats(m) for m in pick["waferMap"]], dtype=float)
pool = {
    "maps": [m.astype(np.uint8) for m in pick["waferMap"]],   # 원본 크기 그대로
    "lot": pick["lotName"].astype(str).tolist(),
    "wafer_index": pick["waferIndex"].astype(int).tolist(),
    "orig_index": pick.index.astype(int).tolist(),
    "defect_ratio": pool_stats[:, 1].tolist(),
}
print(f"[3] 미라벨 풀 {N_POOL}장 추출 — 불량 다이 비율 중앙값 "
      f"{np.nanmedian(pool_stats[:, 1]):.3f}, 10% 이상 {int((pool_stats[:, 1] >= 0.10).sum())}장")

# %% [markdown]
# ## 끌어다 넣기 시연용 파일 — samples/ 폴더
#
# 1. 미라벨 웨이퍼 한 장씩 CSV (0/1/2 격자) — 풀에서 불량 비율 분위수별로 6장
# 2. Lot 한 통(25장) — 라벨이 하나도 없는 25장짜리 Lot 중 평균 불량 비율이
#    가장 높은 Lot. **시연 효과를 위해 고른 것**이라 대표성은 없다(화면에도 명시).
# 3. Bin 코드 형식 예시 — 실제 테스트 장비는 0/1/2가 아니라 bin 번호로 결과를 낸다
#    (다이 없음=빈칸, Pass=1, Fail=여러 번호). 같은 웨이퍼를 그 형식으로 바꾼 파일.

# %%
def to_csv(m: np.ndarray, path: Path):
    pd.DataFrame(m).to_csv(path, header=False, index=False)


order = np.argsort(pool_stats[:, 1])
singles = []
for q in [0.50, 0.90, 0.97, 0.99, 0.995, 0.999]:
    i = order[int(q * (N_POOL - 1))]
    name = f"unlabeled_{pool['lot'][i]}_w{pool['wafer_index'][i]:02d}.csv"
    to_csv(pool["maps"][i], SAMPLES / name)
    singles.append((name, pool_stats[i, 1]))
print("[4] 단일 웨이퍼 CSV")
for n, r in singles:
    print(f"    {n}  (불량 다이 비율 {r:.3f})")

# Lot 한 통 — 25장이 전부 미라벨인 Lot
# 선정 기준: 25장 **모두** 학습 데이터 범위(배경 비율·불량 다이 비율 99% 구간) 안에 있는
# Lot 중 평균 불량 비율이 가장 높은 것. 처음엔 범위를 안 봐서 평균 불량 비율 50%짜리
# 극단 Lot이 뽑혔고, 25장 중 23장이 '학습 범위 밖' 경고를 받아 시연이 되지 않았다.
# 모델 판정 결과로 고르면 '잘 맞는 것만 골랐다'가 되므로 데이터 특성만으로 고른다.
lot_sizes = unlabeled.groupby("lotName").size()
full_lots = lot_sizes[lot_sizes == 25].index
zlo, zhi = ref["zero_frac_q"]
dlo, dhi = ref["defect_ratio_q"]
lot_df = unlabeled[unlabeled["lotName"].isin(full_lots)].copy()
_st = np.array([map_stats(m) for m in lot_df["waferMap"]], dtype=float)
lot_df["zf"], lot_df["dr"] = _st[:, 0], _st[:, 1]
lot_df["in_range"] = lot_df["zf"].between(zlo, zhi) & (lot_df["dr"] <= dhi)
g = lot_df.groupby("lotName").agg(all_in=("in_range", "all"), dr=("dr", "mean"))
lot_mean = g[g["all_in"]]["dr"]
demo_lot = lot_mean.idxmax()
lot_dir = SAMPLES / demo_lot
lot_dir.mkdir(exist_ok=True)
lot_rows = unlabeled[unlabeled["lotName"] == demo_lot].sort_values("waferIndex")
for _, r in lot_rows.iterrows():
    to_csv(r["waferMap"].astype(np.uint8), lot_dir / f"{demo_lot}_w{int(r['waferIndex']):02d}.csv")
print(f"[5] Lot 한 통: {demo_lot} ({len(lot_rows)}장, 평균 불량 다이 비율 {lot_mean.max():.3f}) -> {lot_dir.name}/")

# Bin 코드 형식 예시 — 불량 다이를 여러 fail bin 번호로 흩어 놓는다
i = order[int(0.99 * (N_POOL - 1))]
m = pool["maps"][i]
bins = np.full(m.shape, "", dtype=object)
bins[m == 1] = "1"
fail_codes = rng.choice(["3", "5", "7", "12"], size=int((m == 2).sum()))
bins[m == 2] = fail_codes
bin_name = f"bincode_example_{pool['lot'][i]}_w{pool['wafer_index'][i]:02d}.csv"
pd.DataFrame(bins).to_csv(SAMPLES / bin_name, header=False, index=False)
print(f"[6] Bin 코드 형식 예시: {bin_name} (다이 없음=빈칸, Pass=1, Fail=3/5/7/12)")

# %%
joblib.dump({"pool": pool, "ref": ref, "demo_lot": demo_lot},
            ARTIFACTS / "unlabeled_samples.joblib", compress=COMPRESS)
out = ARTIFACTS / "unlabeled_samples.joblib"
print(f"\n[완료] {out.name} ({out.stat().st_size / 1e6:.2f} MB)")
