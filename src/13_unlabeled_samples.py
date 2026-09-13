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
# ### 처음 방식과 그 문제 (2026-09-13 교체)
# 처음엔 풀에서 불량 비율 분위수별로 뽑았더니, 불량 비율이 높은 쪽에 **가로·세로 줄무늬**
# 웨이퍼(행·열 단위로 통째로 불량)가 걸렸다. 9개 패턴 어디에도 속하지 않는 모양인데 모델은
# "Random 100%, 5/5 만장일치"로 자신 있게 판정했다 — 닫힌 분류기의 한계를 보여주는 사례지만,
# 시연 파일로는 적절하지 않았다. Lot도 불량이 30~45% 흩뿌려진 잡음 같은 Lot이 뽑혔다.
#
# ### 지금 방식
# 1. **데이터 특성으로 거른다** (모델 판정은 쓰지 않는다 — 판정으로 고르면 '잘 맞는 것만 골랐다'가 된다)
#    - 라벨 없음, 배경 비율·불량 비율이 학습 데이터 99% 범위 안, 한 변 25~80칸
#    - 줄무늬 없음: 다이가 많은 가운데 행/열 중 90% 이상이 불량인 줄이 하나도 없어야 함
#    - 뚜렷한 덩어리: 테두리 한 줄을 뺀 안쪽에서, 가장 큰 불량 덩어리가 20다이 이상이고
#      전체 불량의 40% 이상 (테두리를 빼는 이유 — 테두리만 붉은 웨이퍼가 점수를 독차지했다)
# 2. 걸러진 후보를 **덩어리 위치(중심부/중간/가장자리)와 길쭉함**으로 나눠 그림으로 보고,
#    **사람이 모양이 서로 다른 것을 골랐다.** 그래서 아래는 목록으로 고정한다.
#    이 과정은 scratchpad 탐색 스크립트로 했고, 여기서는 고른 결과가 조건을 만족하는지만 다시 검사한다.
#
# **시연용으로 고른 것**이라 미라벨 웨이퍼 전체를 대표하지 않는다(화면에도 명시).
# 미라벨 전체의 무작위 모습은 앱의 '미라벨 웨이퍼 무작위 추출'(위 1,000장 풀)로 본다.

# %%
from scipy import ndimage   # noqa: E402

SINGLES = [   # (lotName, waferIndex, 눈으로 본 모양 — 모델 판정 아님)
    ("lot11705", 23, "불량이 거의 없음"),
    ("lot23658", 3, "중앙 덩어리"),
    ("lot33059", 23, "고리 모양"),
    ("lot17175", 5, "중간 위치 덩어리"),
    ("lot2009", 4, "가장자리 덩어리"),
    ("lot40273", 8, "가장자리 넓은 덩어리"),
    ("lot2614", 5, "가장자리 둘레 전반"),
    ("lot23641", 5, "대각선 긁힘"),
]
DEMO_LOT = "lot42002"          # 25장 모두 중앙부에 덩어리 — 화면 3의 Lot 응집도와 같은 이야기
BIN_SOURCE = ("lot23658", 3)   # Bin 코드 형식 예시로 바꿀 웨이퍼
S8 = np.ones((3, 3), bool)
zlo, zhi = ref["zero_frac_q"]
dhi = ref["defect_ratio_q"][1]


def check_demo(m: np.ndarray, need_blob: bool) -> dict:
    """위 1번 조건을 다시 계산한다. 목록을 손으로 고정했으니, 조건 위반이 섞이지 않았는지 확인용."""
    die, d = m >= 1, m == 2
    zf, dr, _ = map_stats(m)
    rd, rdd = die.sum(1), d.sum(1)
    cd, cdd = die.sum(0), d.sum(0)
    rs, cs = rd >= 0.6 * rd.max(), cd >= 0.6 * cd.max()
    stripes = int(((rdd[rs] / rd[rs]) >= 0.9).sum() + ((cdd[cs] / cd[cs]) >= 0.9).sum())
    inner = d & ndimage.binary_erosion(die, S8, border_value=0)
    lab, n = ndimage.label(inner, structure=S8)
    big = int(np.bincount(lab.ravel())[1:].max()) if n else 0
    share = big / max(int(inner.sum()), 1)
    ok = (zlo <= zf <= zhi and dr <= dhi and stripes == 0 and 25 <= min(m.shape) and max(m.shape) <= 80)
    if need_blob:
        ok = ok and big >= 20 and share >= 0.4
    return {"ok": bool(ok), "dr": dr, "stripes": stripes, "big": big}


def to_csv(m: np.ndarray, path: Path):
    pd.DataFrame(m).to_csv(path, header=False, index=False)


# 예전 시연 파일을 지우고 새로 쓴다 (samples/ 안에는 이 스크립트가 만든 것만 있다)
import shutil   # noqa: E402
for old in SAMPLES.iterdir():
    if old.is_dir():
        shutil.rmtree(old)
    else:
        old.unlink()

by_key = {(l, int(w)): i for i, l, w in zip(unlabeled.index, unlabeled["lotName"], unlabeled["waferIndex"])}
print("[4] 단일 웨이퍼 CSV")
for lot, w, desc in SINGLES:
    m = unlabeled.at[by_key[(lot, w)], "waferMap"].astype(np.uint8)
    c = check_demo(m, need_blob=(desc != "불량이 거의 없음"))
    if not c["ok"]:
        raise RuntimeError(f"{lot} w{w} 이 조건을 만족하지 않습니다: {c}")
    name = f"unlabeled_{lot}_w{w:02d}.csv"
    to_csv(m, SAMPLES / name)
    print(f"    {name:<30} {m.shape}  불량 {c['dr']*100:4.1f}%  덩어리 {c['big']:>3}다이  ({desc})")

# Lot 한 통 — 원본에 25행이 있고 그중 라벨 있는 행이 0개인지 전체 데이터에서 확인
lot_all = df[df["lotName"] == DEMO_LOT]
assert len(lot_all) == 25 and lot_all["failureType_clean"].isna().all(), "시연 Lot이 완전 미라벨이 아닙니다"
lot_dir = SAMPLES / DEMO_LOT
lot_dir.mkdir(exist_ok=True)
lot_rows = lot_all.sort_values("waferIndex")
lot_checks = [check_demo(m.astype(np.uint8), need_blob=False) for m in lot_rows["waferMap"]]
if not all(c["ok"] for c in lot_checks):
    raise RuntimeError(f"{DEMO_LOT}에 조건 위반 웨이퍼가 있습니다")
for _, r in lot_rows.iterrows():
    to_csv(r["waferMap"].astype(np.uint8), lot_dir / f"{DEMO_LOT}_w{int(r['waferIndex']):02d}.csv")
n_blob = sum(c["big"] >= 20 for c in lot_checks)
print(f"[5] Lot 한 통: {DEMO_LOT} (25장 전부 미라벨, 덩어리 20다이 이상 {n_blob}장) -> {lot_dir.name}/")
demo_lot = DEMO_LOT

# Bin 코드 형식 예시 — 같은 웨이퍼를 테스트 장비 형식으로 바꾼다
lot, w = BIN_SOURCE
m = unlabeled.at[by_key[(lot, w)], "waferMap"].astype(np.uint8)
bins = np.full(m.shape, "", dtype=object)
bins[m == 1] = "1"
bins[m == 2] = rng.choice(["3", "5", "7", "12"], size=int((m == 2).sum()))
bin_name = f"bincode_example_{lot}_w{w:02d}.csv"
pd.DataFrame(bins).to_csv(SAMPLES / bin_name, header=False, index=False)
print(f"[6] Bin 코드 형식 예시: {bin_name} (다이 없음=빈칸, Pass=1, Fail=3/5/7/12)")

# %%
joblib.dump({"pool": pool, "ref": ref, "demo_lot": demo_lot},
            ARTIFACTS / "unlabeled_samples.joblib", compress=COMPRESS)
out = ARTIFACTS / "unlabeled_samples.joblib"
print(f"\n[완료] {out.name} ({out.stat().st_size / 1e6:.2f} MB)")
