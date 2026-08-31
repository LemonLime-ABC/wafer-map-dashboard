# %% [markdown]
# # 12_lot_gallery — Lot별 웨이퍼 맵 갤러리 데이터 생성
#
# 화면 3의 "Lot 응집도" 수치(2.92배)를 **눈으로 확인할 수 있게** 하는 것이 목적이다.
# 같은 Lot의 웨이퍼 맵을 나란히 놓고 보면, 숫자로만 본 응집도가 실제로 어떤
# 모습인지 바로 보인다.
#
# 왜 새로 만드나: `data/processed/wm811k_sample.npz`에는 X, y_encoded,
# class_names만 있고 **lotName이 없다.** Lot 정보를 쓰려면 원본에서 다시 뽑아야 한다.
#
# 필터 조건은 Phase 8과 동일하게 맞춘다:
#   - Test 라벨만 (Training은 원 연구자가 균형 맞추려 뽑은 세트라 Lot 구조를 왜곡)
#   - none 제외 (불량끼리 비교해야 의미가 있다)

# %%
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_wm811k       # noqa: E402
from resize_utils import resize_nearest   # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"

N_LOTS_PER_GROUP = 14   # 응집/혼재 각각 몇 개 Lot을 담을지
MIN_WAFERS = 4          # 웨이퍼가 이보다 적은 Lot은 비교가 무의미
MAX_WAFERS = 12         # 한 Lot에서 화면에 보여줄 최대 장수 (용량 제한)

# %%
df = load_wm811k()

lab = df[
    (df["failureType_clean"].notna())
    & (df["failureType_clean"] != "none")
    & (df["trainTestLabel_clean"] == "Test")
].copy()
print(f"[1] 대상 웨이퍼(Test·불량만): {len(lab):,}장")

counts = lab.groupby("lotName").size()
usable = counts[counts >= MIN_WAFERS]
print(f"    웨이퍼 {MIN_WAFERS}장 이상인 Lot: {len(usable):,}개")

# %% [markdown]
# ## Lot을 두 부류로 나눠 고른다
#
# 응집도가 높은 Lot만 보여주면 "좋은 것만 골랐다"는 반박을 받는다.
# **가장 뭉친 Lot과 가장 섞인 Lot을 함께** 담아서, 보는 사람이 직접 비교하게 한다.
# (CLAUDE.md 2.2절 (6) 통제군 원칙과 같은 취지)

# %%
rows = []
for lot, n in usable.items():
    pats = lab[lab["lotName"] == lot]["failureType_clean"]
    top_share = pats.value_counts().iloc[0] / n   # 최빈 패턴이 차지하는 비율
    rows.append({"lot": lot, "n": int(n), "share": float(top_share),
                 "n_kinds": int(pats.nunique())})

import pandas as pd  # noqa: E402
stat = pd.DataFrame(rows)

# 뭉친 쪽: 한 패턴이 100%면서 웨이퍼가 많은 순
cohesive = (stat[stat["share"] == 1.0]
            .sort_values("n", ascending=False)
            .head(N_LOTS_PER_GROUP))
# 섞인 쪽: 패턴 종류가 많은 순
mixed = (stat[stat["n_kinds"] >= 3]
         .sort_values(["n_kinds", "n"], ascending=False)
         .head(N_LOTS_PER_GROUP))

print(f"[2] 단일 패턴 Lot {len(cohesive)}개 / 혼재 Lot {len(mixed)}개 선정")

# %% [markdown]
# ## 웨이퍼 맵을 64x64로 줄여 담는다
#
# 원본은 크기가 제각각(632가지)이라 화면에 나란히 놓으려면 통일해야 한다.
# 픽셀 0/1/2는 밝기가 아니라 범주(다이없음/정상/불량)라서
# **최근접 이웃만** 쓴다 — 보간하면 실존하지 않는 중간값이 생긴다.

# %%
gallery = {}
for _, r in pd.concat([cohesive, mixed]).iterrows():
    sub = lab[lab["lotName"] == r["lot"]].head(MAX_WAFERS)
    maps, pats, idxs = [], [], []
    for orig_idx, row in sub.iterrows():
        maps.append(resize_nearest(row["waferMap"], 64).astype(np.uint8))
        pats.append(row["failureType_clean"])
        idxs.append(int(row["waferIndex"]) if not pd.isna(row["waferIndex"]) else -1)
    kinds = sorted(set(pats))
    gallery[r["lot"]] = {
        "maps": np.stack(maps),
        "patterns": pats,
        "wafer_idx": idxs,
        "n_total": int(r["n"]),
        "n_kinds": int(r["n_kinds"]),
        "top_share": float(r["share"]),
        "kind_list": kinds,
        "group": "단일 패턴" if r["share"] == 1.0 else "혼재",
    }

total_maps = sum(len(v["maps"]) for v in gallery.values())
print(f"[3] Lot {len(gallery)}개 / 웨이퍼 맵 {total_maps}장 수집")

out = ARTIFACTS / "lot_gallery.joblib"
joblib.dump(gallery, out)
print(f"\n[완료] 저장: {out.name}  ({out.stat().st_size/1e6:.2f} MB)")

# 확인용 요약
print("\n--- 담긴 Lot 목록 ---")
for lot, v in list(gallery.items())[:6]:
    print(f"  {lot:<12} {v['group']:<8} 웨이퍼 {len(v['maps'])}장 "
          f"(전체 {v['n_total']}장) 패턴 {v['kind_list']}")
print(f"  ... 외 {max(0, len(gallery)-6)}개")
