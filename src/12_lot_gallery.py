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

# 조건을 만족하는 Lot을 전부 담는다.
# 웨이퍼 맵은 값이 0/1/2 뿐이라 zlib으로 약 17배 압축된다(4.04MB -> 0.24MB).
# 그래서 개수를 인위적으로 줄일 이유가 없다 — 화면에서 필터로 좁히면 된다.
MIN_DEFECT = 3          # 불량이 이보다 적은 Lot은 응집도를 볼 게 없다
MAX_WAFERS = 25         # 한 Lot에서 보여줄 최대 장수 (FOUP 표준 용량과 동일)
COMPRESS = ("zlib", 9)

# %%
df = load_wm811k()

# Lot 전체를 보여주려면 정상(none)도 함께 담아야 한다.
# 불량만 담으면 "이 Lot의 25장 중 몇 장이 불량인가"라는 실제로 궁금한 그림이
# 빠진다. Training 라벨은 원 연구자가 균형 맞추려 뽑은 세트라 제외한다.
lab = df[
    (df["failureType_clean"].notna())
    & (df["trainTestLabel_clean"] == "Test")
].copy()
defect = lab[lab["failureType_clean"] != "none"]
print(f"[1] 대상 웨이퍼(Test 라벨 전체): {len(lab):,}장 "
      f"(그중 불량 {len(defect):,}장)")

# Lot 선정 기준은 '불량 장수'로 본다 — 정상만 25장인 Lot은 볼 게 없다.
d_counts = defect.groupby("lotName").size()
usable = d_counts[d_counts >= MIN_DEFECT]
print(f"    불량 {MIN_DEFECT}장 이상인 Lot: {len(usable):,}개")

# %% [markdown]
# ## Lot을 두 부류로 나눠 고른다
#
# 응집도가 높은 Lot만 보여주면 "좋은 것만 골랐다"는 반박을 받는다.
# **가장 뭉친 Lot과 가장 섞인 Lot을 함께** 담아서, 보는 사람이 직접 비교하게 한다.
# (CLAUDE.md 2.2절 (6) 통제군 원칙과 같은 취지)

# %%
import pandas as pd  # noqa: E402

rows = []
for lot, n_def in usable.items():
    pats = defect[defect["lotName"] == lot]["failureType_clean"]
    top_share = pats.value_counts().iloc[0] / n_def   # 최빈 불량 패턴의 비중
    rows.append({"lot": lot,
                 "n": int(n_def),                       # 불량 장수
                 "n_all": int((lab["lotName"] == lot).sum()),  # 라벨된 전체 장수
                 "share": float(top_share),
                 "n_kinds": int(pats.nunique())})
stat = pd.DataFrame(rows)

selected = stat.sort_values(["n", "n_kinds"], ascending=[False, True])
n_single = int((selected["share"] == 1.0).sum())
print(f"[2] Lot {len(selected)}개 전부 선정 "
      f"(단일 패턴 {n_single}개 / 혼재 {len(selected)-n_single}개)")

# %% [markdown]
# ## 웨이퍼 맵을 64x64로 줄여 담는다
#
# 원본은 크기가 제각각(632가지)이라 화면에 나란히 놓으려면 통일해야 한다.
# 픽셀 0/1/2는 밝기가 아니라 범주(다이없음/정상/불량)라서
# **최근접 이웃만** 쓴다 — 보간하면 실존하지 않는 중간값이 생긴다.

# %%
gallery = {}
for _, r in selected.iterrows():
    # waferIndex = 카세트 안의 슬롯 번호. 이 순서로 정렬해야
    # "몇 번 슬롯이 불량인가"라는 실제로 의미 있는 배열이 된다.
    sub = (lab[lab["lotName"] == r["lot"]]
           .sort_values("waferIndex")
           .head(MAX_WAFERS))
    maps, pats, idxs = [], [], []
    for _, row in sub.iterrows():
        maps.append(resize_nearest(row["waferMap"], 64).astype(np.uint8))
        pats.append(row["failureType_clean"])
        idxs.append(int(row["waferIndex"]) if not pd.isna(row["waferIndex"]) else -1)
    kinds = sorted(set(p for p in pats if p != "none"))
    # 화면 필터용: 이 Lot에서 가장 많이 나온 불량 패턴
    dpats = [p for p in pats if p != "none"]
    dominant = max(set(dpats), key=dpats.count) if dpats else "none"
    gallery[r["lot"]] = {
        "dominant": dominant,
        "maps": np.stack(maps),
        "patterns": pats,
        "wafer_idx": idxs,
        "n_shown": len(maps),                       # 화면에 뜨는 장수
        "n_defect": int(sum(p != "none" for p in pats)),
        "n_normal": int(sum(p == "none" for p in pats)),
        "n_defect_total": int(r["n"]),              # 이 Lot의 전체 불량 장수
        "n_kinds": int(r["n_kinds"]),
        "top_share": float(r["share"]),
        "kind_list": kinds,
        "group": "단일 패턴" if r["share"] == 1.0 else "혼재",
    }

total_maps = sum(len(v["maps"]) for v in gallery.values())
total_def = sum(v["n_defect"] for v in gallery.values())
print(f"[3] Lot {len(gallery)}개 / 웨이퍼 맵 {total_maps}장 수집 "
      f"(불량 {total_def}장 · 정상 {total_maps - total_def}장)")

out = ARTIFACTS / "lot_gallery.joblib"
joblib.dump(gallery, out, compress=COMPRESS)
print(f"\n[완료] 저장: {out.name}  ({out.stat().st_size/1e6:.2f} MB, "
      f"압축 {COMPRESS[0]} 레벨 {COMPRESS[1]})")

# 확인용 요약
print("\n--- 담긴 Lot 목록 ---")
for lot, v in list(gallery.items())[:6]:
    print(f"  {lot:<12} {v['group']:<8} 표시 {v['n_shown']:>2}장 "
          f"(불량 {v['n_defect']:>2} / 정상 {v['n_normal']:>2}) 패턴 {v['kind_list']}")
print(f"  ... 외 {max(0, len(gallery)-6)}개")
