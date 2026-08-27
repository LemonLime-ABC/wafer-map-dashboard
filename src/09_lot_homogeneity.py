# %% [markdown]
# # 09_lot_homogeneity — "같은 Lot의 웨이퍼는 같은 불량 양상을 보이는가"
#
# ## 왜 이걸 재는가
# Lot(로트)은 같은 카세트에 담겨 **같은 공정 경로를 같은 순서로 통과한**
# 웨이퍼 묶음이다(Phase 1에서 Lot당 웨이퍼 수 중앙값 24, 최댓값 25로
# 실제 FOUP 표준 용량과 일치함을 확인했다). 그렇다면 공정에 문제가
# 생겼을 때 그 Lot 안의 웨이퍼들이 **비슷한 불량 패턴**을 보여야 자연스럽다.
#
# 이게 사실이라면 의미가 크다:
#   - 불량 원인이 "웨이퍼 개별"이 아니라 "설비/공정 조건"에 있다는 증거가 된다
#   - Phase 6 SPC 관리도를 Lot 단위로 묶는 것이 타당하다는 근거가 된다
# 반대로 사실이 아니라면, Lot 단위 집계 자체를 재검토해야 한다.
#
# ## 핵심: 통제군 없이는 아무 말도 할 수 없다
# "같은 Lot의 두 웨이퍼가 같은 라벨일 확률"을 그냥 재면 **무조건 높게 나온다.**
# 라벨의 85%가 none이라, 아무 웨이퍼 둘을 뽑아도 둘 다 none일 확률이
# 0.85 x 0.85 = 0.72나 되기 때문이다. 즉 Lot과 아무 상관이 없어도 72%가 나온다.
#
# 그래서 CLAUDE.md 2.2절 (6) "통제군을 둔다" 원칙을 그대로 적용한다:
#   통제군 = 라벨을 전체에서 무작위로 섞어(shuffle) Lot 구조를 파괴한 뒤
#            똑같은 계산을 한 것. "Lot이 아무 의미 없었다면 나왔을 값"이다.
# 실제값이 이 통제군보다 유의하게 높아야만 "Lot 안에서 뭉친다"고 말할 수 있다.
#
# ## 계산을 빠르게 하는 수학 (중요)
# 쌍을 하나하나 만들어 세면 수천만 번 반복이라 몇 분씩 걸린다. 그런데
# 조합 공식을 쓰면 셀 필요가 없다:
#
#   어떤 Lot에 라벨 k인 웨이퍼가 n_k장 있다면,
#   그 Lot에서 "둘 다 라벨 k인 쌍"의 개수 = C(n_k, 2) = n_k(n_k-1)/2
#
# 즉 **각 (Lot, 라벨) 조합의 개수만 세면** 쌍을 만들지 않고도 답이 나온다.
# pandas의 groupby로 개수를 세는 건 순식간이라, 전체가 몇 초에 끝난다.
# (같은 결과를 얻으면서 훨씬 빠른 길이 있으면 그쪽을 택한다.)

# %%
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_wm811k  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RNG_SEED = 42
N_SHUFFLES = 20  # 통제군 반복 횟수 (한 번만 하면 우연에 흔들리므로 평균낸다)

rng = np.random.default_rng(RNG_SEED)


def n_pairs(counts: np.ndarray) -> int:
    """개수 배열을 받아 C(n,2)의 합을 구한다. n<2면 0이 되도록 자동 처리됨."""
    c = counts.astype(np.int64)
    return int((c * (c - 1) // 2).sum())


# %% [markdown]
# ## 1. 데이터 로드 및 Lot별 라벨 수 파악

# %%
df = load_wm811k()
lab = df[df["failureType_clean"].notna()][
    ["lotName", "waferIndex", "failureType_clean", "trainTestLabel_clean"]
].copy()

print(f"\n[1] 라벨된 웨이퍼: {len(lab):,}장")

lot_counts = lab.groupby("lotName").size()
print(f"    라벨이 있는 Lot: {len(lot_counts):,}개")
print(f"    Lot당 라벨 수: 중앙값 {lot_counts.median():.0f}, "
      f"평균 {lot_counts.mean():.1f}, 최대 {lot_counts.max()}")
multi = lot_counts[lot_counts >= 2]
print(f"    라벨 2장 이상인 Lot: {len(multi):,}개 "
      f"({len(multi)/len(lot_counts)*100:.1f}%) — 이 Lot들만 비교 가능")


# %% [markdown]
# ## 2. 핵심 계산 — Lot 내 쌍 일치율 (실제 vs 통제군)
#
# 함수 하나로 만들어두면 실제 라벨에도, 섞은 라벨에도 **똑같은 계산**을
# 적용할 수 있다 — 비교가 공정해진다.

# %%
def pair_agreement_rate(lot_codes: np.ndarray, label_codes: np.ndarray,
                        n_labels: int) -> float:
    """
    같은 Lot 안의 모든 웨이퍼 쌍 중 라벨이 일치하는 비율.

    lot_codes / label_codes 는 정수로 인코딩된 배열이어야 한다
    (문자열보다 정수가 훨씬 빠르고, np.bincount를 쓸 수 있다).
    """
    # (Lot, 라벨) 조합마다 몇 장인지 세기 — 2차원을 1차원 키로 합쳐서 한 번에
    combo_key = lot_codes.astype(np.int64) * n_labels + label_codes
    same = n_pairs(np.bincount(combo_key))       # 분자: 라벨이 같은 쌍
    total = n_pairs(np.bincount(lot_codes))      # 분모: Lot 내 모든 쌍
    return same / total if total else np.nan


def run_comparison(sub: pd.DataFrame, tag: str) -> dict:
    """실제 라벨 vs 라벨 무작위 섞기(통제군)를 같은 방식으로 계산해 비교."""
    # 문법 설명: pd.factorize()는 문자열 배열을 0,1,2... 정수로 바꿔준다.
    # ("lot5","lot9","lot5") -> ([0,1,0], ["lot5","lot9"])
    lot_codes, _ = pd.factorize(sub["lotName"])
    label_codes, label_uniques = pd.factorize(sub["failureType_clean"])
    n_labels = len(label_uniques)

    total_pairs = n_pairs(np.bincount(lot_codes))
    obs_rate = pair_agreement_rate(lot_codes, label_codes, n_labels)

    # 통제군: 라벨만 무작위로 섞는다. Lot 크기 구성과 전체 라벨 비율은
    # 그대로 두고 "어느 웨이퍼에 어느 라벨이 붙는가"만 파괴한다.
    ctrl_rates = [
        pair_agreement_rate(lot_codes, rng.permutation(label_codes), n_labels)
        for _ in range(N_SHUFFLES)
    ]
    ctrl_rate = float(np.mean(ctrl_rates))
    ctrl_std = float(np.std(ctrl_rates))

    lift = obs_rate / ctrl_rate if ctrl_rate > 0 else np.nan
    z = (obs_rate - ctrl_rate) / ctrl_std if ctrl_std > 0 else np.nan

    print(f"\n  [{tag}]")
    print(f"    웨이퍼 {len(sub):,}장 / 비교 쌍 {total_pairs:,}개")
    print(f"    실제 일치율    : {obs_rate*100:6.2f}%")
    print(f"    통제군(무작위) : {ctrl_rate*100:6.2f}%  (±{ctrl_std*100:.3f}%p, {N_SHUFFLES}회 평균)")
    print(f"    배율           : {lift:.2f}x   (Z = {z:+.1f})")
    return {"tag": tag, "wafers": len(sub), "pairs": total_pairs,
            "observed": obs_rate, "control": ctrl_rate,
            "control_std": ctrl_std, "lift": lift, "z": z}


def keep_multi(frame: pd.DataFrame) -> pd.DataFrame:
    """라벨이 2장 이상인 Lot만 남긴다 (1장짜리는 쌍을 못 만들어 무의미)."""
    c = frame.groupby("lotName").size()
    return frame[frame["lotName"].isin(c[c >= 2].index)]


# %% [markdown]
# ## 3. 네 가지 조건에서 측정
#
# A) 전체 라벨(none 포함) — none이 85%라 결과가 none에 지배당한다.
# B) 불량만(none 제외) — "불량이 난 웨이퍼끼리 같은 종류의 불량인가".
#    공정 관점에서 진짜 궁금한 것은 이쪽이다.
# C) Test 라벨만 — Phase 6에서 원 연구자의 Training 세트가 인위적으로
#    다양하게 구성돼 Lot 순서 분석을 망가뜨린 전례가 있다. 같은 오염이
#    여기에도 있는지 반드시 확인해야 한다.
# D) Test + 불량만 — 위 둘을 모두 적용한, 가장 깨끗한 조건.

# %%
print("\n[2] Lot 내 라벨 일치율 — 실제 vs 통제군")

results = [
    run_comparison(keep_multi(lab), "A. 전체 라벨(none 포함)"),
    run_comparison(keep_multi(lab[lab["failureType_clean"] != "none"]),
                   "B. 불량만(none 제외)"),
    run_comparison(keep_multi(lab[lab["trainTestLabel_clean"] == "Test"]),
                   "C. Test 라벨만(none 포함)"),
    run_comparison(keep_multi(lab[(lab["trainTestLabel_clean"] == "Test") &
                                  (lab["failureType_clean"] != "none")]),
                   "D. Test + 불량만"),
]

# %% [markdown]
# ## 4. 패턴별로 나눠 보기
#
# "어떤 패턴이 특히 Lot 안에서 뭉치는가"를 본다.
# 물리적으로 예상하자면: 설비 상태에서 오는 패턴(Edge-Ring, Center 등)은
# Lot 전체에 영향을 주니 뭉쳐야 하고, 우발적 사건(Scratch — 특정 웨이퍼
# 한 장만 긁힘)은 덜 뭉쳐야 한다. 실제로 그런지 확인한다.

# %%
print("\n[3] 패턴별 Lot 응집도")


def per_pattern_table(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    패턴 p마다: (p끼리의 쌍) / (p가 한쪽 이상 낀 모든 쌍)
    = "같은 Lot의 다른 불량 웨이퍼도 같은 패턴일 확률"

    p가 낀 쌍의 수 = C(n_p,2) + n_p*(N_lot - n_p)
      앞항 = p끼리의 쌍, 뒷항 = p와 p아닌 것의 쌍
    """
    lot_codes, _ = pd.factorize(frame["lotName"])
    label_codes, label_names = pd.factorize(frame["failureType_clean"])
    n_lab = len(label_names)
    n_lots = lot_codes.max() + 1

    def rates(lc, labc):
        combo = lc.astype(np.int64) * n_lab + labc
        tbl = np.bincount(combo, minlength=n_lots * n_lab).reshape(n_lots, n_lab)
        lot_sizes = tbl.sum(axis=1, keepdims=True)
        same_p = (tbl * (tbl - 1) // 2).sum(axis=0)
        involved_p = same_p + (tbl * (lot_sizes - tbl)).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(involved_p > 0, same_p / involved_p, np.nan), involved_p

    obs, involved = rates(lot_codes, label_codes)
    ctrl = np.vstack([rates(lot_codes, rng.permutation(label_codes))[0]
                      for _ in range(N_SHUFFLES)]).mean(axis=0)

    out = pd.DataFrame({
        "패턴": label_names,
        "실제(%)": obs * 100,
        "통제군(%)": ctrl * 100,
        "배율": obs / ctrl,
        "관련 쌍 수": involved,
    }).sort_values("실제(%)", ascending=False).reset_index(drop=True)
    print(f"\n  [{tag}] '같은 Lot의 다른 불량 웨이퍼도 같은 패턴일 확률'")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return out


defect2 = keep_multi(lab[lab["failureType_clean"] != "none"])
pattern_df = per_pattern_table(defect2, "B조건: 전체 불량 (Training 포함)")

test_def2 = keep_multi(lab[(lab["trainTestLabel_clean"] == "Test") &
                           (lab["failureType_clean"] != "none")])
pattern_df_test = per_pattern_table(test_def2, "D조건: Test 라벨 불량만 (권장)")

lot_codes_d, _ = pd.factorize(defect2["lotName"])
label_codes_d, label_names_d = pd.factorize(defect2["failureType_clean"])
n_lab_d = len(label_names_d)
n_lots_d = lot_codes_d.max() + 1

# %% [markdown]
# ## 5. "Lot 전체가 단일 패턴" 비율 — 가장 직관적인 지표

# %%
print("\n[4] Lot 전체가 하나의 패턴인 비율 (불량만, none 제외)")


def uniform_lot_rate(lot_codes, label_codes):
    combo = lot_codes.astype(np.int64) * n_lab_d + label_codes
    tbl = np.bincount(combo, minlength=n_lots_d * n_lab_d).reshape(n_lots_d, n_lab_d)
    n_distinct = (tbl > 0).sum(axis=1)   # 그 Lot에 몇 종류의 패턴이 있나
    valid = tbl.sum(axis=1) >= 2         # 라벨 2장 이상인 Lot만
    return float((n_distinct[valid] == 1).mean())


obs_u = uniform_lot_rate(lot_codes_d, label_codes_d)
ctrl_u = float(np.mean([uniform_lot_rate(lot_codes_d, rng.permutation(label_codes_d))
                        for _ in range(N_SHUFFLES)]))
print(f"    실제   : {obs_u*100:.2f}%")
print(f"    통제군 : {ctrl_u*100:.2f}%")
print(f"    배율   : {obs_u/ctrl_u:.2f}x")

# %% [markdown]
# ## 6. 저장

# %%
summary_lines = [
    "=== Lot 내 불량 패턴 응집도 분석 ===",
    "",
    f"라벨된 웨이퍼 {len(lab):,}장 / 라벨 있는 Lot {len(lot_counts):,}개",
    f"Lot당 라벨 수: 중앙값 {lot_counts.median():.0f}, 평균 {lot_counts.mean():.1f}, 최대 {lot_counts.max()}",
    f"라벨 2장 이상인 Lot: {len(multi):,}개 ({len(multi)/len(lot_counts)*100:.1f}%)",
    "",
    f"--- Lot 내 쌍 일치율: 실제 vs 통제군(라벨 무작위 섞기 {N_SHUFFLES}회 평균) ---",
]
for r in results:
    summary_lines.append(
        f"{r['tag']:<26} 웨이퍼 {r['wafers']:>7,} 쌍 {r['pairs']:>10,} | "
        f"실제 {r['observed']*100:6.2f}% | 통제군 {r['control']*100:6.2f}% | "
        f"배율 {r['lift']:.2f}x | Z={r['z']:+.1f}"
    )
summary_lines += [
    "",
    "--- 패턴별 Lot 응집도: B조건 (전체 불량, Training 포함) ---",
    "'같은 Lot의 다른 불량 웨이퍼도 같은 패턴일 확률'",
    pattern_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"),
    "",
    "--- 패턴별 Lot 응집도: D조건 (Test 라벨 불량만 — 권장) ---",
    pattern_df_test.to_string(index=False, float_format=lambda v: f"{v:.2f}"),
    "",
    "--- Lot 전체가 하나의 패턴인 비율 (불량만) ---",
    f"실제 {obs_u*100:.2f}% vs 통제군 {ctrl_u*100:.2f}% (배율 {obs_u/ctrl_u:.2f}x)",
    "",
    "※ 통제군 = 라벨을 전체에서 무작위로 섞어 Lot 구조를 파괴한 뒤 같은 계산을 한 값.",
    "  'Lot이 아무 의미 없었다면 나왔을 값'이며, 실제값이 이보다 높아야만",
    "  'Lot 안에서 패턴이 뭉친다'고 말할 수 있다.",
    "※ 이 분석은 '같은 Lot이면 패턴이 비슷하다'는 통계적 연관을 보여줄 뿐,",
    "  '공정 조건이 원인이다'를 증명하지 않는다 — 이 데이터엔 설비ID·레시피가 없다.",
]
out_path = OUTPUT_DIR / "phase8_lot_homogeneity.txt"
out_path.write_text("\n".join(summary_lines), encoding="utf-8")
pattern_df.to_csv(OUTPUT_DIR / "phase8_lot_pattern_cohesion.csv",
                  index=False, encoding="utf-8-sig")
pattern_df_test.to_csv(OUTPUT_DIR / "phase8_lot_pattern_cohesion_test.csv",
                       index=False, encoding="utf-8-sig")
print(f"\n[완료] 저장: {out_path}")
