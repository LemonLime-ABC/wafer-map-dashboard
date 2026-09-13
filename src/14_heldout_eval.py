# %% [markdown]
# # 14_heldout_eval — 모델이 한 번도 못 본 라벨 웨이퍼로 "현장 투입 시" 성능 확인
#
# ## 왜 필요한가
# 5-Fold 결과(macro-F1 0.8728)는 **균형을 맞춘 12,763장** 안에서 잰 값이다.
# 이 샘플은 정상(none)을 147,431장 중 2,000장(15.7%)만 넣었다. 그런데 실제 생산
# 분포는 정상이 85%다. 화면 6(즉석 판정)에 현장 웨이퍼를 넣으면 모델은
# **학습 때와 전혀 다른 비율**의 입력을 받게 된다.
#
# SECOM에서 "정확도 92.7%인데 불량 0장 검출"을 겪었던 것(CLAUDE.md 2.2 ①)과
# 방향만 반대인 같은 종류의 문제다 — 이번엔 불량을 **과하게** 부를 수 있다.
#
# ## 어떻게 재나
# 라벨된 172,950장 중 학습 샘플 12,763장에 **안 들어간** 웨이퍼는 5개 fold 모델
# 누구도 본 적이 없다. 04_preprocessing.py의 샘플링을 그대로 재현해(시드 42)
# 이 웨이퍼들을 골라낸다. 소수 클래스 4종(Scratch/Random/Donut/Near-full)은 전량이
# 학습에 들어가서 남는 게 없으므로, 그 4종은 5-Fold OOF 혼동 행렬로 대신한다.
#
# ## 결과물
# - artifacts/heldout_eval.joblib — 화면 6이 읽는 수치와 설명 문구
# - outputs/phase9_heldout_eval.txt — 사람이 읽는 요약

# %%
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_wm811k                            # noqa: E402
from resize_utils import resize_nearest                        # noqa: E402
from live_inference import load_fold_models, predict_probs     # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RNG_SEED = 42           # 04_preprocessing.py와 반드시 같아야 한다
MAJORITY_CAP = 2000
MAJ = ["none", "Edge-Ring", "Edge-Loc", "Center", "Loc"]
MIN = ["Scratch", "Random", "Donut", "Near-full"]
N_NONE_EVAL = 20000     # 정상은 14.5만 장이라 전부 돌리면 25분 — 2만 장이면 비율 오차 약 ±0.3%p
torch.set_num_threads(8)

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(msg)


# %% [markdown]
# ## 1. 학습 샘플 재현 → 본 적 없는 웨이퍼 골라내기
#
# 재현이 조금이라도 틀리면 "본 적 없는 웨이퍼"에 학습 웨이퍼가 섞여 성능이 부풀려진다.
# 그래서 재현한 샘플을 리사이즈해 저장된 npz와 **픽셀 단위로 완전히 같은지** 확인하고,
# 다르면 멈춘다.

# %%
df = load_wm811k(verbose=False)
labeled = df[df["failureType_clean"].notna()].reset_index(drop=True)

parts, used = [], set()
for c in MAJ:
    sub = labeled[labeled["failureType_clean"] == c]
    s = sub.sample(min(MAJORITY_CAP, len(sub)), random_state=RNG_SEED)
    parts.append(s)
    used |= set(s.index)
for c in MIN:
    s = labeled[labeled["failureType_clean"] == c]
    parts.append(s)
    used |= set(s.index)

sample_df = pd.concat(parts)
X_rep = np.stack([resize_nearest(m, 64) for m in sample_df["waferMap"]]).astype(np.uint8)
X_npz = np.load(PROJECT_ROOT / "data" / "processed" / "wm811k_sample.npz")["X"]
if not np.array_equal(X_rep, X_npz):
    raise RuntimeError("학습 샘플 재현 실패 — 본 적 없는 웨이퍼를 골라낼 수 없습니다.")
log(f"[1] 학습 샘플 {len(sample_df):,}장 재현 — 저장된 npz와 픽셀 단위 완전 일치")

held = labeled[~labeled.index.isin(used)]
none_part = held[held["failureType_clean"] == "none"].sample(N_NONE_EVAL, random_state=7)
held = pd.concat([none_part, held[held["failureType_clean"] != "none"]])
log(f"    본 적 없는 라벨 웨이퍼(평가용): {held['failureType_clean'].value_counts().to_dict()}")

# %% [markdown]
# ## 2. 5개 fold 모델로 판정

# %%
bundle = joblib.load(ARTIFACTS / "dashboard_bundle.joblib")
cn = list(bundle["class_names"])
models = load_fold_models(ARTIFACTS, len(cn))

chunks = []
for i in range(0, len(held), 5000):
    chunks.append(predict_probs(list(held["waferMap"].iloc[i:i + 5000]), models))
pr = np.concatenate(chunks, axis=1)                 # (5, N, 9)
true = held["failureType_clean"].to_numpy()
ti = np.array([cn.index(t) for t in true])

p0 = pr[0]                                          # 화면 6의 대표 판정 = fold 0
pred = p0.argmax(1)
srt = np.sort(p0, axis=1)[:, ::-1]
gap = srt[:, 0] - srt[:, 1]
votes = (pr.argmax(2) == pred[None, :]).sum(0)
wrong = pred != ti
is_none = true == "none"
log(f"[2] {len(held):,}장 판정 완료")

# %% [markdown]
# ## 3. 클래스별 — 본 적 없는 웨이퍼에서도 5-Fold 결과만큼 맞히는가

# %%
per_class = []
log("\n[3] 클래스별 정답률 (fold 0, 본 적 없는 웨이퍼)")
for c in MAJ:
    m = true == c
    dist = pd.Series(np.array(cn)[pred[m]]).value_counts(normalize=True)
    top = dist.drop(c, errors="ignore").head(3)
    per_class.append({"class": c, "n": int(m.sum()), "recall": float(dist.get(c, 0.0)),
                      "top_confusions": ", ".join(f"{k} {v*100:.1f}%" for k, v in top.items())})
    log(f"    {c:<9} {m.sum():>6,}장  정답 {dist.get(c, 0)*100:5.1f}%  | 오판 상위: {per_class[-1]['top_confusions']}")
per_class = pd.DataFrame(per_class)
none_as_defect = float((pred[is_none] != cn.index("none")).mean())

# %% [markdown]
# ## 4. '재확인 권장' 표시가 실제로 오판을 걸러내는가
#
# 후보 두 가지를 같은 데이터로 비교한다.
# - 1-2등 확률 격차 20%p 미만 (fold 0 검증셋에서 오분류율 6배 차이가 났던 기준)
# - 5개 fold 모델 판정이 만장일치가 아님
# 좋은 표시는 **오판은 많이 걸고, 맞힌 판정은 적게 건다.**

# %%
fp = is_none & (pred != cn.index("none"))          # 정상을 불량으로 판정(오경보)
tp_def = (~is_none) & (pred == ti)                  # 불량을 맞힌 판정
flag_gap, flag_vote = gap < 0.20, votes < 5
flag = {
    "fp_n": int(fp.sum()),
    "fp_caught_gap": float(flag_gap[fp].mean()),
    "fp_caught_vote": float(flag_vote[fp].mean()),
    "fp_caught_either": float((flag_gap | flag_vote)[fp].mean()),
    "tp_flagged_gap": float(flag_gap[tp_def].mean()),
    "tp_flagged_vote": float(flag_vote[tp_def].mean()),
    "tp_flagged_either": float((flag_gap | flag_vote)[tp_def].mean()),
    "none_false_alarm_unanimous": float((fp & ~flag_vote).sum() / is_none.sum()),
}
log("\n[4] 정상 웨이퍼 오경보를 표시가 잡아내는 비율")
log(f"    오경보 {flag['fp_n']:,}장 (정상의 {none_as_defect*100:.1f}%)")
log(f"    격차<20%p {flag['fp_caught_gap']*100:.1f}% / 만장일치 아님 {flag['fp_caught_vote']*100:.1f}% / 둘 중 하나 {flag['fp_caught_either']*100:.1f}%")
log(f"    (대가) 맞힌 불량 판정이 표시되는 비율: 격차 {flag['tp_flagged_gap']*100:.1f}% / 만장일치 {flag['tp_flagged_vote']*100:.1f}% / 둘 중 하나 {flag['tp_flagged_either']*100:.1f}%")
log(f"    만장일치 불량 판정만 남기면 정상 웨이퍼 오경보율 {none_as_defect*100:.1f}% → {flag['none_false_alarm_unanimous']*100:.1f}%")

votes_table = pd.DataFrame([
    {"votes": v, "n": int((votes == v).sum()), "error": float(wrong[votes == v].mean())}
    for v in range(1, 6) if (votes == v).sum()
])
log("\n    일치도별 오분류율 (평가셋 클래스 혼합)")
for _, r in votes_table.iterrows():
    log(f"    {int(r.votes)}/5 일치: {int(r.n):>6,}장  오분류 {r.error*100:5.1f}%")

# %% [markdown]
# ## 5. 현장 비율을 가정한 추정 — "불량 판정 중 실제 정상은 얼마나"
#
# 판정 비율 = Σ(실제 클래스 비율 × 그 클래스가 해당 판정을 받을 확률).
# 실제 클래스 비율은 라벨된 WM-811K 전체 비율(정상 85.24%)을 가정한다.
# 받을 확률은 본 적 없는 웨이퍼가 있는 5종은 위 측정값, 소수 4종은 5-Fold OOF
# 혼동 행렬 행을 쓴다. **서로 다른 두 출처를 섞은 추정치**라는 점을 명시한다.

# %%
prior = labeled["failureType_clean"].value_counts(normalize=True)
cm = bundle["cm"].astype(float)
cmn = cm / cm.sum(axis=1, keepdims=True)
mass = np.zeros((9, 9))                              # [실제, 판정] 비율
for t in cn:
    m = true == t
    rate = np.bincount(pred[m], minlength=9) / m.sum() if m.sum() else cmn[cn.index(t)]
    mass[cn.index(t)] = prior[t] * rate
col = mass.sum(axis=0)
k_none = cn.index("none")
defect_idx = [j for j, c in enumerate(cn) if c != "none"]
field = pd.DataFrame([
    {"pred": cn[j], "call_rate": float(col[j]), "share_actually_none": float(mass[k_none, j] / col[j])}
    for j in defect_idx
]).sort_values("share_actually_none", ascending=False)
share_none_all = float(mass[k_none, defect_idx].sum() / col[defect_idx].sum())
npv = float(mass[k_none, k_none] / col[k_none])
log(f"\n[5] 현장 비율(정상 {prior['none']*100:.2f}%) 가정 추정")
for _, r in field.iterrows():
    log(f"    {r['pred']:<10} 판정 비율 {r.call_rate*100:5.2f}% | 그중 실제 정상 {r.share_actually_none*100:5.1f}%")
log(f"    불량 판정 전체 중 실제 정상 {share_none_all*100:.1f}% / 정상 판정 중 실제 정상 {npv*100:.1f}%")

# %% [markdown]
# ## 6. 화면 6에 들어갈 문구 — 전부 위 계산값으로 만든다 (하드코딩 금지)

# %%
worst = field.head(3)
headline = (f"⚠️ 먼저 읽을 것 — 현장 비율에서는 불량 판정의 약 {share_none_all*100:.0f}%가 "
            "실제로는 정상 웨이퍼로 추정됩니다")
summary_md = f"""
**왜 이런가.** 모델은 정상을 **15.7%**만 넣어 균형을 맞춘 샘플로 학습했습니다. 실제 생산에서는
정상이 **{prior['none']*100:.1f}%**입니다. 정상 웨이퍼 수가 압도적으로 많으니, 그중 일부만 잘못 불러도
불량 판정 목록이 정상 웨이퍼로 채워집니다.

**어떻게 확인했나.** 라벨이 있지만 학습 샘플에 들어가지 않아 **5개 모델 누구도 본 적 없는** 웨이퍼
{len(held):,}장으로 재봤습니다 (학습 샘플 재현은 픽셀 단위로 완전 일치 확인).

| 실제 클래스 | 본 적 없는 웨이퍼 | 정답률 | 주요 오판 |
|---|---|---|---|
""" + "\n".join(f"| {r['class']} | {r.n:,}장 | {r.recall*100:.1f}% | {r.top_confusions} |"
                for _, r in per_class.iterrows()) + f"""

형태 분류 자체는 본 적 없는 웨이퍼에서도 5-Fold 결과와 비슷하게 유지됩니다. 문제는
**정상 웨이퍼의 {none_as_defect*100:.1f}%를 불량으로 부른다**는 점입니다.

**현장 비율을 가정한 추정** — 불량 판정 중 실제 정상 비율이 가장 높은 판정:
{", ".join(f"**{r['pred']}** {r.share_actually_none*100:.0f}%" for _, r in worst.iterrows())}.
반대로 **정상 판정은 {npv*100:.1f}%가 실제 정상**입니다.
(소수 클래스 4종은 본 적 없는 웨이퍼가 없어 5-Fold 결과로 대신한, 출처가 섞인 추정치입니다.)

**그래서 이 화면은 '5개 모델 만장일치'를 함께 봅니다.** 정상 웨이퍼 오경보 중
**{flag['fp_caught_vote']*100:.0f}%는 만장일치가 아니었고**, 맞힌 불량 판정은 {flag['tp_flagged_vote']*100:.0f}%만
만장일치가 아니었습니다. 만장일치 불량 판정만 남기면 정상 웨이퍼 오경보율이
**{none_as_defect*100:.1f}% → {flag['none_false_alarm_unanimous']*100:.1f}%**로 줄어듭니다.
참고로 1-2등 확률 격차 기준은 이 오경보를 {flag['fp_caught_gap']*100:.0f}%밖에 못 잡았습니다.

이 표시는 검사를 줄이는 용도가 아니라, 전수검사를 전제로 **사람이 먼저 다시 볼 순서**를 정하는 용도입니다.
"""
v5 = votes_table.set_index("votes")
votes_caption = (
    "본 적 없는 라벨 웨이퍼에서 잰 일치도별 오분류율: "
    + " · ".join(f"{int(v)}/5 → {v5.loc[v, 'error']*100:.0f}%" for v in sorted(v5.index, reverse=True))
    + ". 5개 모델은 서로 다른 80%로 학습됐습니다."
)
flag_caption = (
    f"재확인 권장 = 5개 모델 만장일치가 아니거나 1-2등 격차 20%p 미만인 웨이퍼(입력 검사 경고는 별도 열). "
    f"본 적 없는 웨이퍼에서 정상 웨이퍼 오경보의 {flag['fp_caught_either']*100:.0f}%를 걸렀고, "
    f"맞힌 불량 판정은 {flag['tp_flagged_either']*100:.0f}%를 함께 걸었습니다."
)

joblib.dump({
    "headline": headline, "summary_md": summary_md,
    "votes_caption": votes_caption, "flag_caption": flag_caption,
    "votes_error": {int(v): float(v5.loc[v, "error"]) for v in v5.index},
    "per_class": per_class, "votes_table": votes_table, "field_estimate": field,
    "flag": flag, "none_as_defect": none_as_defect,
    "share_none_all": share_none_all, "npv": npv, "n_eval": int(len(held)),
}, ARTIFACTS / "heldout_eval.joblib", compress=("zlib", 6))
(OUTPUT_DIR / "phase9_heldout_eval.txt").write_text("\n".join(log_lines), encoding="utf-8")
log(f"\n[완료] artifacts/heldout_eval.joblib, outputs/phase9_heldout_eval.txt")
