# %% [markdown]
# # 08_precompute_dashboard — Phase 7 대시보드용 데이터 사전 계산
#
# SECOM `app.py`와 같은 원칙: 대시보드는 무거운 계산을 직접 하지 않고
# 여기서 미리 계산해 joblib 하나로 저장해둔 것만 불러 쓴다. 이러면
# Streamlit Cloud처럼 자원이 제한된 환경에서도 대시보드가 가볍게 뜬다.
#
# 이 스크립트가 준비하는 것:
# 1. fold 0 검증셋(모델이 학습에 안 쓴 데이터) 전체의 예측 결과
#    — 화면 1(웨이퍼 진단)에서 쓸 목록
# 2. 최종(Flatten) 5-Fold 혼동 행렬에서 클래스별 정밀도/재현율/F1 재계산
#    — 화면 2(패턴별 성능)에서 쓸 표
# 3. Phase 6에서 이미 만든 SPC 번들, 수율 기여도 표를 그대로 포함
#    — 화면 3(Lot 모니터링), 화면 4(공정 매핑)에서 쓸 데이터
# 4. Phase 4 개선 과정 전체 요약 텍스트 — 화면 5(방법론)에서 쓸 내용

# %%
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent))
from model import WaferCNN  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RNG_SEED = 42
N_FOLDS = 5

# %% [markdown]
# ## 1. fold 0 검증셋 예측

# %%
data = np.load(PROCESSED_DIR / "wm811k_sample.npz")
X_raw, y_encoded, class_names = data["X"], data["y_encoded"], data["class_names"]
n_classes = len(class_names)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
_, val_idx0 = next(iter(skf.split(X_raw, y_encoded)))

model = WaferCNN(in_channels=1, n_classes=n_classes, head="flatten")
state = torch.load(ARTIFACTS_DIR / "model_v2_flatten_fold0.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

X_val = (X_raw[val_idx0].astype(np.float32) / 2.0)[:, None, :, :]
with torch.no_grad():
    logits = model(torch.from_numpy(X_val))
    probs = torch.softmax(logits, dim=1).numpy()
pred_idx = probs.argmax(axis=1)

wafer_table = pd.DataFrame({
    "orig_idx": val_idx0,
    "true_label": class_names[y_encoded[val_idx0]],
    "pred_label": class_names[pred_idx],
    "pred_prob": probs.max(axis=1),
    "correct": class_names[y_encoded[val_idx0]] == class_names[pred_idx],
})

# 화면 1에서 "1등 말고 나머지 8개 클래스는 몇 %로 봤는가"까지 보여주기 위해
# 최댓값만이 아니라 9개 클래스 확률을 전부 보관한다. 최댓값 하나만 보면
# "0.99로 확신"인지 "0.35 vs 0.33으로 간신히 이김"인지 구분이 안 되는데,
# 이 둘은 실무에서 신뢰도가 전혀 다르다.
probs_all = probs.astype(np.float32)  # (검증셋 장수, 9)
print(f"    전체 확률 배열 보관: {probs_all.shape} "
      f"(행별 합계 검증: {probs_all.sum(axis=1).min():.4f}~{probs_all.sum(axis=1).max():.4f})")
print(f"[1] fold 0 검증셋 {len(wafer_table):,}장 예측 완료 "
      f"(정확도 {wafer_table['correct'].mean():.3f})")

# %% [markdown]
# ## 2. 최종 혼동 행렬에서 클래스별 지표 재계산

# %%
cm_data = np.load(OUTPUT_DIR / "phase4_confusion_matrix_v2_flatten.npz")
cm = cm_data["cm"]
cm_class_names = cm_data["class_names"]

per_class_rows = []
for i, cls in enumerate(cm_class_names):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    support = cm[i, :].sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    per_class_rows.append({
        "class": cls, "precision": precision, "recall": recall,
        "f1": f1, "support": int(support),
    })
per_class_df = pd.DataFrame(per_class_rows)
print("[2] 클래스별 지표 재계산 완료")
print(per_class_df.to_string(index=False))

with open(OUTPUT_DIR / "phase4_metrics_v2_flatten.json", encoding="utf-8") as f:
    overall_metrics = json.load(f)

# %% [markdown]
# ## 3. Phase 6 SPC 번들 + 수율 기여도 로드 (이미 만들어진 것 재사용)

# %%
spc_bundle = joblib.load(ARTIFACTS_DIR / "spc_bundle.joblib")
contribution_df = pd.read_csv(OUTPUT_DIR / "phase6_yield_contribution.csv")
print("[3] Phase 6 산출물 로드 완료")

# %% [markdown]
# ## 3b. Phase 8 Lot 응집도 — "같은 Lot이면 같은 불량이 나오는가"
#
# 화면 3(Lot 수율 모니터링)에 올릴 데이터다. Test 라벨 + none 제외 조건(D)만
# 쓴다 — Training 라벨을 섞으면 원 연구자가 학습셋을 균형 있게 뽑느라
# 생긴 인위적 뭉침이 결과를 부풀린다(Phase 6에서 겪은 것과 같은 함정).

# %%
lot_pattern_df = pd.read_csv(OUTPUT_DIR / "phase8_lot_pattern_cohesion_test.csv")

# 전체 요약 수치는 09_lot_homogeneity.py가 출력한 txt에서 D조건 줄만 파싱한다.
# (수치를 여기에 다시 적으면 분석을 다시 돌렸을 때 화면이 조용히 거짓말을 한다)
import re  # noqa: E402

lot_summary = None
_txt = (OUTPUT_DIR / "phase8_lot_homogeneity.txt").read_text(encoding="utf-8")
for line in _txt.splitlines():
    if line.startswith("D."):
        nums = re.findall(r"[\d,]+\.?\d*", line.replace("D.", ""))
        lot_summary = {
            "wafers": int(nums[0].replace(",", "")),
            "pairs": int(nums[1].replace(",", "")),
            "observed": float(nums[2]),
            "control": float(nums[3]),
            "lift": float(nums[4]),
            "z": float(nums[5]),
        }
        break
if lot_summary is None:
    raise RuntimeError("phase8_lot_homogeneity.txt에서 D조건 줄을 찾지 못했습니다")
print(f"[3b] Lot 응집도 로드: 실제 {lot_summary['observed']}% vs "
      f"통제군 {lot_summary['control']}% (배율 {lot_summary['lift']}x)")

# %% [markdown]
# ## 4. Phase 4 개선 과정 텍스트 요약 (방법론 화면용)

# %%
def read_text(name):
    p = OUTPUT_DIR / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


phase4_journey = {
    "v1_summary": read_text("phase4_train_evaluate_summary.txt"),
    "v2_summary": read_text("phase4_train_evaluate_summary_v2_flatten.txt"),
    "head_check": read_text("phase4_head_architecture_check.txt"),
    "resolution_check": read_text("phase4_resolution_check.txt"),
    "aux_check": read_text("phase4_aux_feature_check.txt"),
}

with open(OUTPUT_DIR / "phase4_metrics.json", encoding="utf-8") as f:
    metrics_v1 = json.load(f)

# %% [markdown]
# ## 5. 전부 하나로 묶어 저장

# %%
bundle = {
    "class_names": class_names,
    "wafer_table": wafer_table,        # 화면 1
    "probs_all": probs_all,            # 화면 1 (9개 클래스 전체 확률)
    "X_val0": X_raw[val_idx0],         # 화면 1 (원본 64x64 이미지, uint8)
    "cm": cm,                          # 화면 2
    "per_class_df": per_class_df,      # 화면 2
    "overall_metrics": overall_metrics,  # 화면 2, 5
    "metrics_v1": metrics_v1,          # 화면 5 (1차 결과, 비교용)
    "spc_bundle": spc_bundle,          # 화면 3
    "lot_pattern_df": lot_pattern_df,  # 화면 3 (Lot 응집도, 패턴별)
    "lot_summary": lot_summary,        # 화면 3 (Lot 응집도, 전체)
    "contribution_df": contribution_df,  # 화면 4
    "phase4_journey": phase4_journey,  # 화면 5
}

dashboard_path = ARTIFACTS_DIR / "dashboard_bundle.joblib"
joblib.dump(bundle, dashboard_path)
print(f"\n[완료] 대시보드 번들 저장: {dashboard_path} "
      f"({dashboard_path.stat().st_size / 1e6:.1f} MB)")
