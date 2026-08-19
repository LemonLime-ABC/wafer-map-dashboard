# %% [markdown]
# # 06_gradcam — Phase 5 Grad-CAM 예측 근거 시각화
#
# 이 스크립트가 하는 일:
# 1. fold 0 모델(`artifacts/model_v2_flatten_fold0.pt`)과 그 fold의
#    검증셋(=이 모델이 학습에 안 쓴 데이터)을 불러온다
# 2. 클래스별로 "정답을 맞힌" 샘플 몇 개를 골라 Grad-CAM 히트맵을 그린다
# 3. **정량화**: 히트맵이 실제 불량 다이 위치에 얼마나 몰려있는지를
#    숫자로 재서(클래스별 평균), "모델이 진짜 근거를 보고 판단하는지"를
#    확인한다 — SECOM에서 "SHAP 원인 지목 정확도"를 낸 것과 같은 방식이다.
#
# **주의**: fold 0 모델은 fold 0의 검증셋을 학습에 쓰지 않았으므로,
# 여기서 보여주는 예시들은 전부 이 모델 입장에서 "처음 보는" 데이터다
# (학습 데이터를 재예측해 근거를 짜맞추는 것이 아님).

# %%
import sys
import base64
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(str(Path(__file__).resolve().parent))
from model import WaferCNN  # noqa: E402
from gradcam import GradCAM  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RNG_SEED = 42
N_FOLDS = 5
N_EXAMPLES_PER_CLASS = 3

CLASS_ORDER = ["none", "Center", "Donut", "Edge-Ring", "Edge-Loc",
               "Scratch", "Loc", "Random", "Near-full"]

# %% [markdown]
# ## 1. 모델과 fold 0 검증셋 로드

# %%
data = np.load(PROCESSED_DIR / "wm811k_sample.npz")
X_raw, y_encoded, class_names = data["X"], data["y_encoded"], data["class_names"]
n_classes = len(class_names)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
# fold 0의 (학습, 검증) 인덱스를 그대로 재현 — 05_train_evaluate.py와
# 완전히 같은 시드·같은 분할 방식이라 정확히 같은 검증셋을 얻는다.
train_idx0, val_idx0 = next(iter(skf.split(X_raw, y_encoded)))

model = WaferCNN(in_channels=1, n_classes=n_classes, head="flatten")
state = torch.load(ARTIFACTS_DIR / "model_v2_flatten_fold0.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"[1] fold 0 모델 로드 완료, 검증셋 {len(val_idx0):,}장")

gradcam = GradCAM(model, model.features[10])  # 마지막 Conv 블록의 ReLU (16x16x128)


def prepare_one(x_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((x_uint8.astype(np.float32) / 2.0)[None, None, :, :])


# %% [markdown]
# ## 2. 검증셋 전체에 대해 예측 + Grad-CAM 히트맵 정량 평가
#
# "히트맵 질량 집중도" = 히트맵 값의 합 중, 실제 불량 다이(값==2) 칸에
# 몰려있는 비율. 1에 가까울수록 "모델이 진짜 불량 위치를 보고 판단했다"는
# 뜻이고, 클래스 평균 비율(무작위로 보면 나올 값)보다 높으면 유의미하다.

# %%
print("[2] 검증셋 전체 Grad-CAM 정량 평가 중...")

overlap_by_class = {c: [] for c in class_names}
baseline_by_class = {c: [] for c in class_names}  # 통제군: 히트맵이 무작위였다면 나올 값
correct_examples = {c: [] for c in class_names}  # (idx, heatmap, prob) 저장용

for idx in val_idx0:
    x_t = prepare_one(X_raw[idx])
    true_label = int(y_encoded[idx])

    heatmap, pred_label, prob = gradcam.generate(x_t, target_class=None)

    if pred_label != true_label:
        continue  # 오분류 샘플은 "이 클래스라고 판단한 근거"가 왜곡되므로 정량 평가에서 제외

    cls = class_names[true_label]
    defect_mask = (X_raw[idx] == 2)
    heatmap_sum = heatmap.sum()
    if defect_mask.sum() == 0 or heatmap_sum < 1e-8:
        continue  # 불량 다이가 아예 없는 웨이퍼는 "위치 일치"를 잴 기준이 없음

    concentration = heatmap[defect_mask].sum() / heatmap_sum
    overlap_by_class[cls].append(concentration)

    # ---- 통제군: 히트맵이 완전히 무작위(균일)였다면 나왔을 값 ----
    # = 불량 다이가 전체 칸 중 차지하는 비율. 이보다 실제 집중도가
    # 높아야 "모델이 실제로 불량 위치를 보고 있다"고 말할 수 있다.
    baseline = defect_mask.sum() / defect_mask.size
    baseline_by_class[cls].append(baseline)

    if len(correct_examples[cls]) < N_EXAMPLES_PER_CLASS:
        correct_examples[cls].append((idx, heatmap.copy(), prob))

print("[2] 완료")

# %% [markdown]
# ## 3. 클래스별 예시 그림 (원본 / Grad-CAM 히트맵 / 오버레이)

# %%
print("[3] 예시 그림 생성 중...")

fig, axes = plt.subplots(len(CLASS_ORDER), N_EXAMPLES_PER_CLASS * 2,
                          figsize=(N_EXAMPLES_PER_CLASS * 3.2, len(CLASS_ORDER) * 1.7))

for row_i, cls in enumerate(CLASS_ORDER):
    examples = correct_examples.get(cls, [])
    for col_i in range(N_EXAMPLES_PER_CLASS):
        ax_img = axes[row_i, col_i * 2]
        ax_cam = axes[row_i, col_i * 2 + 1]
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_cam.set_xticks([]); ax_cam.set_yticks([])
        for spine in list(ax_img.spines.values()) + list(ax_cam.spines.values()):
            spine.set_visible(False)

        if col_i < len(examples):
            idx, heatmap, prob = examples[col_i]
            ax_img.imshow(X_raw[idx], cmap="gray_r", vmin=0, vmax=2)
            ax_cam.imshow(X_raw[idx], cmap="gray_r", vmin=0, vmax=2, alpha=0.4)
            ax_cam.imshow(heatmap, cmap="jet", alpha=0.55, vmin=0, vmax=1)
            ax_cam.set_title(f"p={prob:.2f}", fontsize=7)

        if col_i == 0:
            ax_img.set_ylabel(cls, rotation=0, labelpad=40, fontsize=10, ha="right", va="center")

fig.suptitle("클래스별 정답 예측 사례 — 원본(좌) / Grad-CAM 오버레이(우), 밝을수록 판단 근거", y=1.0)
plt.tight_layout()

cam_grid_path = OUTPUT_DIR / "phase5_gradcam_grid.png"
fig.savefig(cam_grid_path, dpi=120, bbox_inches="tight")
plt.close(fig)
with open(cam_grid_path, "rb") as f:
    img_grid_b64 = base64.b64encode(f.read()).decode("ascii")
print(f"[3] 저장됨: {cam_grid_path}")

# %% [markdown]
# ## 4. 정량화 결과 표

# %%
print("\n[4] === 클래스별 히트맵-실제불량위치 집중도 (통제군 대비) ===")
summary_lines = ["=== Grad-CAM 정량 평가 (fold 0 검증셋, 정답 맞힌 샘플만) ==="]
header = f"{'클래스':10s}{'실제집중도':>12s}{'통제군(무작위)':>14s}{'배율':>8s}{'표본 수':>10s}"
print(header)
summary_lines.append(header)
for cls in CLASS_ORDER:
    vals = overlap_by_class[cls]
    base_vals = baseline_by_class[cls]
    if len(vals) == 0:
        row = f"{cls:10s}{'N/A':>12s}{'N/A':>14s}{'N/A':>8s}{0:>10d}"
    else:
        actual = np.mean(vals)
        baseline = np.mean(base_vals)
        ratio = actual / baseline if baseline > 1e-8 else float("nan")
        row = f"{cls:10s}{actual:>12.3f}{baseline:>14.3f}{ratio:>7.1f}x{len(vals):>10d}"
    print(row)
    summary_lines.append(row)

overall_vals = [v for vs in overlap_by_class.values() for v in vs]
overall_base = [v for vs in baseline_by_class.values() for v in vs]
overall_line = (f"\n전체 평균 집중도: {np.mean(overall_vals):.3f} vs 통제군 {np.mean(overall_base):.3f} "
                 f"(배율 {np.mean(overall_vals)/np.mean(overall_base):.1f}x) (표본 {len(overall_vals):,}개)")
print(overall_line)
summary_lines.append(overall_line)
summary_lines.append(
    "\n※ '배율'이 1.0보다 크게 클수록, 모델이 무작위로 보는 것보다 실제 불량 "
    "위치에 훨씬 더 집중해서 판단했다는 뜻이다."
)

with open(OUTPUT_DIR / "phase5_gradcam_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"\n[완료] 요약 저장: {OUTPUT_DIR / 'phase5_gradcam_summary.txt'}")
print(f"[완료] 그림 저장: {cam_grid_path}")
