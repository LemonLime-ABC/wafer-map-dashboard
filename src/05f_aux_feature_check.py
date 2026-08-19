# %% [markdown]
# # 05f_aux_feature_check — 선형성 보조 특징 효과 검증
#
# Flatten head(macro-F1=0.827, 05b)에 `shape_features.compute_elongation`
# 값을 보조 입력으로 추가했을 때 실제로 좋아지는지 2-Fold x 15epoch로
# 빠르게 확인한다. Scratch/Loc F1을 특히 주목한다.

# %%
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parent))
from model import WaferCNN  # noqa: E402
from augment import augment_batch  # noqa: E402
from shape_features import compute_elongation_batch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RNG_SEED = 42
N_FOLDS_QUICK = 2
EPOCHS_QUICK = 15
BATCH_SIZE = 64
LR = 1e-3
MINORITY_CLASSES = ["Scratch", "Random", "Donut", "Near-full"]

# 05b에서 이미 확인한 "보조 특징 없는" 결과 (재사용, 다시 안 돌림)
KNOWN_NOAUX_MACRO_F1 = 0.8270
KNOWN_NOAUX_PER_CLASS = {
    "Center": 0.9312, "Donut": 0.9018, "Edge-Loc": 0.8214, "Edge-Ring": 0.9657,
    "Loc": 0.7047, "Near-full": 0.8490, "Random": 0.9045, "Scratch": 0.5261, "none": 0.8386,
}

torch.manual_seed(RNG_SEED)


def prepare_input(X: np.ndarray) -> np.ndarray:
    return (X.astype(np.float32) / 2.0)[:, None, :, :]


def run_one_fold(X_raw, y_encoded, y_str, aux_all, class_names, train_idx, val_idx) -> dict:
    n_classes = len(class_names)
    y_train_lab = y_str[train_idx]
    minority_mask = np.isin(y_train_lab, MINORITY_CLASSES)

    X_aug_raw, y_aug_lab = augment_batch(X_raw[train_idx][minority_mask], y_train_lab[minority_mask])
    # 증강된 이미지(회전/뒤집기)도 선형성은 원본과 같다 — 선을 돌리거나
    # 뒤집어도 "얼마나 길쭉한지"는 안 바뀌므로, 원본 aux 값을 그대로 8번 복제한다.
    aux_train_minor = aux_all[train_idx][minority_mask]
    aux_aug = np.repeat(aux_train_minor, 8)

    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_aug_enc = np.array([label_to_idx[c] for c in y_aug_lab])

    X_train_final_raw = np.concatenate([X_raw[train_idx][~minority_mask], X_aug_raw], axis=0)
    y_train_final_enc = np.concatenate([y_encoded[train_idx][~minority_mask], y_aug_enc], axis=0)
    aux_train_final = np.concatenate([aux_all[train_idx][~minority_mask], aux_aug], axis=0)

    class_counts_final = np.bincount(y_train_final_enc, minlength=n_classes)
    class_weights = 1.0 / np.maximum(class_counts_final, 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    X_train = prepare_input(X_train_final_raw)
    X_val = prepare_input(X_raw[val_idx])
    y_val_enc = y_encoded[val_idx]
    aux_val = aux_all[val_idx].reshape(-1, 1).astype(np.float32)
    aux_train_final = aux_train_final.reshape(-1, 1).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(aux_train_final), torch.from_numpy(y_train_final_enc)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = WaferCNN(in_channels=1, n_classes=n_classes, head="flatten", n_aux_features=1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    model.train()
    for epoch in range(EPOCHS_QUICK):
        ep_start = time.time()
        for xb, ab, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb, aux=ab), yb)
            loss.backward()
            opt.step()
        print(f"      epoch {epoch + 1}/{EPOCHS_QUICK} 완료 ({time.time()-ep_start:.1f}초)", flush=True)

    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val), aux=torch.from_numpy(aux_val)).argmax(dim=1).numpy()

    return {"y_val_enc": y_val_enc, "val_pred": val_pred}


# %% [markdown]
# ## 실행

# %%
print("[1] 데이터 로드 및 선형성 특징 계산", flush=True)
data = np.load(PROCESSED_DIR / "wm811k_sample.npz")
X_raw, y_encoded, class_names = data["X"], data["y_encoded"], data["class_names"]
y_str = class_names[y_encoded]
n_classes = len(class_names)

aux_all = compute_elongation_batch(X_raw)
print(f"    선형성 계산 완료: {aux_all.shape}", flush=True)

skf = StratifiedKFold(n_splits=N_FOLDS_QUICK, shuffle=True, random_state=RNG_SEED)
folds = list(skf.split(X_raw, y_encoded))

start = time.time()
y_val_all, pred_all = [], []
for fold_i, (train_idx, val_idx) in enumerate(folds):
    print(f"  --- fold {fold_i + 1}/{N_FOLDS_QUICK} 시작 ---", flush=True)
    r = run_one_fold(X_raw, y_encoded, y_str, aux_all, class_names, train_idx, val_idx)
    y_val_all.append(r["y_val_enc"])
    pred_all.append(r["val_pred"])
    fold_f1 = f1_score(r["y_val_enc"], r["val_pred"], average="macro", zero_division=0)
    print(f"    fold {fold_i + 1} 완료: macro-F1 = {fold_f1:.4f} (누적 {time.time()-start:.0f}초)", flush=True)
elapsed = time.time() - start

y_val_all = np.concatenate(y_val_all)
pred_all = np.concatenate(pred_all)
macro_f1 = f1_score(y_val_all, pred_all, average="macro", zero_division=0)
per_class_f1 = f1_score(y_val_all, pred_all, average=None, zero_division=0, labels=range(n_classes))

print(f"\n[결과] 2-Fold 합산 macro-F1 = {macro_f1:.4f} (소요 {elapsed/60:.1f}분)", flush=True)

# %% [markdown]
# ## 결론

# %%
lines = ["=== 보조특징(선형성) 없음(05b 재사용) vs 있음(신규) 비교 ==="]
header = f"{'클래스':10s}{'없음':>12s}{'있음':>12s}{'차이':>12s}"
lines.append(header)
print("\n" + header)
for i, cls in enumerate(class_names):
    f1_no = KNOWN_NOAUX_PER_CLASS.get(cls, float("nan"))
    f1_yes = per_class_f1[i]
    marker = " <- Scratch (핵심 관심사)" if cls == "Scratch" else (" <- Loc" if cls == "Loc" else "")
    row = f"{cls:10s}{f1_no:>12.4f}{f1_yes:>12.4f}{f1_yes - f1_no:>+12.4f}{marker}"
    print(row)
    lines.append(row)

overall = f"{'전체(macro)':10s}{KNOWN_NOAUX_MACRO_F1:>12.4f}{macro_f1:>12.4f}{macro_f1 - KNOWN_NOAUX_MACRO_F1:>+12.4f}"
print("\n" + overall)
lines.append("\n" + overall)
lines.append(f"\n소요 시간: {elapsed/60:.1f}분")

with open(OUTPUT_DIR / "phase4_aux_feature_check.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[완료] 저장됨: {OUTPUT_DIR / 'phase4_aux_feature_check.txt'}")
