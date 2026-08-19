# %% [markdown]
# # 05e_resolution_check_128only — 128x128만 재검증
#
# 64x64+Flatten 결과는 05b에서 이미 확인했으므로(macro-F1=0.8270,
# outputs/phase4_head_architecture_check.txt) 다시 안 돌리고 재사용한다.
# 여기서는 128x128만 2-Fold x 15epoch로 돌려서 64x64와 비교한다.

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RNG_SEED = 42
N_FOLDS_QUICK = 2
EPOCHS_QUICK = 15
BATCH_SIZE = 64
LR = 1e-3
MINORITY_CLASSES = ["Scratch", "Random", "Donut", "Near-full"]

# 05b에서 이미 확인한 64x64+flatten 결과 (재사용, 다시 안 돌림)
KNOWN_64_MACRO_F1 = 0.8270
KNOWN_64_PER_CLASS = {
    "Center": 0.9312, "Donut": 0.9018, "Edge-Loc": 0.8214, "Edge-Ring": 0.9657,
    "Loc": 0.7047, "Near-full": 0.8490, "Random": 0.9045, "Scratch": 0.5261, "none": 0.8386,
}

torch.manual_seed(RNG_SEED)


def prepare_input(X: np.ndarray) -> np.ndarray:
    return (X.astype(np.float32) / 2.0)[:, None, :, :]


def run_one_fold(X_raw, y_encoded, y_str, class_names, train_idx, val_idx) -> dict:
    n_classes = len(class_names)
    y_train_lab = y_str[train_idx]
    minority_mask = np.isin(y_train_lab, MINORITY_CLASSES)

    X_aug_raw, y_aug_lab = augment_batch(X_raw[train_idx][minority_mask], y_train_lab[minority_mask])
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_aug_enc = np.array([label_to_idx[c] for c in y_aug_lab])

    X_train_final_raw = np.concatenate([X_raw[train_idx][~minority_mask], X_aug_raw], axis=0)
    y_train_final_enc = np.concatenate([y_encoded[train_idx][~minority_mask], y_aug_enc], axis=0)

    class_counts_final = np.bincount(y_train_final_enc, minlength=n_classes)
    class_weights = 1.0 / np.maximum(class_counts_final, 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    X_train = prepare_input(X_train_final_raw)
    X_val = prepare_input(X_raw[val_idx])
    y_val_enc = y_encoded[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_final_enc))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = WaferCNN(in_channels=1, n_classes=n_classes, head="flatten")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    model.train()
    for epoch in range(EPOCHS_QUICK):
        ep_start = time.time()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        print(f"      epoch {epoch + 1}/{EPOCHS_QUICK} 완료 ({time.time()-ep_start:.1f}초)", flush=True)

    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val)).argmax(dim=1).numpy()

    return {"y_val_enc": y_val_enc, "val_pred": val_pred}


# %% [markdown]
# ## 128x128 실행

# %%
print("[128] 데이터 로드", flush=True)
data = np.load(PROCESSED_DIR / "wm811k_sample_128.npz")
X_raw, y_encoded, class_names = data["X"], data["y_encoded"], data["class_names"]
y_str = class_names[y_encoded]
n_classes = len(class_names)

skf = StratifiedKFold(n_splits=N_FOLDS_QUICK, shuffle=True, random_state=RNG_SEED)
folds = list(skf.split(X_raw, y_encoded))

start = time.time()
y_val_all, pred_all = [], []
for fold_i, (train_idx, val_idx) in enumerate(folds):
    print(f"  --- fold {fold_i + 1}/{N_FOLDS_QUICK} 시작 ---", flush=True)
    r = run_one_fold(X_raw, y_encoded, y_str, class_names, train_idx, val_idx)
    y_val_all.append(r["y_val_enc"])
    pred_all.append(r["val_pred"])
    fold_f1 = f1_score(r["y_val_enc"], r["val_pred"], average="macro", zero_division=0)
    print(f"    fold {fold_i + 1} 완료: macro-F1 = {fold_f1:.4f} (누적 {time.time()-start:.0f}초)", flush=True)
elapsed = time.time() - start

y_val_all = np.concatenate(y_val_all)
pred_all = np.concatenate(pred_all)
macro_f1 = f1_score(y_val_all, pred_all, average="macro", zero_division=0)
per_class_f1 = f1_score(y_val_all, pred_all, average=None, zero_division=0, labels=range(n_classes))

print(f"\n[128] 2-Fold 합산 macro-F1 = {macro_f1:.4f} (소요 {elapsed/60:.1f}분)", flush=True)

# %% [markdown]
# ## 결론 (64x64는 05b 재사용)

# %%
lines = ["=== 64x64(05b 재사용) vs 128x128(신규) 비교, Flatten head ==="]
header = f"{'클래스':10s}{'64x64':>12s}{'128x128':>12s}{'차이':>12s}"
lines.append(header)
print("\n" + header)
for i, cls in enumerate(class_names):
    f1_64 = KNOWN_64_PER_CLASS.get(cls, float("nan"))
    f1_128 = per_class_f1[i]
    marker = " <- Scratch (핵심 관심사)" if cls == "Scratch" else ""
    row = f"{cls:10s}{f1_64:>12.4f}{f1_128:>12.4f}{f1_128 - f1_64:>+12.4f}{marker}"
    print(row)
    lines.append(row)

overall = f"{'전체(macro)':10s}{KNOWN_64_MACRO_F1:>12.4f}{macro_f1:>12.4f}{macro_f1 - KNOWN_64_MACRO_F1:>+12.4f}"
print("\n" + overall)
lines.append("\n" + overall)
lines.append(f"\n128x128 소요 시간: {elapsed/60:.1f}분")

with open(OUTPUT_DIR / "phase4_resolution_check.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[완료] 저장됨: {OUTPUT_DIR / 'phase4_resolution_check.txt'}")
