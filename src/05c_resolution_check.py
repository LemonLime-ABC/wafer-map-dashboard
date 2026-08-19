# %% [markdown]
# # 05c_resolution_check — 64x64 vs 128x128 해상도 비교 (Flatten head 고정)
#
# Phase 4 개선 2단계 검증. Flatten head는 이미 채택 확정(05b 결과).
# 여기서는 해상도만 바꿔서 **Scratch가 실제로 좋아지는지**를 중점 확인한다.

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

torch.manual_seed(RNG_SEED)

DATASETS = {
    64: PROCESSED_DIR / "wm811k_sample.npz",
    128: PROCESSED_DIR / "wm811k_sample_128.npz",
}


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
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(torch.from_numpy(X_val)).argmax(dim=1).numpy()

    return {"y_val_enc": y_val_enc, "val_pred": val_pred}


# %% [markdown]
# ## 64 vs 128 비교

# %%
all_results = {}
for res, npz_path in DATASETS.items():
    print(f"\n[해상도={res}] 데이터 로드: {npz_path.name}")
    data = np.load(npz_path)
    X_raw, y_encoded, class_names = data["X"], data["y_encoded"], data["class_names"]
    y_str = class_names[y_encoded]
    n_classes = len(class_names)

    skf = StratifiedKFold(n_splits=N_FOLDS_QUICK, shuffle=True, random_state=RNG_SEED)
    folds = list(skf.split(X_raw, y_encoded))

    start = time.time()
    y_val_all, pred_all = [], []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        r = run_one_fold(X_raw, y_encoded, y_str, class_names, train_idx, val_idx)
        y_val_all.append(r["y_val_enc"])
        pred_all.append(r["val_pred"])
        fold_f1 = f1_score(r["y_val_enc"], r["val_pred"], average="macro", zero_division=0)
        print(f"    fold {fold_i + 1}: macro-F1 = {fold_f1:.4f}")
    elapsed = time.time() - start

    y_val_all = np.concatenate(y_val_all)
    pred_all = np.concatenate(pred_all)
    macro_f1 = f1_score(y_val_all, pred_all, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_val_all, pred_all, average=None, zero_division=0, labels=range(n_classes))

    all_results[res] = {"macro_f1": macro_f1, "per_class_f1": per_class_f1,
                         "elapsed": elapsed, "class_names": class_names}
    print(f"    2-Fold 합산 macro-F1 = {macro_f1:.4f} (소요 {elapsed / 60:.1f}분)")

# %% [markdown]
# ## 결론

# %%
print("\n=== 64 vs 128 해상도 비교 결과 (Flatten head) ===")
lines = ["=== 64x64 vs 128x128 비교 (Flatten head, 2-Fold x 15epoch) ==="]

class_names = all_results[64]["class_names"]
header = f"{'클래스':10s}{'64x64':>12s}{'128x128':>12s}{'차이':>12s}"
print(header)
lines.append(header)
for i, cls in enumerate(class_names):
    f1_64 = all_results[64]["per_class_f1"][i]
    f1_128 = all_results[128]["per_class_f1"][i]
    marker = " <- Scratch (핵심 관심사)" if cls == "Scratch" else ""
    row = f"{cls:10s}{f1_64:>12.4f}{f1_128:>12.4f}{f1_128 - f1_64:>+12.4f}{marker}"
    print(row)
    lines.append(row)

overall = (f"{'전체(macro)':10s}{all_results[64]['macro_f1']:>12.4f}"
           f"{all_results[128]['macro_f1']:>12.4f}"
           f"{all_results[128]['macro_f1'] - all_results[64]['macro_f1']:>+12.4f}")
print("\n" + overall)
lines.append("\n" + overall)

time_line = f"\n소요 시간: 64x64={all_results[64]['elapsed']/60:.1f}분, 128x128={all_results[128]['elapsed']/60:.1f}분"
print(time_line)
lines.append(time_line)

with open(OUTPUT_DIR / "phase4_resolution_check.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[완료] 저장됨: {OUTPUT_DIR / 'phase4_resolution_check.txt'}")
