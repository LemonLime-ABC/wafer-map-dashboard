# %% [markdown]
# # 05b_head_architecture_check — GAP vs Flatten head 비교
#
# Phase 4 1차 결과에서 GAP이 위치·모양 정보를 지워 Loc/Scratch/Center/Donut
# 성능이 낮게 나왔다는 가설을 세웠다. 이걸 본 학습(5-Fold, 40epoch) 전에
# 작게(2-Fold, 15epoch) 먼저 검증한다.
#
# **확인 포인트**: 전체 macro-F1뿐 아니라, 가설이 지목한 클래스
# (Center/Loc/Scratch/Donut)의 F1이 실제로 오르는지를 따로 본다.

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
from sklearn.metrics import f1_score, classification_report

sys.path.append(str(Path(__file__).resolve().parent))
from model import WaferCNN  # noqa: E402
from augment import augment_batch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RNG_SEED = 42
N_FOLDS_QUICK = 2
EPOCHS_QUICK = 15   # raw/onehot 비교(5epoch)보다 넉넉하게 — flatten은 파라미터가 많아 수렴에 시간이 더 걸릴 수 있음
BATCH_SIZE = 64
LR = 1e-3
MINORITY_CLASSES = ["Scratch", "Random", "Donut", "Near-full"]
FOCUS_CLASSES = ["Center", "Loc", "Scratch", "Donut"]  # 가설이 지목한 클래스들

torch.manual_seed(RNG_SEED)

# %% [markdown]
# ## 1. 데이터 로드

# %%
data = np.load(PROCESSED_DIR / "wm811k_sample.npz")
X_raw, y_encoded, class_names = data["X"], data["y_encoded"], data["class_names"]
y_str = class_names[y_encoded]
n_classes = len(class_names)
print(f"[1] 데이터: X={X_raw.shape}, 클래스 {n_classes}개")


def prepare_input(X: np.ndarray) -> np.ndarray:
    return (X.astype(np.float32) / 2.0)[:, None, :, :]  # raw 방식(Phase4 1차에서 채택)


def run_one_fold(train_idx: np.ndarray, val_idx: np.ndarray, head: str) -> dict:
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

    model = WaferCNN(in_channels=1, n_classes=n_classes, head=head)
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
# ## 2. gap vs flatten 비교 (2-Fold x 15 epoch)

# %%
skf = StratifiedKFold(n_splits=N_FOLDS_QUICK, shuffle=True, random_state=RNG_SEED)
folds = list(skf.split(X_raw, y_encoded))

all_results = {}
for head in ["gap", "flatten"]:
    print(f"\n[2] head = {head} 테스트 중...")
    start = time.time()
    y_val_all, pred_all = [], []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        r = run_one_fold(train_idx, val_idx, head)
        y_val_all.append(r["y_val_enc"])
        pred_all.append(r["val_pred"])
        fold_f1 = f1_score(r["y_val_enc"], r["val_pred"], average="macro", zero_division=0)
        print(f"    fold {fold_i + 1}: macro-F1 = {fold_f1:.4f}")
    elapsed = time.time() - start

    y_val_all = np.concatenate(y_val_all)
    pred_all = np.concatenate(pred_all)
    macro_f1 = f1_score(y_val_all, pred_all, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_val_all, pred_all, average=None, zero_division=0, labels=range(n_classes))

    all_results[head] = {"macro_f1": macro_f1, "per_class_f1": per_class_f1, "elapsed": elapsed}
    print(f"    2-Fold 합산 macro-F1 = {macro_f1:.4f} (소요 {elapsed / 60:.1f}분)")

# %% [markdown]
# ## 3. 결론 — 특히 가설이 지목한 클래스들을 확인

# %%
print("\n[3] === 비교 결과 ===")
lines = ["=== GAP vs Flatten head 비교 (2-Fold x 15epoch, raw 입력) ==="]

header = f"{'클래스':10s}" + "".join(f"{h:>12s}" for h in all_results) + f"{'차이(flatten-gap)':>18s}"
print(header)
lines.append(header)
for i, cls in enumerate(class_names):
    gap_f1 = all_results["gap"]["per_class_f1"][i]
    flat_f1 = all_results["flatten"]["per_class_f1"][i]
    marker = " <- 가설이 지목한 클래스" if cls in FOCUS_CLASSES else ""
    row = f"{cls:10s}" + f"{gap_f1:>12.4f}" + f"{flat_f1:>12.4f}" + f"{flat_f1 - gap_f1:>+18.4f}{marker}"
    print(row)
    lines.append(row)

overall_row = (f"{'전체(macro)':10s}" + f"{all_results['gap']['macro_f1']:>12.4f}"
               + f"{all_results['flatten']['macro_f1']:>12.4f}"
               + f"{all_results['flatten']['macro_f1'] - all_results['gap']['macro_f1']:>+18.4f}")
print("\n" + overall_row)
lines.append("\n" + overall_row)

focus_gap = np.mean([all_results["gap"]["per_class_f1"][list(class_names).index(c)] for c in FOCUS_CLASSES])
focus_flat = np.mean([all_results["flatten"]["per_class_f1"][list(class_names).index(c)] for c in FOCUS_CLASSES])
focus_line = f"\n가설 지목 클래스(Center/Loc/Scratch/Donut) 평균: gap={focus_gap:.4f} -> flatten={focus_flat:.4f} ({focus_flat - focus_gap:+.4f})"
print(focus_line)
lines.append(focus_line)

better = "flatten" if all_results["flatten"]["macro_f1"] > all_results["gap"]["macro_f1"] else "gap"
conclusion = f"\n-> 전체 macro-F1 기준 더 나은 head: {better}"
print(conclusion)
lines.append(conclusion)

summary_path = OUTPUT_DIR / "phase4_head_architecture_check.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n[완료] 저장됨: {summary_path}")
