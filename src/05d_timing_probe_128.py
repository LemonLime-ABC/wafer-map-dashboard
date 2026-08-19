# %% [markdown]
# # 05d_timing_probe_128 — 128x128 실제 epoch 시간 측정
#
# 이전 테스트(05c)가 예상보다 훨씬 오래 걸려서, 이번엔 본 테스트를
# 돌리기 전에 **딱 3epoch만** 먼저 돌려 실제 속도를 재고, 그 수치로
# 전체 예상 시간을 계산한다. 추측이 아니라 실측으로 판단한다.

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

sys.path.append(str(Path(__file__).resolve().parent))
from model import WaferCNN  # noqa: E402
from augment import augment_batch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RNG_SEED = 42
BATCH_SIZE = 64
LR = 1e-3
MINORITY_CLASSES = ["Scratch", "Random", "Donut", "Near-full"]
PROBE_EPOCHS = 3

torch.manual_seed(RNG_SEED)

# %% [markdown]
# ## 1. 128x128 데이터로 fold 1 하나만 준비

# %%
t0 = time.time()
data = np.load(PROCESSED_DIR / "wm811k_sample_128.npz")
X_raw, y_encoded, class_names = data["X"], data["y_encoded"], data["class_names"]
y_str = class_names[y_encoded]
n_classes = len(class_names)
print(f"[1] 데이터 로드: {time.time()-t0:.1f}초, X={X_raw.shape}")

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RNG_SEED)
train_idx, val_idx = next(iter(skf.split(X_raw, y_encoded)))

t0 = time.time()
y_train_lab = y_str[train_idx]
minority_mask = np.isin(y_train_lab, MINORITY_CLASSES)
X_aug_raw, y_aug_lab = augment_batch(X_raw[train_idx][minority_mask], y_train_lab[minority_mask])
print(f"[1] 증강(augment_batch) 소요 시간: {time.time()-t0:.1f}초 "
      f"({minority_mask.sum():,}장 -> {len(y_aug_lab):,}장)")

label_to_idx = {c: i for i, c in enumerate(class_names)}
y_aug_enc = np.array([label_to_idx[c] for c in y_aug_lab])
X_train_final_raw = np.concatenate([X_raw[train_idx][~minority_mask], X_aug_raw], axis=0)
y_train_final_enc = np.concatenate([y_encoded[train_idx][~minority_mask], y_aug_enc], axis=0)
print(f"[1] 최종 학습 fold 크기: {len(y_train_final_enc):,}장")


def prepare_input(X):
    return (X.astype(np.float32) / 2.0)[:, None, :, :]


t0 = time.time()
X_train = prepare_input(X_train_final_raw)
X_val = prepare_input(X_raw[val_idx])
print(f"[1] float32 변환 소요 시간: {time.time()-t0:.1f}초")

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_final_enc))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
n_batches = len(train_loader)
print(f"[1] 배치 수: {n_batches}개 (배치 크기 {BATCH_SIZE})")

# %% [markdown]
# ## 2. 정확히 3epoch만 실행하며 시간 측정

# %%
model = WaferCNN(in_channels=1, n_classes=n_classes, head="flatten")
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

epoch_times = []
model.train()
for epoch in range(PROBE_EPOCHS):
    t0 = time.time()
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    epoch_time = time.time() - t0
    epoch_times.append(epoch_time)
    print(f"[2] epoch {epoch + 1}/{PROBE_EPOCHS}: {epoch_time:.1f}초")

avg_epoch_time = np.mean(epoch_times)

# %% [markdown]
# ## 3. 검증 fold 예측 시간 측정 (fold당 1번만 발생, 무시 가능한 수준인지 확인)

# %%
t0 = time.time()
model.eval()
with torch.no_grad():
    _ = model(torch.from_numpy(X_val))
val_time = time.time() - t0
print(f"[3] 검증 fold({len(val_idx):,}장) 예측 시간: {val_time:.1f}초")

# %% [markdown]
# ## 4. 전체 예상 시간 계산

# %%
print("\n=== 실측 기반 전체 예상 시간 ===")
print(f"1epoch 평균: {avg_epoch_time:.1f}초")

for n_folds, n_epochs in [(1, 15), (2, 15), (1, 10), (2, 10)]:
    total_sec = n_folds * (n_epochs * avg_epoch_time + val_time)
    print(f"  {n_folds}-Fold x {n_epochs}epoch: 약 {total_sec/60:.1f}분")
