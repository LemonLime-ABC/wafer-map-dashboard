# %% [markdown]
# # 04b_preprocessing_128 — 128x128 해상도 데이터 준비
#
# Phase 4 개선 2단계: Scratch 클래스가 64x64 리사이즈에서 선 연속성을
# 잃는 문제(Phase 1에서 발견)를 겨냥해, 같은 샘플을 128x128로 다시
# 리사이즈해서 저장한다. `04_preprocessing.py`와 로직은 완전히 같고
# TARGET_SIZE와 저장 파일명만 다르다 — **같은 RNG_SEED를 쓰므로 64x64
# 버전과 정확히 같은 웨이퍼 샘플**이 뽑힌다(해상도만 다르게 비교하기 위함).

# %%
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_wm811k  # noqa: E402
from resize_utils import resize_nearest  # noqa: E402
from augment import augment_batch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 128  # 04_preprocessing.py는 64 -> 여기서는 128
RNG_SEED = 42       # 04_preprocessing.py와 동일 -> 같은 샘플이 뽑힘
MAJORITY_CAP = 2000
MAJORITY_CLASSES = ["none", "Edge-Ring", "Edge-Loc", "Center", "Loc"]
MINORITY_CLASSES = ["Scratch", "Random", "Donut", "Near-full"]
N_FOLDS = 5

report_lines = []


def log(msg: str):
    print(msg)
    report_lines.append(msg)


# %% [markdown]
# ## 1. 데이터 로드 및 샘플링 (04_preprocessing.py와 동일 로직)

# %%
df = load_wm811k()
labeled = df[df["failureType_clean"].notna()].reset_index(drop=True)

log(f"[1] 라벨된 전체: {len(labeled):,}장")

sampled_parts = []
for cls in MAJORITY_CLASSES:
    subset = labeled[labeled["failureType_clean"] == cls]
    n_take = min(MAJORITY_CAP, len(subset))
    sampled_parts.append(subset.sample(n_take, random_state=RNG_SEED))
    log(f"    {cls:10s}: {len(subset):>7,}장 중 {n_take:,}장 샘플링 (다수 클래스, 축소)")

for cls in MINORITY_CLASSES:
    subset = labeled[labeled["failureType_clean"] == cls]
    sampled_parts.append(subset)
    log(f"    {cls:10s}: {len(subset):>7,}장 전량 사용 (소수 클래스)")

sample_df = pd.concat(sampled_parts, ignore_index=True)
log(f"    합계: {len(sample_df):,}장")

# %% [markdown]
# ## 2. 128x128 리사이즈 후 저장

# %%
log(f"\n[2] {TARGET_SIZE}x{TARGET_SIZE} 리사이즈 중...")
X = np.stack([resize_nearest(m, TARGET_SIZE) for m in sample_df["waferMap"]]).astype(np.uint8)
y = sample_df["failureType_clean"].to_numpy()

class_names = sorted(np.unique(y))
label_to_idx = {c: i for i, c in enumerate(class_names)}
y_encoded = np.array([label_to_idx[c] for c in y], dtype=np.int64)

log(f"    X: {X.shape} {X.dtype} / y: {y.shape}")

npz_path = PROCESSED_DIR / f"wm811k_sample_{TARGET_SIZE}.npz"
np.savez_compressed(npz_path, X=X, y_encoded=y_encoded, class_names=np.array(class_names))
log(f"    저장됨: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## 3. 증강 누수 차단 구조 재검증 (해상도가 바뀌어도 구조는 그대로 성립해야 함)

# %%
log(f"\n[3] 5-Fold 분할 + 증강 누수 차단 구조 검증 ({TARGET_SIZE}x{TARGET_SIZE})")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    y_train = y[train_idx]
    train_minority_mask = np.isin(y_train, MINORITY_CLASSES)
    X_aug, y_aug = augment_batch(X[train_idx][train_minority_mask], y_train[train_minority_mask])
    assert len(y_aug) == train_minority_mask.sum() * 8, "증강 배수가 8배가 아닙니다"
    log(f"    fold {fold_i + 1}: 학습 {len(train_idx):,} / 검증 {len(val_idx):,} / "
        f"소수클래스 증강 {train_minority_mask.sum():,}->{len(y_aug):,}")

log(f"\n[3] 검증 통과 ({TARGET_SIZE}x{TARGET_SIZE}에서도 누수 차단 구조 정상)")

summary_path = OUTPUT_DIR / f"phase3_preprocessing_{TARGET_SIZE}_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\n[완료] 요약: {summary_path}")
print(f"[완료] 데이터: {npz_path}")
