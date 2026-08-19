# %% [markdown]
# # 04_preprocessing — Phase 3 데이터 준비
#
# (PLAN.md엔 `03_preprocessing.py`로 적혀 있지만, `03`은 이미 시각화
# 스크립트가 쓰고 있어서 번호를 `04`로 옮겼다.)
#
# 이 스크립트가 하는 일 3가지:
# 1. **샘플링** — 다수 클래스는 2,000장으로 줄이고, 소수 클래스는 전량 사용
# 2. **저장** — 리사이즈까지 끝낸 학습용 데이터셋을 파일로 저장 (Phase 4가 재사용)
# 3. **증강 누수 차단 구조 검증** — 5-Fold로 나눈 뒤, 증강이 학습 fold
#    안에서만 적용되고 검증 fold는 원본 그대로 남는지 실제로 확인
#
# **주의**: 이 스크립트는 CNN을 학습시키지 않는다. 모델 학습은 Phase 4다.
# 여기서는 "학습에 쓸 데이터를 올바른 구조로 준비하는 것"까지만 한다.

# %%
import sys
from pathlib import Path

# Windows 콘솔 코드페이지(cp949)는 —(줄대시) 같은 일부 유니코드 문자를
# 표현하지 못해 print()가 그대로 에러를 낸다. 출력 인코딩을 UTF-8로
# 강제로 바꿔서(표현 못 하는 문자는 물음표로 대체) 죽지 않게 한다.
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

TARGET_SIZE = 64
RNG_SEED = 42
MAJORITY_CAP = 2000
# PLAN.md 3.1절 기준 분류
MAJORITY_CLASSES = ["none", "Edge-Ring", "Edge-Loc", "Center", "Loc"]
MINORITY_CLASSES = ["Scratch", "Random", "Donut", "Near-full"]  # 전량 사용 + 증강 대상
N_FOLDS = 5

report_lines = []


def log(msg: str):
    print(msg)
    report_lines.append(msg)


# %% [markdown]
# ## 1. 데이터 로드 및 샘플링

# %%
df = load_wm811k()
labeled = df[df["failureType_clean"].notna()].reset_index(drop=True)

log(f"[1] 라벨된 전체: {len(labeled):,}장")

rng = np.random.default_rng(RNG_SEED)
sampled_parts = []

for cls in MAJORITY_CLASSES:
    subset = labeled[labeled["failureType_clean"] == cls]
    n_take = min(MAJORITY_CAP, len(subset))
    # -----------------------------------------------------------
    # 문법 설명: DataFrame.sample(random_state=...)
    # -----------------------------------------------------------
    # sample(n, random_state=시드)는 n개를 무작위로 뽑되, 같은 시드를
    # 쓰면 항상 같은 결과가 나온다(재현 가능성). random_state는
    # np.random.default_rng와 별개의 pandas 자체 난수 체계라서,
    # 정수 시드를 그대로 넘기면 된다.
    # -----------------------------------------------------------
    sampled_parts.append(subset.sample(n_take, random_state=RNG_SEED))
    log(f"    {cls:10s}: {len(subset):>7,}장 중 {n_take:,}장 샘플링 (다수 클래스, 축소)")

for cls in MINORITY_CLASSES:
    subset = labeled[labeled["failureType_clean"] == cls]
    sampled_parts.append(subset)
    log(f"    {cls:10s}: {len(subset):>7,}장 전량 사용 (소수 클래스)")

sample_df = pd.concat(sampled_parts, ignore_index=True)
log(f"    합계: {len(sample_df):,}장 (PLAN.md 예상치: 약 12,763장)")

# %% [markdown]
# ## 2. 리사이즈 후 배열로 저장

# %%
log("\n[2] 64x64 리사이즈 중...")
X = np.stack([resize_nearest(m, TARGET_SIZE) for m in sample_df["waferMap"]]).astype(np.uint8)
y = sample_df["failureType_clean"].to_numpy()

class_names = sorted(np.unique(y))
label_to_idx = {c: i for i, c in enumerate(class_names)}
y_encoded = np.array([label_to_idx[c] for c in y], dtype=np.int64)

log(f"    X: {X.shape} {X.dtype} / y: {y.shape}")
log(f"    클래스 순서(정수 인코딩): {class_names}")

npz_path = PROCESSED_DIR / "wm811k_sample.npz"
np.savez_compressed(
    npz_path,
    X=X,
    y_encoded=y_encoded,
    class_names=np.array(class_names),
)
log(f"    저장됨: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
log("    (X=리사이즈된 이미지, y_encoded=정수 라벨, class_names=인덱스→이름 매핑 — Phase 4가 그대로 불러 씀)")

# %% [markdown]
# ## 3. 증강 누수 차단 구조 검증 — 5-Fold로 직접 확인
#
# CLAUDE.md 6.1절의 원칙을 코드로 실제 검증한다:
# **학습 fold만 증강하고, 검증 fold는 원본 그대로 남아야 한다.**
# 여기서는 모델을 학습시키지 않고, "폴드를 나눈 뒤 증강을 적용하면
# 실제로 무슨 일이 일어나는지"만 숫자로 확인한다.

# %%
log("\n[3] 5-Fold 분할 + 증강 누수 차단 구조 검증")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)

for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    y_train = y[train_idx]
    y_val = y[val_idx]

    # ---- 검증 fold는 절대 건드리지 않는다 (원본 그대로) ----
    val_size_before = len(val_idx)

    # ---- 학습 fold 중 "소수 클래스"만 골라 증강한다 ----
    train_minority_mask = np.isin(y_train, MINORITY_CLASSES)
    X_train_minor = X[train_idx][train_minority_mask]
    y_train_minor = y_train[train_minority_mask]

    X_aug, y_aug = augment_batch(X_train_minor, y_train_minor)

    log(f"\n  --- Fold {fold_i + 1}/{N_FOLDS} ---")
    log(f"    학습 fold 크기: {len(train_idx):,} / 검증 fold 크기: {len(val_idx):,}")
    log(f"    학습 fold 내 소수 클래스 원본: {len(y_train_minor):,}장 "
        f"-> 증강 후: {len(y_aug):,}장 (기대값: {len(y_train_minor) * 8:,}장)")
    assert len(y_aug) == len(y_train_minor) * 8, "증강 배수가 8배가 아닙니다"

    # ---- 검증: 검증 fold 크기가 증강 전후로 전혀 변하지 않았는가 ----
    assert len(val_idx) == val_size_before, "검증 fold가 오염되었습니다"

    # ---- 참고: Near-full처럼 극소수인 클래스는 fold당 몇 장인지 ----
    near_full_val = int(np.sum(y_val == "Near-full"))
    log(f"    이 fold의 Near-full 검증 샘플 수: {near_full_val}장 "
        f"(CLAUDE.md 4.2절에서 예고한 '약 30장, 3장 차이로 Recall 10% 요동' 상황)")

log("\n[3] 검증 통과: 모든 fold에서 검증셋은 원본 그대로, 학습 fold의 소수 클래스만 8배 증강됨")

# %% [markdown]
# ## 4. 요약 저장

# %%
summary_path = OUTPUT_DIR / "phase3_preprocessing_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n[완료] 요약 저장됨: {summary_path}")
print(f"[완료] 전처리된 데이터: {npz_path}")
