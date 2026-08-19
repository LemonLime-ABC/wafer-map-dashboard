# %% [markdown]
# # 05a_input_representation_check — 입력 표현 비교 (PLAN.md 4.1절)
#
# 웨이퍼 맵 픽셀 값 0/1/2를 CNN에 어떻게 넣을지 두 가지를 비교한다.
#
# - **raw**: 0/1/2를 숫자 그대로(0~1로 정규화) 1채널로 넣는다.
#   문제: CNN 입장에서 "0과 2의 차이(2)가 0과 1의 차이(1)보다 크다"는
#   식으로 **크기 관계가 있는 숫자**처럼 취급한다. 하지만 실제로는
#   0(다이없음)/1(정상)/2(불량)이 "얼마나 다른가"가 아니라 그냥 **서로
#   다른 범주**일 뿐이다 — Phase 1에서 리사이즈 방식을 고를 때 겪은
#   문제와 근본적으로 같다.
# - **onehot**: 0/1/2 각각을 독립된 채널로 쪼갠다(3채널). "배경인가/
#   아닌가", "정상인가/아닌가", "불량인가/아닌가"를 각각 따로 알려주는
#   방식이라, 범주 사이에 억지로 크기 관계를 만들지 않는다.
#
# 원리상 onehot이 더 타당해 보이지만, 말로만 판단하지 않고 실제로 작게
# (2-Fold, 5 epoch) 학습시켜 macro-F1로 비교한다.

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
N_FOLDS_QUICK = 2   # 빠른 비교용 — 본 학습(5-Fold)보다 적게
EPOCHS_QUICK = 5
BATCH_SIZE = 64
LR = 1e-3
MINORITY_CLASSES = ["Scratch", "Random", "Donut", "Near-full"]

torch.manual_seed(RNG_SEED)

# %% [markdown]
# ## 1. 데이터 로드

# %%
data = np.load(PROCESSED_DIR / "wm811k_sample.npz")
X_raw, y_encoded, class_names = data["X"], data["y_encoded"], data["class_names"]
y_str = class_names[y_encoded]  # 정수 라벨 -> 문자열 라벨 (augment_batch가 문자열 라벨을 기대함)
print(f"[1] 데이터: X={X_raw.shape}, 클래스 {len(class_names)}개")


def prepare_input(X: np.ndarray, mode: str) -> np.ndarray:
    """(N,H,W) uint8(값 0/1/2)을 CNN 입력 형태 (N,C,H,W) float32로 바꾼다."""
    if mode == "raw":
        return (X.astype(np.float32) / 2.0)[:, None, :, :]  # 0~1로 정규화, 1채널
    if mode == "onehot":
        # 문법 설명: 리스트 컴프리헨션 + np.stack으로 원-핫 채널 만들기
        # (X == c)는 각 칸이 값 c인지 아닌지를 True/False로 채운 배열을
        # 만든다. 이걸 c=0,1,2 세 번 반복해 만든 배열 3개를 새 축(채널
        # 축)으로 쌓으면 (N, 3, H, W) 모양이 된다.
        channels = [(X == c).astype(np.float32) for c in (0, 1, 2)]
        return np.stack(channels, axis=1)
    raise ValueError(f"알 수 없는 mode: {mode}")


def run_one_fold(y_enc: np.ndarray, y_lab: np.ndarray,
                  train_idx: np.ndarray, val_idx: np.ndarray,
                  mode: str, in_channels: int) -> float:
    """한 fold를 학습시키고 검증 macro-F1을 반환한다."""
    # ---- 학습 fold의 소수 클래스만 증강 (검증 fold는 절대 건드리지 않음) ----
    X_train_raw = X_raw[train_idx]
    y_train_lab = y_lab[train_idx]
    minority_mask = np.isin(y_train_lab, MINORITY_CLASSES)

    X_aug_raw, y_aug_lab = augment_batch(X_train_raw[minority_mask], y_train_lab[minority_mask])
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_aug_enc = np.array([label_to_idx[c] for c in y_aug_lab])

    # 다수 클래스(원본, 증강 안 함) + 소수 클래스(증강됨)를 합쳐 최종 학습셋 구성
    X_train_final_raw = np.concatenate([X_train_raw[~minority_mask], X_aug_raw], axis=0)
    y_train_final_enc = np.concatenate([y_enc[train_idx][~minority_mask], y_aug_enc], axis=0)

    X_train = prepare_input(X_train_final_raw, mode)
    X_val = prepare_input(X_raw[val_idx], mode)
    y_val_enc = y_enc[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_final_enc))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = WaferCNN(in_channels=in_channels, n_classes=len(class_names))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS_QUICK):
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    # ---- 검증: 원본(증강 안 된) 검증 fold로만 평가 ----
    model.eval()
    # 문법 설명: torch.no_grad() 컨텍스트 매니저
    # 검증할 땐 역전파(기울기 계산)가 필요 없다. no_grad() 블록 안에서는
    # PyTorch가 기울기 추적을 꺼서 메모리를 아끼고 속도도 빨라진다.
    with torch.no_grad():
        val_logits = model(torch.from_numpy(X_val))
        val_pred = val_logits.argmax(dim=1).numpy()

    return f1_score(y_val_enc, val_pred, average="macro", zero_division=0)


# %% [markdown]
# ## 2. raw vs onehot 비교 (2-Fold x 5 epoch)

# %%
skf = StratifiedKFold(n_splits=N_FOLDS_QUICK, shuffle=True, random_state=RNG_SEED)
folds = list(skf.split(X_raw, y_encoded))

results = {}
for mode_name, in_ch in [("raw", 1), ("onehot", 3)]:
    print(f"\n[2] 입력 표현 = {mode_name} (in_channels={in_ch}) 테스트 중...")
    start = time.time()
    fold_scores = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        score = run_one_fold(y_encoded, y_str, train_idx, val_idx, mode_name, in_ch)
        fold_scores.append(score)
        print(f"    fold {fold_i + 1}: macro-F1 = {score:.4f}")
    elapsed = time.time() - start
    avg_score = float(np.mean(fold_scores))
    results[mode_name] = avg_score
    print(f"    평균 macro-F1 = {avg_score:.4f} (소요 {elapsed:.1f}초)")

# %% [markdown]
# ## 3. 결론

# %%
print("\n[3] === 비교 결과 ===")
for mode_name, score in results.items():
    print(f"    {mode_name:8s}: 평균 macro-F1 = {score:.4f}")

better = max(results, key=results.get)
print(f"\n    -> 이번 실험 기준 더 나은 쪽: {better}")
print("    (5 epoch만 돌린 빠른 비교이므로 차이가 작으면 원리상 타당한 쪽을 우선한다)")

summary_path = OUTPUT_DIR / "phase4_input_representation_check.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"raw macro-F1: {results['raw']:.4f}\n")
    f.write(f"onehot macro-F1: {results['onehot']:.4f}\n")
    f.write(f"결론: {better}\n")
print(f"\n[완료] 저장됨: {summary_path}")
