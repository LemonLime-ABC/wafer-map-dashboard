# %% [markdown]
# # 05_train_evaluate — Phase 4 CNN 학습 및 5-Fold 평가 (v2: Flatten head)
#
# 1차 실행(GAP head)은 macro-F1 0.7691로 PLAN.md 목표 미달이었다. 원인
# 분석(GAP이 위치·모양 정보를 지움) 후 Flatten head로 빠르게 검증(05b)해
# macro-F1 +0.09를 확인했고, 128x128 해상도는 오히려 역효과(05e)라 폐기했다.
# 이번이 그 확정된 구조(**Flatten head + 64x64**)로 도는 **공식 5-Fold 본 학습**이다.
#
# 1차(GAP) 결과 파일을 덮어쓰지 않도록, 이번 산출물은 전부 `_v2_flatten`
# 접미사를 붙여 별도로 저장한다 — 두 결과를 나중에 나란히 비교할 수 있게.
#
# 이 스크립트가 하는 일:
# 1. 5-Fold Stratified 교차검증으로 CNN을 학습 (학습 fold만 8배 증강, 검증 fold는 원본)
# 2. fold마다 macro-F1 기준 조기 종료(early stopping)
# 3. 5개 fold의 검증 예측을 모두 모아(OOF) **교차검증 예측 기준**으로 최종 성능 산출
#    (CLAUDE.md 2.2절 원칙 ③ — 학습 데이터 재예측이 아니라 OOF로 보고)
# 4. macro-F1/balanced accuracy를 주 지표로, 정확도는 참고용으로만 같이 표시
# 5. 9x9 혼동 행렬 저장 (Edge-Loc/Scratch, Center/Loc 칸 특히 확인)
# 6. fold별 모델 가중치를 artifacts/에 저장 (Phase 5 Grad-CAM이 재사용)

# %%
import sys
import time
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, classification_report,
)

sys.path.append(str(Path(__file__).resolve().parent))
from model import WaferCNN  # noqa: E402
from augment import augment_batch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

RNG_SEED = 42
N_FOLDS = 5
MAX_EPOCHS = 40
PATIENCE = 6          # 이만큼 연속으로 val macro-F1이 안 좋아지면 조기 종료
BATCH_SIZE = 64
LR = 1e-3
MINORITY_CLASSES = ["Scratch", "Random", "Donut", "Near-full"]

# 05a에서 확인한 결과에 따라 설정: raw macro-F1=0.579 vs onehot macro-F1=0.380
# (2-Fold x 5epoch 빠른 비교). onehot이 원리상 더 타당해 보였지만 실측 격차가
# 커서 raw를 채택한다 — onehot은 입력 채널이 3배라 짧은 epoch에서 수렴이
# 느렸을 가능성이 있다(근본적으로 나쁘다는 뜻은 아님). 자세한 기록은
# outputs/phase4_input_representation_check.txt 참고.
INPUT_MODE = "raw"
IN_CHANNELS = {"raw": 1, "onehot": 3}[INPUT_MODE]

# 05b에서 확인: GAP macro-F1=0.7356 vs Flatten macro-F1=0.8270 (2-Fold x 15epoch).
# Flatten 채택 확정. 128x128 해상도는 05e에서 역효과 확인(-0.052)돼 폐기,
# 64x64 그대로 사용한다.
HEAD = "flatten"
RUN_TAG = "v2_flatten"  # 산출물 파일명 접미사 — 1차(GAP) 결과와 구분

torch.manual_seed(RNG_SEED)

# %% [markdown]
# ## 0. 데이터 로드

# %%
data = np.load(PROCESSED_DIR / "wm811k_sample.npz")
X_raw, y_encoded, class_names = data["X"], data["y_encoded"], data["class_names"]
y_str = class_names[y_encoded]
n_classes = len(class_names)
print(f"[0] 데이터: X={X_raw.shape}, 클래스 {n_classes}개, 입력 표현={INPUT_MODE}")


def prepare_input(X: np.ndarray) -> np.ndarray:
    """(N,H,W) uint8(값 0/1/2)을 CNN 입력 (N,C,H,W) float32로 바꾼다."""
    if INPUT_MODE == "raw":
        return (X.astype(np.float32) / 2.0)[:, None, :, :]
    channels = [(X == c).astype(np.float32) for c in (0, 1, 2)]
    return np.stack(channels, axis=1)


# %% [markdown]
# ## 1. Fold별 학습 함수
#
# **증강 누수 차단**: `train_idx`로 뽑은 학습 fold 안에서만 `augment_batch()`를
# 호출한다. `val_idx`로 뽑은 검증 fold는 이 함수 안 어디에서도 증강되지
# 않는다 — Phase 3(`04_preprocessing.py`)에서 이미 이 구조 자체를
# `assert`로 검증해뒀고, 여기서는 그 위에 실제 학습 루프만 얹는다.

# %%
def train_one_fold(fold_i: int, train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
    y_train_lab = y_str[train_idx]
    minority_mask = np.isin(y_train_lab, MINORITY_CLASSES)

    X_aug_raw, y_aug_lab = augment_batch(X_raw[train_idx][minority_mask], y_train_lab[minority_mask])
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_aug_enc = np.array([label_to_idx[c] for c in y_aug_lab])

    X_train_final_raw = np.concatenate([X_raw[train_idx][~minority_mask], X_aug_raw], axis=0)
    y_train_final_enc = np.concatenate([y_encoded[train_idx][~minority_mask], y_aug_enc], axis=0)

    X_train = prepare_input(X_train_final_raw)
    X_val = prepare_input(X_raw[val_idx])
    y_val_enc = y_encoded[val_idx]

    # ---- 클래스 가중치: 학습 fold가 "실제로 보는" 최종 분포 기준으로 계산 ----
    # (다수 클래스는 원본 그대로, 소수 클래스는 8배 증강된 뒤이므로, 그
    #  최종 분포에 맞춰 가중치를 매겨야 손실 함수가 보정하려는 대상과 일치한다)
    class_counts_final = np.bincount(y_train_final_enc, minlength=n_classes)
    class_weights = (1.0 / np.maximum(class_counts_final, 1))
    class_weights = class_weights / class_weights.sum() * n_classes  # 평균이 1이 되도록 정규화
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_final_enc))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = WaferCNN(in_channels=IN_CHANNELS, n_classes=n_classes, head=HEAD)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    X_val_t = torch.from_numpy(X_val)

    best_f1 = -1.0
    best_state = None
    epochs_since_improve = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        ep_start = time.time()
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        # ---- 검증 fold로 조기 종료 판단 (원본 그대로, 증강 없음) ----
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).argmax(dim=1).numpy()
        val_f1 = f1_score(y_val_enc, val_pred, average="macro", zero_division=0)
        history.append(val_f1)
        print(f"      fold {fold_i + 1} epoch {epoch + 1}: val macro-F1={val_f1:.4f} "
              f"({time.time() - ep_start:.1f}초)", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            # 문법 설명: model.state_dict()
            # 모델의 모든 가중치를 이름:텐서 딕셔너리로 꺼낸 것. 나중에
            # model.load_state_dict(...)로 그대로 복원할 수 있다. "지금까지
            # 가장 좋았던 시점의 가중치"를 따로 저장해뒀다가, 학습이 끝나면
            # 마지막이 아니라 이 시점으로 되돌리기 위해 쓴다.
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= PATIENCE:
            print(f"    fold {fold_i + 1} epoch {epoch + 1}: {PATIENCE}epoch 연속 개선 없음 -> 조기 종료")
            break

    # 가장 좋았던 시점의 가중치로 복원해서 최종 예측 생성
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_val_pred = model(X_val_t).argmax(dim=1).numpy()

    torch.save(best_state, ARTIFACTS_DIR / f"model_{RUN_TAG}_fold{fold_i}.pt")

    return {
        "fold": fold_i,
        "val_idx": val_idx,
        "val_pred": final_val_pred,
        "best_val_f1": best_f1,
        "n_epochs_trained": len(history),
    }


# %% [markdown]
# ## 2. 5-Fold 전체 실행

# %%
print(f"\n[2] {N_FOLDS}-Fold 학습 시작 (입력={INPUT_MODE}, head={HEAD}, "
      f"최대 {MAX_EPOCHS}epoch, patience={PATIENCE})", flush=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)

# OOF(out-of-fold) 예측을 담을 배열 — 전체 12,763장 각각이 "검증 fold였을 때"의
# 예측값으로 정확히 한 번씩 채워진다. 이러면 학습에 안 쓰인 데이터로만
# 평가한 전체 성능을 낼 수 있다 (CLAUDE.md 2.2절 원칙 ③).
oof_pred = np.full(len(y_encoded), -1, dtype=np.int64)
fold_results = []

start_all = time.time()
for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_encoded)):
    print(f"  --- fold {fold_i + 1}/{N_FOLDS} 시작 ---", flush=True)
    t0 = time.time()
    result = train_one_fold(fold_i, train_idx, val_idx)
    oof_pred[val_idx] = result["val_pred"]
    fold_results.append(result)
    print(f"  fold {fold_i + 1}/{N_FOLDS} 완료: "
          f"{result['n_epochs_trained']}epoch, best val macro-F1={result['best_val_f1']:.4f}, "
          f"{time.time() - t0:.1f}초", flush=True)

total_elapsed = time.time() - start_all
print(f"\n[2] 전체 5-Fold 학습 완료: {total_elapsed / 60:.1f}분")

assert (oof_pred != -1).all(), "OOF 예측이 채워지지 않은 샘플이 있습니다"

# %% [markdown]
# ## 3. 최종 성능 (OOF 기준)

# %%
report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


log("\n[3] === 최종 성능 (교차검증 OOF 예측 기준) ===")

oof_macro_f1 = f1_score(y_encoded, oof_pred, average="macro", zero_division=0)
oof_balanced_acc = balanced_accuracy_score(y_encoded, oof_pred)
oof_accuracy = accuracy_score(y_encoded, oof_pred)

log(f"macro-F1        : {oof_macro_f1:.4f}  <- 주 지표")
log(f"balanced accuracy: {oof_balanced_acc:.4f}  <- 주 지표")
log(f"accuracy         : {oof_accuracy:.4f}  <- 참고용. None이 없는 균형 샘플이라 SECOM만큼 함정은 아니지만 "
    f"클래스별 편차를 가릴 수 있으니 macro-F1을 우선한다")

log("\n--- 클래스별 정밀도/재현율/F1 ---")
log(classification_report(y_encoded, oof_pred, target_names=class_names, digits=3, zero_division=0))

log("\n--- fold별 최고 val macro-F1 (안정성 확인용) ---")
for r in fold_results:
    log(f"  fold {r['fold'] + 1}: {r['best_val_f1']:.4f} ({r['n_epochs_trained']}epoch)")

# %% [markdown]
# ## 4. 혼동 행렬 (Edge-Loc / Scratch 칸 특히 확인)

# %%
cm = confusion_matrix(y_encoded, oof_pred, labels=range(n_classes))
cm_df_lines = ["\n--- 9x9 혼동 행렬 (행=실제, 열=예측) ---"]
header = "실제\\예측".ljust(10) + "".join(f"{c[:8]:>9s}" for c in class_names)
cm_df_lines.append(header)
for i, cls in enumerate(class_names):
    row = cls.ljust(10) + "".join(f"{cm[i, j]:>9d}" for j in range(n_classes))
    cm_df_lines.append(row)
for line in cm_df_lines:
    log(line)

# Edge-Loc <-> Scratch 혼동을 별도로 강조 (CLAUDE.md 7.1절 핵심 논점)
idx_el = list(class_names).index("Edge-Loc")
idx_sc = list(class_names).index("Scratch")
el_total = cm[idx_el].sum()
sc_total = cm[idx_sc].sum()
el_to_sc = cm[idx_el, idx_sc]
sc_to_el = cm[idx_sc, idx_el]
log(f"\n--- Edge-Loc <-> Scratch 혼동 (CLAUDE.md 7.1절 핵심 논점) ---")
log(f"Edge-Loc를 Scratch로 오분류: {el_to_sc}/{el_total} ({el_to_sc / el_total * 100:.1f}%)")
log(f"Scratch를 Edge-Loc로 오분류: {sc_to_el}/{sc_total} ({sc_to_el / sc_total * 100:.1f}%)")

np.savez(OUTPUT_DIR / f"phase4_confusion_matrix_{RUN_TAG}.npz", cm=cm, class_names=class_names)

# %% [markdown]
# ## 5. 저장

# %%
summary_path = OUTPUT_DIR / f"phase4_train_evaluate_summary_{RUN_TAG}.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

meta = {
    "input_mode": INPUT_MODE,
    "head": HEAD,
    "n_folds": N_FOLDS,
    "oof_macro_f1": oof_macro_f1,
    "oof_balanced_accuracy": oof_balanced_acc,
    "oof_accuracy": oof_accuracy,
    "class_names": class_names.tolist(),
}
with open(OUTPUT_DIR / f"phase4_metrics_{RUN_TAG}.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n[완료] 요약 저장: {summary_path}")
print(f"[완료] 혼동행렬 저장: {OUTPUT_DIR / f'phase4_confusion_matrix_{RUN_TAG}.npz'}")
print(f"[완료] 모델 가중치 저장: {ARTIFACTS_DIR}/model_{RUN_TAG}_fold0~4.pt")
