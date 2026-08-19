# %% [markdown]
# # augment.py — 웨이퍼 맵 증강 (공용 모듈)
#
# 회전 4방향(0/90/180/270도) x 좌우 뒤집기 2가지 = 8배로 증강한다.
# 웨이퍼 맵은 회전·뒤집기를 해도 불량 패턴의 "종류"가 바뀌지 않는다
# (Center를 90도 돌려도 여전히 Center, Edge-Ring도 마찬가지) — 그래서
# 이 증강이 물리적으로 타당하다(PLAN.md 3.2절). 일반 사진에 이런 증강을
# 쓰면 안 되는 경우가 많다(예: 숫자 '6'을 180도 돌리면 '9'가 됨) — 웨이퍼
# 맵은 그런 문제가 없는 특수한 경우다.
#
# **반드시 학습 fold 내부에서만 호출한다.** 분할 전에 증강하면 같은
# 원본의 회전판이 학습셋과 검증셋에 나뉘어 들어가는 데이터 누수가
# 생긴다 — CLAUDE.md 6.1절 최우선 원칙이며, SECOM의 SMOTE 누수와
# 정확히 같은 종류의 실수다.

# %%
import numpy as np


def augment_batch(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (N, H, W) 웨이퍼 맵 배치를 회전+뒤집기로 8배 증강해서 반환한다.
    라벨(y)은 원본 값 그대로 8번 복제된다 (회전/뒤집기가 클래스를 바꾸지
    않으므로 라벨을 바꿀 이유가 없다).
    """
    # -----------------------------------------------------------
    # 문법 설명: zip(), np.rot90(), np.fliplr()
    # -----------------------------------------------------------
    # zip(X, y)는 X와 y를 짝지어 (img, label) 쌍을 하나씩 꺼내준다 —
    # "이미지와 그 라벨을 같이 순회하고 싶을 때" 쓰는 표준 패턴.
    # np.rot90(img, k)는 img를 반시계로 90도씩 k번 돌린다(k=0이면 원본).
    # np.fliplr(img)는 좌우(가로 방향)로 뒤집는다.
    # -----------------------------------------------------------
    aug_X, aug_y = [], []
    for img, label in zip(X, y):
        for k in range(4):  # 0, 90, 180, 270도
            rotated = np.rot90(img, k)
            aug_X.append(rotated)
            aug_y.append(label)
            aug_X.append(np.fliplr(rotated))  # 각 회전본마다 좌우 뒤집기 버전도 추가
            aug_y.append(label)
    return np.stack(aug_X), np.array(aug_y)
