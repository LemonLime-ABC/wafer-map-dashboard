# %% [markdown]
# # shape_features.py — 불량 패턴의 "선형성" 계산 (공용 모듈)
#
# Scratch(선 모양)와 Loc(뭉친 모양)이 계속 헷갈리는 문제를 겨냥해, CNN이
# 이미지만 보고 다시 배우게 하지 않고 "이 패턴이 얼마나 길쭉한가"를
# 숫자 하나로 직접 계산해서 알려준다.
#
# ## 원리 — 주성분분석(PCA)의 고윳값 비율
#
# 불량 다이들의 (행, 열) 좌표를 점 구름으로 보면:
# - **선 모양(Scratch)**: 점들이 한 방향으로 길게 늘어서 있다 → 그 방향의
#   분산은 크고, 수직 방향 분산은 작다.
# - **뭉친 모양(Loc)**: 점들이 사방으로 고르게 퍼져있다 → 어느 방향이든
#   분산이 비슷하다.
#
# 이 "분산이 한쪽으로 쏠린 정도"를 재는 표준적인 방법이 공분산 행렬의
# 고윳값(eigenvalue) 두 개를 비교하는 것이다. 큰 고윳값을 λ1, 작은 걸
# λ2라 하면:
#
#     elongation = λ1 / (λ1 + λ2)
#
# 완전히 둥근 뭉치는 λ1≈λ2라 0.5에 가깝고, 완전히 곧은 선은 λ2≈0이라
# 1.0에 가깝다.

# %%
import numpy as np
from scipy import ndimage

# 대각선 방향도 "붙어있다"고 인정하는 8방향 연결 구조.
# (상하좌우 4방향만 인정하면 대각선으로 이어진 스크래치 선이 여러 조각으로
#  쪼개져 버린다 — 스크래치는 흔히 대각선 방향이라 이게 중요하다)
_CONNECTIVITY_8 = np.ones((3, 3), dtype=int)


def compute_elongation(wafer_map: np.ndarray) -> float:
    """
    웨이퍼 맵 하나에서 불량 다이(값==2)의 선형성을 0.5(둥근 뭉치)~1.0(곧은 선)
    사이 숫자로 계산한다.

    1차 시도(전체 불량 칸을 다 모아서 계산)는 흩어진 잡음성 불량이 방향성을
    희석시켜 Scratch와 Loc을 오히려 잘 못 구분했다. 그래서 이번엔 "서로
    붙어있는 덩어리(연결 요소)" 중 **가장 큰 것 하나만** 골라서, 그 덩어리의
    모양만 잰다 — 잡음(작은 조각들)은 무시하고 진짜 패턴만 본다.
    """
    defect_mask = (wafer_map == 2)
    if defect_mask.sum() < 3:
        return 0.5

    # 문법 설명: scipy.ndimage.label
    # 서로 붙어있는 True 칸들을 하나의 "덩어리"로 묶어 번호(1,2,3,...)를
    # 매긴 배열을 돌려준다. 두 번째 반환값은 덩어리 개수.
    labeled, n_components = ndimage.label(defect_mask, structure=_CONNECTIVITY_8)
    if n_components == 0:
        return 0.5

    # 덩어리별 크기(칸 수)를 세서 가장 큰 것의 번호를 찾는다.
    sizes = ndimage.sum(defect_mask, labeled, index=range(1, n_components + 1))
    largest_label = np.argmax(sizes) + 1  # 라벨은 1번부터 시작

    rows, cols = np.where(labeled == largest_label)
    if len(rows) < 3:
        return 0.5

    coords = np.stack([rows, cols], axis=1).astype(np.float64)  # (N, 2)
    coords -= coords.mean(axis=0)  # 중심을 원점으로 이동 (분산 계산 준비)

    # 문법 설명: np.cov(rowvar=False)
    # rowvar=False는 "각 열이 변수(행 좌표, 열 좌표), 각 행이 관측치"라는
    # 뜻이다 — 기본값(rowvar=True)은 반대로 해석해서 결과가 완전히 달라지니
    # 주의해야 한다. 결과는 2x2 공분산 행렬.
    cov = np.cov(coords, rowvar=False)

    # 문법 설명: np.linalg.eigvalsh
    # 대칭 행렬(공분산 행렬은 항상 대칭)의 고윳값만 빠르고 안정적으로
    # 계산해준다. 오름차순으로 반환하므로 [0]=작은 값(λ2), [1]=큰 값(λ1).
    eigvals = np.linalg.eigvalsh(cov)
    lam2, lam1 = eigvals[0], eigvals[1]

    if lam1 + lam2 < 1e-9:
        return 0.5  # 모든 점이 한 자리에 겹쳐있는 등 극단적 케이스 방지

    return float(lam1 / (lam1 + lam2))


def compute_elongation_batch(X: np.ndarray) -> np.ndarray:
    """(N, H, W) 웨이퍼 맵 배치 전체에 대해 선형성을 계산해 (N,) 배열로 반환."""
    return np.array([compute_elongation(m) for m in X], dtype=np.float32)
