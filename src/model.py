# %% [markdown]
# # model.py — CNN 모델 정의 (공용 모듈)
#
# PLAN.md 4.1절 구조를 기본으로 하되, "마지막 head(머리)"를 GAP과
# Flatten 두 가지 중 골라 쓸 수 있게 만들었다.
#
# Conv(32) -> Conv(64) -> Conv(128) -> [GAP 또는 Flatten] -> Dropout -> Linear(9)
#
# **Phase 4 1차 학습 결과**: GAP(Global Average Pooling)을 쓰면 위치·모양
# 정보(어디에 뭉쳐 있는지, 어떤 형태인지)가 사라져, "밀도"만으로 구분되는
# 클래스(Edge-Ring/Random/Near-full/none)는 잘 맞히지만 "위치·모양"이
# 정체성인 클래스(Center/Loc/Scratch/Donut)는 성능이 떨어졌다(특히 Loc이
# 다른 모든 클래스와 뒤섞이는 허브가 됨). Flatten head는 8x8 격자의 위치
# 정보를 그대로 다음 층에 넘겨서 이 문제를 완화하려는 시도다.
#
# ## PyTorch가 처음이므로: `nn.Module` 상속 구조 설명
#
# PyTorch에서 신경망은 `nn.Module`을 상속받아 만든다. 지켜야 할 규칙은 2개뿐이다.
#
# 1. `__init__`에서 `super().__init__()`을 **제일 먼저** 호출하고(PyTorch가
#    내부적으로 가중치를 추적할 수 있도록 준비하는 과정이다), 이 신경망이
#    쓸 "층(layer)"들을 `self.xxx = ...` 형태로 등록해둔다. 각 층은 학습
#    가능한 가중치를 자기 안에 갖고 있는 객체다(예: `nn.Conv2d`는 필터
#    가중치를, `nn.Linear`는 행렬 가중치를 갖고 있다).
# 2. `forward(self, x)`에 "입력 x가 각 층을 어떤 순서로 통과하는지"를 적는다.
#    나중에 `model(x)`라고 호출하면 PyTorch가 알아서 이 `forward`를 실행해준다
#    (`model.forward(x)`라고 직접 안 써도 됨 — `__call__`이 내부적으로 연결해줌).
#
# `__init__`(부품 목록)과 `forward`(조립 순서)를 나누는 이유: 부품이 뭔지와
# 그걸 어떻게 쓰는지를 분리해두면 구조를 한눈에 파악하기 쉽고, 같은 부품을
# forward 안에서 여러 번 재사용하기도 편하다.

# %%
import torch
import torch.nn as nn


class WaferCNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 9, head: str = "gap",
                 n_aux_features: int = 0):
        """
        head="gap"    : 기존 방식. 8x8 위치 정보를 평균내 1개 값으로 압축.
        head="flatten": 8x8x128을 안 뭉개고 그대로 펴서 Dense층에 넘김.
                        위치 정보가 살아남는 대신 파라미터가 훨씬 많아진다.
        n_aux_features: CNN이 이미지에서 직접 뽑은 특징 말고, 우리가 미리
                        계산해서 "정답 힌트"처럼 얹어주는 보조 숫자 개수.
                        (예: shape_features.compute_elongation의 선형성 값 1개)
                        flatten head에서만 사용한다.
        """
        super().__init__()
        self.n_aux_features = n_aux_features
        if head not in ("gap", "flatten"):
            raise ValueError(f"head는 'gap' 또는 'flatten'이어야 합니다: {head}")
        self.head = head

        # nn.Sequential: 층들을 순서대로 쭉 이어붙인 묶음. forward에서
        # 층을 하나하나 안 불러도 self.features(x) 한 줄로 전부 통과시킬 수 있다.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.dropout = nn.Dropout(0.3)

        if head == "gap":
            # Global Average Pooling: 채널별로 남은 공간을 통째로 평균내
            # 값 1개로 압축한다. "어디에" 있었는지는 버리고 "얼마나"
            # 있었는지만 남긴다. AdaptiveAvgPool2d(1)은 입력 크기가
            # 몇이든 상관없이 항상 1x1로 줄여준다.
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, n_classes)
        else:  # flatten
            # 문법 설명: nn.AdaptiveMaxPool2d((8, 8))
            # 64x64 입력은 3번의 MaxPool을 거치면 8x8이 되지만, 나중에
            # 더 큰 입력(예: 128x128)을 쓰면 3번 거쳐도 16x16이 남는다.
            # AdaptiveMaxPool2d((8, 8))는 입력 공간 크기가 몇이든(8x8이든
            # 16x16이든) **항상 8x8로 맞춰서 내보낸다** — 그래야 그 뒤에
            # 오는 Linear층 크기(128*8*8)를 입력 해상도가 달라져도 안
            # 바꿔도 된다. Avg가 아니라 Max를 쓰는 이유: 평균은 값을
            # 섞어서 뭉개지만(GAP과 같은 문제), Max는 그 구역에서 제일
            # 강한 신호만 남겨 형태 정보를 상대적으로 덜 지운다.
            self.pool_to_fixed = nn.AdaptiveMaxPool2d((8, 8))
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(128 * 8 * 8, 128)
            self.relu_fc = nn.ReLU()
            # 보조 특징이 있으면 마지막 층 입력 크기에 그만큼 더해준다.
            self.fc = nn.Linear(128 + n_aux_features, n_classes)

    def forward(self, x, aux=None):
        x = self.features(x)  # (batch, 128, H, H) — H는 입력 해상도에 따라 다름(64 입력이면 8, 128 입력이면 16)

        if self.head == "gap":
            x = self.gap(x).flatten(1)  # (batch, 128, 1, 1) -> (batch, 128)
            x = self.dropout(x)
            return self.fc(x)

        # flatten head
        x = self.pool_to_fixed(x)  # (batch, 128, H, H) -> (batch, 128, 8, 8)로 고정
        x = self.flatten(x)        # (batch, 8192)
        x = self.fc1(x)            # (batch, 128) - 위치 정보를 요약하는 층
        x = self.relu_fc(x)

        if self.n_aux_features > 0:
            # 문법 설명: torch.cat([...], dim=1)
            # 두 텐서를 이어붙인다. dim=1은 "배치 축(0번)은 그대로 두고,
            # 특징 축(1번)을 이어붙여라"는 뜻 — (batch,128)과 (batch,1)을
            # 이어붙이면 (batch,129)가 된다. CNN이 이미지에서 배운 128개
            # 특징 옆에, 우리가 직접 계산한 선형성 값을 129번째 특징처럼
            # 나란히 놓아주는 것이다.
            x = torch.cat([x, aux], dim=1)

        x = self.dropout(x)
        return self.fc(x)          # (batch, 9) - 클래스별 점수(logit, 아직 확률 아님)
