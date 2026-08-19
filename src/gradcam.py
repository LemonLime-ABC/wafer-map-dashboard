# %% [markdown]
# # gradcam.py — Grad-CAM (예측 근거 시각화) 공용 모듈
#
# CNN이 "왜" 이 클래스라고 판단했는지, 웨이퍼 맵의 어느 위치를 보고
# 판단했는지를 히트맵으로 보여준다. SECOM 프로젝트의 SHAP(어느 센서가
# 원인인지 짚어주는 도구)에 대응하는 이미지 버전이다.
#
# ## 원리
# 1. 예측하려는 클래스의 점수(logit)를, 마지막 Conv층의 활성화 값들에
#    대해 미분(역전파)한다 — "이 클래스 점수를 높이려면 이 위치의
#    활성화가 얼마나 더 커져야 하는가"를 위치별 기울기로 얻는다.
# 2. 채널마다 이 기울기의 평균을 내 "이 채널이 이 클래스 판단에 얼마나
#    중요한지" 가중치로 삼는다.
# 3. 그 가중치로 활성화 값들을 가중합하면, 클래스 판단에 크게 기여한
#    위치일수록 값이 큰 히트맵이 나온다.
#
# ## PyTorch 새 문법 — Hook(후크)
# hook은 "레이어를 통과할 때, 또는 그 레이어로 기울기가 흘러들어올 때
# 자동으로 실행되는 콜백 함수"다. `register_forward_hook`은 그 레이어의
# **출력값**을, `register_full_backward_hook`은 역전파 시 그 레이어로
# 들어오는 **기울기**를 가로채서 저장해준다. 모델 코드 자체를 고치지
# 않고도 중간 계산값을 훔쳐볼 수 있게 해주는 장치다.

# %%
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        # 두 hook을 등록해두면, 이후 model(x)와 score.backward()를
        # 호출할 때마다 아래 두 메서드가 자동으로 실행되어 값을 저장한다.
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output은 튜플이다(레이어가 출력을 여러 개 낼 수도 있어서).
        # 우리 레이어는 출력이 하나뿐이라 [0]만 쓴다.
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, target_class: int = None):
        """
        x: (1, C, H, W) 웨이퍼 맵 1장.
        target_class를 안 주면 모델이 예측한 클래스를 기준으로 히트맵을 만든다.
        반환: (히트맵 (H,W) numpy, 사용된 target_class, 그 클래스 확률)
        """
        self.model.eval()
        output = self.model(x)  # (1, n_classes)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()  # 이 시점에 hook들이 activations/gradients를 채워준다

        # 채널별 중요도 = 그 채널 기울기의 공간(가로x세로) 평균
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # 클래스 점수를 "깎아내리는" 방향의 기여는 관심 없으므로 0으로

        # 특징 맵 크기(16x16)를 원본 입력 크기(64x64)로 다시 키운다
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().numpy()

        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max  # 0~1로 정규화 (히트맵끼리 밝기 비교 가능하게)

        # backward() 이후라 probs는 여전히 기울기 추적 상태다. 이제 숫자만
        # 꺼내 쓸 거라 detach()로 추적을 끊어준다(안 끊어도 동작은 하지만
        # 불필요한 경고가 뜬다).
        prob_value = float(probs[0, target_class].detach())
        return cam, target_class, prob_value
