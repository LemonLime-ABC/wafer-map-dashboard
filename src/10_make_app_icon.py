# %% [markdown]
# # 10_make_app_icon — 앱 아이콘 생성
#
# 폰 홈 화면이나 브라우저 탭에 뜨는 아이콘을 실제 웨이퍼 맵으로 만든다.
# 기본 아이콘(Streamlit 로고)이면 홈 화면에서 우리 앱인지 구분이 안 된다.
#
# Edge-Ring 패턴을 고른 이유: 9개 패턴 중 형태가 가장 뚜렷해서
# 작은 크기(48px 등)로 줄여도 무엇인지 알아볼 수 있다.

# %%
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS = PROJECT_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

data = np.load(PROJECT_ROOT / "data" / "processed" / "wm811k_sample.npz")
X, y, cn = data["X"], data["y_encoded"], data["class_names"]

# 해당 클래스에서 불량 비율이 상위권인 표본을 골라 대비를 높인다
# (앞쪽 표본은 불량이 옅어 축소하면 패턴이 안 보인다)
idx = int(np.where(cn == "Edge-Ring")[0][0])
ids = np.where(y == idx)[0]
ratio = [(X[i] == 2).sum() / max((X[i] > 0).sum(), 1) for i in ids]
img = X[ids[np.argsort(ratio)[int(len(ids) * 0.9)]]]

# 배경(0)은 투명하게, 정상 다이(1)는 남색, 불량(2)은 빨강
rgba = np.zeros((*img.shape, 4), dtype=np.float32)
rgba[img == 1] = [0.30, 0.45, 0.69, 1.0]   # #4C72B0
rgba[img == 2] = [0.77, 0.31, 0.32, 1.0]   # #C44E52
# 배경은 alpha=0 그대로 두어 둥근 웨이퍼 모양이 그대로 아이콘 실루엣이 된다

SIZE = 512
fig = plt.figure(figsize=(SIZE / 100, SIZE / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])   # 여백 없이 꽉 채운다
ax.imshow(rgba, interpolation="nearest")
ax.axis("off")

out = ASSETS / "app_icon.png"
fig.savefig(out, dpi=100, transparent=True)
plt.close(fig)

from PIL import Image
im = Image.open(out)
print(f"저장: {out}  ({im.size[0]}x{im.size[1]}, {out.stat().st_size/1024:.1f} KB)")
print(f"사용한 패턴: Edge-Ring / 불량 다이 비율 {(img == 2).sum() / (img > 0).sum() * 100:.1f}%")
