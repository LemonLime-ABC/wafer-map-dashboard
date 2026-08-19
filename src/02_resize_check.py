# %% [markdown]
# # 02_resize_check — 리사이즈 방식 검증
#
# Phase 1에서 "웨이퍼 맵을 전부 64x64로 리사이즈"하기로 결정했다.
# 이 스크립트는 그 결정을 실제 데이터로 검증한다.
#
# 확인할 것 3가지:
# 1. 리사이즈 후에도 픽셀 값이 {0, 1, 2} 밖으로 절대 벗어나지 않는가
#    (벗어나면 = 보간이 섞여 들어갔다는 뜻 = 심각한 버그)
# 2. 사람 눈으로 봤을 때 리사이즈 전/후 패턴의 형태가 유지되는가
#    (예: Edge-Ring이 리사이즈 후에도 여전히 가장자리 링 모양인가)
# 3. 실제 데이터 규모(라벨된 172,950장)를 리사이즈하는 데 시간이 얼마나 걸리는가

# %%
import sys
import time
from pathlib import Path

# Windows 콘솔 코드페이지(cp949)가 표현 못 하는 유니코드 문자(—) 때문에
# print()가 죽지 않도록 출력 인코딩을 UTF-8로 강제한다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt

# matplotlib 기본 폰트(DejaVu Sans)는 한글 글리프가 없어 그래프 제목이
# 네모(□)로 깨진다. Windows에 기본 내장된 '맑은 고딕'으로 바꿔준다.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 위 폰트에서 마이너스(-) 기호가 깨지는 것 방지

# -----------------------------------------------------------
# 문법 설명: sys.path.append와 우리가 만든 모듈 불러오기
# -----------------------------------------------------------
# data_loader.py, resize_utils.py는 표준 라이브러리가 아니라 우리가
# 이 프로젝트에 직접 만든 파일이다. 같은 폴더(src/)에 있으면 파이썬이
# 자동으로 찾아 import할 수 있다 — 이 스크립트를 src/ 폴더 안에서
# 실행하는 한 별도 조치가 필요 없다.
# -----------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_wm811k  # noqa: E402
from resize_utils import resize_nearest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SIZE = 64
RNG_SEED = 42

# %% [markdown]
# ## 1. 데이터 로드 및 라벨된 데이터만 추출

# %%
df = load_wm811k()

labeled = df[df["failureType_clean"].notna()].reset_index(drop=True)
classes = sorted(labeled["failureType_clean"].unique())
print(f"[1] 라벨된 웨이퍼: {len(labeled):,}장, 클래스: {classes}")

# %% [markdown]
# ## 2. 클래스별 샘플 1장씩 리사이즈 + 픽셀 값 검증
#
# `assert`는 "이 조건이 거짓이면 즉시 에러를 내고 멈춰라"는 뜻의 파이썬
# 문법이다. 여기서는 "리사이즈 후 값이 전부 {0,1,2} 안에 있어야 한다"는
# 조건을 걸어, 만약 보간 때문에 이상한 값이 섞여 들어가면 그 즉시
# 알아챌 수 있게 한다 — 조용히 넘어가서 나중에 원인 모를 버그가 되는 것을 막는다.

# %%
rng = np.random.default_rng(RNG_SEED)

fig, axes = plt.subplots(len(classes), 2, figsize=(6, 3 * len(classes)))

example_rows = []
for i, cls in enumerate(classes):
    subset = labeled[labeled["failureType_clean"] == cls]
    row = subset.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
    original = row["waferMap"]
    resized = resize_nearest(original, TARGET_SIZE)

    # ---- 검증: 리사이즈 후 값이 원본에 없던 값으로 오염되지 않았는가 ----
    original_values = set(np.unique(original))
    resized_values = set(np.unique(resized))
    assert resized_values.issubset(original_values), (
        f"[{cls}] 리사이즈 후 원본에 없던 값이 생겼습니다: "
        f"{resized_values - original_values}"
    )

    example_rows.append((cls, original.shape, original_values, resized_values))

    axes[i, 0].imshow(original, cmap="viridis", vmin=0, vmax=2)
    axes[i, 0].set_title(f"{cls} 원본 {original.shape}", fontsize=9)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(resized, cmap="viridis", vmin=0, vmax=2)
    axes[i, 1].set_title(f"{cls} 리사이즈 {TARGET_SIZE}x{TARGET_SIZE}", fontsize=9)
    axes[i, 1].axis("off")

plt.tight_layout()
fig_path = OUTPUT_DIR / "resize_before_after.png"
plt.savefig(fig_path, dpi=120)
plt.close()

print(f"[2] 픽셀 값 검증 통과 (모든 클래스에서 원본에 없던 값 없음)")
print(f"    -> 비교 이미지 저장됨: {fig_path}")
for cls, shape, orig_vals, resz_vals in example_rows:
    print(f"    {cls:10s} 원본{str(shape):>12s} 원본값{sorted(orig_vals)} -> 리사이즈값{sorted(resz_vals)}")

# %% [markdown]
# ## 3. 라벨된 데이터 전체(172,950장) 리사이즈 — 소요 시간 측정
#
# Phase 3에서는 이보다 훨씬 적은 수(PLAN.md 기준 약 12,763장)만 리사이즈하므로,
# 여기서 172,950장 전체를 재는 것은 "최악의 경우에도 이 정도면 충분히 빠르다"는
# 걸 확인하기 위함이다.

# %%
start = time.time()
resized_maps = [resize_nearest(m, TARGET_SIZE) for m in labeled["waferMap"]]
elapsed = time.time() - start

print(f"[3] 라벨된 {len(labeled):,}장 리사이즈 완료: {elapsed:.1f}초 "
      f"(장당 평균 {elapsed / len(labeled) * 1000:.3f}ms)")

# 최종 배열 형태 확인 (전부 64x64로 통일됐는지)
resized_stack = np.stack(resized_maps)
print(f"    통일된 배열 형태: {resized_stack.shape} (전체 장수, {TARGET_SIZE}, {TARGET_SIZE})")

# %% [markdown]
# ## 4. 요약 저장

# %%
summary_lines = [
    "=== 리사이즈 방식 검증 요약 ===",
    f"방식: 최근접 이웃(nearest neighbor), 목표 크기 {TARGET_SIZE}x{TARGET_SIZE}",
    f"픽셀 값 오염 검증: 통과 (9개 클래스 전부에서 원본에 없던 값 없음)",
    f"라벨된 전체 {len(labeled):,}장 리사이즈 소요 시간: {elapsed:.1f}초",
    f"장당 평균: {elapsed / len(labeled) * 1000:.3f}ms",
    f"최종 배열 형태: {resized_stack.shape}",
]
summary_text = "\n".join(summary_lines)
with open(OUTPUT_DIR / "resize_check_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

print("\n" + summary_text)
print(f"\n[4] 요약 저장됨: {OUTPUT_DIR / 'resize_check_summary.txt'}")
