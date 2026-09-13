# %% [markdown]
# # live_inference.py — "즉석 판정" 화면의 계산 부분 (공용 모듈)
#
# 대시보드 화면 6은 사용자가 올린 웨이퍼 맵 파일을 그 자리에서 판정한다.
# 화면 코드(app.py)와 계산 코드를 나눈 이유: 계산 부분을 Streamlit 없이
# 단독으로 실행해 검증할 수 있게 하기 위해서다(scratchpad 검증 스크립트가
# 이 모듈을 그대로 불러 쓴다). 화면에서 조용히 틀린 값이 뜨는 걸 막으려면
# 화면 밖에서 먼저 확인할 수 있어야 한다.
#
# 이 모듈이 하는 일
# 1. 파일 해석 — CSV/TXT/NPY를 2차원 0/1/2 격자로 바꾼다
# 2. 입력 검사 — 학습 데이터와 너무 다른 입력이면 경고한다
# 3. 예측 — 학습 때와 똑같은 전처리(64x64 최근접 리사이즈, /2.0)로 모델에 넣는다

# %%
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import WaferCNN
from resize_utils import resize_nearest

MAX_BYTES = 5_000_000   # 웨이퍼 맵 한 장이 이보다 클 일은 없다(212x212 CSV도 100KB 미만)
MAX_SIDE = 1000         # 비정상적으로 큰 격자는 메모리 문제가 되므로 거절

RULE_WM = "WM-811K 형식 (0=다이 없음 / 1=정상 / 2=불량)"
RULE_BIN = "Bin 코드 형식 (빈칸=다이 없음 / Pass bin / 그 외 번호=Fail)"


# %% [markdown]
# ## 1. 파일 해석
#
# 왜 두 가지 규칙을 받나: WM-811K는 이미 0/1/2로 정리된 데이터지만, 실제 팹의
# 테스트 장비는 다이마다 **bin 번호**를 낸다(Pass는 보통 1번, Fail은 원인별로
# 여러 번호). 현장 파일을 그대로 넣으려면 bin 번호를 0/1/2로 바꾸는 규칙이 필요하다.
# 모델은 "어떤 이유로 Fail인가"는 모르고 "Fail인가"만 보므로, Fail bin들은 전부 2가 된다.

# %%
def _read_cells(raw: bytes, filename: str) -> np.ndarray:
    """파일을 문자열 2차원 배열로 읽는다. 숫자 변환은 규칙에 따라 뒤에서 한다."""
    if filename.lower().endswith(".npy"):
        # allow_pickle=False — 파이썬 객체가 들어 있는 npy를 열면 임의 코드가
        # 실행될 수 있다. 숫자 배열만 받는다.
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"2차원 배열이 아닙니다 (차원 수 {arr.ndim}).")
        return arr.astype(str)

    text = raw.decode("utf-8-sig", errors="replace")   # utf-8-sig: 엑셀이 붙이는 BOM 제거
    # 쉼표가 있으면 CSV, 없으면 공백으로 구분된 텍스트로 본다
    sep = "," if "," in text else r"\s+"
    df = pd.read_csv(io.StringIO(text), header=None, sep=sep, dtype=str,
                     keep_default_na=False, engine="python")
    df = df.fillna("")                    # 행마다 칸 수가 다르면 모자란 칸은 빈칸으로
    cells = df.to_numpy(dtype=str)
    cells = np.char.strip(cells)
    # 줄 끝에 쉼표가 하나 더 붙어 생긴 "완전히 빈 열/행"은 데이터가 아니므로 제거
    cells = cells[:, ~(cells == "").all(axis=0)]
    cells = cells[~(cells == "").all(axis=1), :]
    return cells


def parse_wafer_file(raw: bytes, filename: str, rule: str, pass_bin: int = 1):
    """
    반환: (웨이퍼 맵 uint8 배열 또는 None, 사람이 읽을 오류 메시지 또는 None)
    오류가 나도 예외를 던지지 않는다 — 파일 여러 개 중 하나가 잘못돼도
    나머지는 판정되어야 하기 때문이다.
    """
    if len(raw) > MAX_BYTES:
        return None, f"파일이 너무 큽니다 ({len(raw)/1e6:.1f}MB). 웨이퍼 맵 한 장은 보통 수십 KB입니다."
    try:
        cells = _read_cells(raw, filename)
    except Exception as e:  # 형식이 깨진 파일은 원인을 그대로 보여준다
        return None, f"파일을 읽지 못했습니다: {e}"

    if cells.ndim != 2 or min(cells.shape) < 3:
        return None, f"격자가 너무 작습니다 (크기 {cells.shape})."
    if max(cells.shape) > MAX_SIDE:
        return None, f"격자가 너무 큽니다 (크기 {cells.shape}, 최대 {MAX_SIDE})."

    # 문법 설명: pd.to_numeric(errors="coerce")
    # 숫자로 바꿀 수 없는 칸("", "X" 등)을 오류 대신 NaN으로 만든다.
    # 그다음 "NaN인데 원래 빈칸이 아니었던 칸"을 세면 이상한 글자가 섞였는지 알 수 있다.
    num = pd.to_numeric(pd.Series(cells.ravel()), errors="coerce").to_numpy().reshape(cells.shape)
    bad_text = np.isnan(num) & (cells != "")
    if bad_text.any():
        r, c = np.argwhere(bad_text)[0]
        return None, f"숫자가 아닌 값이 있습니다 (예: {r+1}행 {c+1}열 '{cells[r, c]}')."

    if rule == RULE_WM:
        if np.isnan(num).any():
            return None, ("빈칸이 있습니다. WM-811K 형식은 모든 칸이 0/1/2여야 합니다 — "
                          "빈칸이 '다이 없음'이라면 값 규칙을 Bin 코드 형식으로 바꾸세요.")
        vals = set(np.unique(num).tolist())
        if not vals <= {0.0, 1.0, 2.0}:
            extra = sorted(vals - {0.0, 1.0, 2.0})[:5]
            return None, (f"0/1/2 외의 값이 있습니다 ({extra}). 테스트 장비의 bin 번호라면 "
                          "값 규칙을 Bin 코드 형식으로 바꾸세요.")
        m = num.astype(np.uint8)
    else:
        m = np.zeros(num.shape, dtype=np.uint8)            # 빈칸·음수 = 다이 없음(0)
        is_die = ~np.isnan(num) & (num >= 0)
        m[is_die & (num == pass_bin)] = 1                  # Pass bin = 정상(1)
        m[is_die & (num != pass_bin)] = 2                  # 그 외 bin = 불량(2)

    if not ((m == 1) | (m == 2)).any():
        return None, "다이가 하나도 없습니다 (전부 0 또는 빈칸)."
    return m, None


# %% [markdown]
# ## 2. 입력 검사
#
# softmax 분류기는 무엇을 넣어도 9개 중 하나를 고르고, 확률도 높게 낼 수 있다.
# "확률 95%"가 "이 입력이 학습 데이터와 비슷하다"는 뜻은 아니다.
# 그래서 학습 데이터(라벨된 172,950장)에서 실측한 범위와 비교해 경고한다.
# 경고는 판정을 막지 않는다 — 판단은 사람이 한다.

# %%
def map_summary(m: np.ndarray) -> dict:
    dies = int(((m == 1) | (m == 2)).sum())
    return {
        "shape": m.shape,
        "dies": dies,
        "zero_frac": float((m == 0).mean()),
        "defect_ratio": float((m == 2).sum() / dies) if dies else float("nan"),
    }


def input_warnings(m: np.ndarray, ref: dict) -> list[str]:
    s = map_summary(m)
    out = []
    zlo, zhi = ref["zero_frac_q"]
    # 원형 웨이퍼를 사각 격자에 담으면 모서리가 비어 배경 비율이 약 21%가 된다(1-π/4).
    # 이게 크게 다르면 원형 웨이퍼 맵이 아니거나, 다이 없음 표기가 다른 파일이다.
    if s["zero_frac"] < zlo * 0.8 or s["zero_frac"] > zhi * 1.2:
        out.append(
            f"배경(다이 없음) 비율이 {s['zero_frac']*100:.1f}%입니다. 학습 웨이퍼는 99%가 "
            f"{zlo*100:.1f}~{zhi*100:.1f}% 범위였습니다 — 원형 웨이퍼 맵이 아니거나 "
            "'다이 없음' 표기 규칙이 다를 수 있습니다.")
    dlo, dhi = ref["defect_ratio_q"]
    if s["defect_ratio"] > dhi:
        out.append(
            f"불량 다이 비율이 {s['defect_ratio']*100:.1f}%로, 학습 웨이퍼의 99.5%보다 높습니다 "
            f"(기준 {dhi*100:.1f}%). 전면 불량(Near-full) 계열이거나 학습 때 드물었던 입력입니다.")
    if max(s["shape"]) > ref["side_max"]:
        out.append(
            f"격자 크기 {s['shape']}가 학습 데이터 최대({ref['side_max']})보다 큽니다. "
            "64×64로 줄이면서 가는 선 모양이 뭉개질 수 있습니다(Scratch 성능 한계와 같은 원인).")
    return out


# %% [markdown]
# ## 3. 예측
#
# 전처리를 학습 때와 **정확히 같게** 해야 한다. 학습은 64x64 최근접 리사이즈 후
# 0/1/2를 2.0으로 나눠 0/0.5/1.0으로 넣었다(08_precompute_dashboard.py와 동일).
# 여기서 하나라도 다르면 모델은 학습 때 본 적 없는 값을 받게 된다.

# %%
def to_model_input(m: np.ndarray) -> np.ndarray:
    """(H,W) 0/1/2 -> (64,64) float32 0/0.5/1.0"""
    return resize_nearest(m, 64).astype(np.float32) / 2.0


def load_fold_models(art_dir: Path, n_classes: int) -> list:
    """5개 fold 모델을 전부 불러온다. 미라벨 웨이퍼는 어떤 fold도 학습에 쓰지 않았으므로
    다섯 모델 모두 이 입력에 대해 공정한 '처음 보는' 판정을 낸다."""
    models = []
    for k in range(5):
        mdl = WaferCNN(in_channels=1, n_classes=n_classes, head="flatten")
        state = torch.load(Path(art_dir) / f"model_v2_flatten_fold{k}.pt",
                           map_location="cpu", weights_only=True)
        mdl.load_state_dict(state)
        mdl.eval()
        models.append(mdl)
    return models


def predict_probs(maps: list, models: list) -> np.ndarray:
    """반환: (모델 수, 웨이퍼 수, 클래스 수) 확률 배열"""
    x = np.stack([to_model_input(m) for m in maps])[:, None, :, :]
    xt = torch.from_numpy(x)
    # 문법 설명: with torch.no_grad():
    # 이 블록 안에서는 PyTorch가 역전파용 계산 기록을 남기지 않는다.
    # 판정만 할 때는 기울기가 필요 없으므로 메모리와 시간을 아낀다.
    # (Grad-CAM은 기울기가 필요해서 이 함수를 쓰지 않고 따로 계산한다)
    with torch.no_grad():
        return np.stack([torch.softmax(mdl(xt), dim=1).numpy() for mdl in models])
