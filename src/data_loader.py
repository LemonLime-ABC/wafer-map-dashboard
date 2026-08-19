# %% [markdown]
# # data_loader.py — WM-811K 로더 (공용 모듈)
#
# Phase 1에서 겪은 세 가지 호환성 문제(옛 pandas Index 모듈 위치 /
# Python2 문자열 인코딩 / `trianTestLabel` 오타)를 한 곳에 모아뒀다.
# 앞으로 Phase 2~4의 모든 스크립트는 이 파일의 `load_wm811k()`
# 하나만 불러 쓰면 되고, 저 세 가지 문제를 다시 겪지 않는다.
#
# 이 파일은 직접 실행하는 스크립트가 아니라, 다른 스크립트에서
# `from data_loader import load_wm811k` 형태로 "가져다 쓰는" 모듈이다.

# %%
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.compat.pickle_compat as pickle_compat

_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent  # src/의 상위 폴더 = 프로젝트 루트

# WM-811K가 만들어진 시절(pandas 0.2x대)의 Index 하위 모듈들을
# 지금 pandas의 대응 위치로 미리 연결해 둔다. (Phase 1에서 확인한 패치)
_OLD_TO_NEW_INDEX_MODULES = {
    "pandas.indexes": "pandas.core.indexes.base",
    "pandas.indexes.base": "pandas.core.indexes.base",
    "pandas.indexes.range": "pandas.core.indexes.range",
    "pandas.indexes.numeric": "pandas.core.indexes.base",
    "pandas.indexes.multi": "pandas.core.indexes.multi",
    "pandas.indexes.category": "pandas.core.indexes.category",
    "pandas.indexes.datetimes": "pandas.core.indexes.datetimes",
    "pandas.indexes.timedeltas": "pandas.core.indexes.timedeltas",
    "pandas.indexes.period": "pandas.core.indexes.period",
    "pandas.indexes.frozen": "pandas.core.indexes.frozen",
}


def _patch_old_pandas_pickle_compat() -> None:
    """옛 pandas 모듈 이름을 지금 pandas의 대응 모듈로 sys.modules에 등록한다."""
    for old_name, new_name in _OLD_TO_NEW_INDEX_MODULES.items():
        try:
            sys.modules.setdefault(old_name, importlib.import_module(new_name))
        except ImportError:
            pass  # 지금 pandas 버전에 아예 없는 모듈이면 건너뛴다


def _extract_label(x):
    """중첩된 numpy 배열에서 실제 라벨 문자열 하나를 꺼낸다. 없으면 None."""
    arr = np.asarray(x)
    flat = arr.flatten()
    if flat.size == 0:
        return None
    return str(flat[0])


def find_pkl_file(data_dir: Path) -> Path:
    """data_dir 폴더 안에서 .pkl 확장자 파일을 찾아 첫 번째 것을 반환한다."""
    pkl_files = list(Path(data_dir).glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"'{data_dir}' 폴더에 .pkl 파일이 없습니다. "
            "Kaggle(qingyi/wm811k-wafer-map)에서 LSWMD.pkl을 받아 이 폴더에 넣으세요."
        )
    return pkl_files[0]


def load_wm811k(data_dir: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    WM-811K 데이터를 로드해서 아래 3개 컬럼이 추가된 DataFrame을 반환한다.

    - failureType_clean   : 정리된 라벨 문자열 (없으면 None)
    - trainTestLabel_clean: 정리된 Training/Test 구분 문자열 (없으면 None)
    - map_shape           : waferMap의 (세로, 가로) 크기 튜플
    """
    # -----------------------------------------------------------
    # 문법 설명: Path | None 타입 힌트
    # -----------------------------------------------------------
    # 함수 정의의 `data_dir: Path | None = None`은 "data_dir 인자는
    # Path 타입이거나 None일 수 있고, 안 주면 기본값 None을 쓴다"는 뜻이다.
    # 타입 힌트는 강제되지 않고(파이썬이 실제로 검사하지 않음) 사람이
    # 읽을 때 이 함수가 뭘 기대하는지 알려주는 주석 같은 역할을 한다.
    # -----------------------------------------------------------
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"

    _patch_old_pandas_pickle_compat()
    pkl_path = find_pkl_file(data_dir)

    if verbose:
        print(f"[data_loader] 로드할 파일: {pkl_path.name} "
              f"({pkl_path.stat().st_size / 1e6:.1f} MB)")

    with open(pkl_path, "rb") as f:
        df = pickle_compat.Unpickler(f, encoding="latin1").load()

    df["failureType_clean"] = df["failureType"].apply(_extract_label)
    # 원본 데이터셋 자체의 오타(trainTestLabel -> trianTestLabel)를 그대로 참조
    df["trainTestLabel_clean"] = df["trianTestLabel"].apply(_extract_label)
    df["map_shape"] = df["waferMap"].apply(lambda m: m.shape)

    if verbose:
        print(f"[data_loader] 로드 완료: {len(df):,}장")

    return df
