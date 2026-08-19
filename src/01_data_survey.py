# %% [markdown]
# # Phase 1 — WM-811K 데이터 실태 조사
#
# 이 파일은 VSCode에서 `# %%`로 나뉜 "셀" 단위로 실행할 수도 있고(Jupyter처럼),
# 터미널에서 `python src/01_data_survey.py`로 처음부터 끝까지 한 번에 실행할 수도 있다.
#
# 확인할 것 4가지 (PLAN.md Phase 1과 동일):
# 1. 클래스별 실제 장수 (문헌 기준 표와 대조)
# 2. 웨이퍼 맵 크기 분포 (Phase 3 샘플링 전략을 결정짓는 핵심 변수)
# 3. Lot 정보 존재 여부와 형태 (Phase 6 관리도 가능 여부를 좌우)
# 4. 픽셀 값 분포 (0/1/2 세 값만 있는지)

# %%
import sys
import pickle  # noqa: F401  (참고용 import — 실제 로드는 pandas.read_pickle을 씀)
from pathlib import Path
from collections import Counter

# Windows 콘솔 코드페이지(cp949)는 —(줄대시) 같은 일부 유니코드 문자를
# 표현하지 못해 print()가 그대로 에러를 낸다. 출력 인코딩을 UTF-8로
# 강제로 바꿔서(표현 못 하는 문자는 물음표로 대체) 죽지 않게 한다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 문법 설명 1: pathlib.Path
# ------------------------------------------------------------------
# os.path.join("data", "LSWMD.pkl") 대신 Path("data") / "LSWMD.pkl" 처럼
# '/' 연산자로 경로를 이어붙일 수 있게 해주는 표준 라이브러리다.
# 문자열보다 안전한 이유: 윈도우(\)와 리눅스(/) 경로 구분자를 자동으로 맞춰준다.
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 이 스크립트(src/)의 상위 폴더 = 프로젝트 루트
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 0. 데이터 파일 자동 탐색
#
# CLAUDE.md에는 파일명이 `WM811K.pkl`로 적혀 있지만, Kaggle에서 실제로 받으면
# `LSWMD.pkl`이라는 이름이다. 파일명을 하드코딩하지 않고 `data/` 폴더 안의
# `.pkl` 확장자를 가진 파일을 자동으로 찾도록 만들었다 — 이름이 뭐든 상관없이 동작한다.

# %%
# ------------------------------------------------------------------
# 문법 설명 2: Path.glob()과 리스트 컴프리헨션
# ------------------------------------------------------------------
# glob("*.pkl")은 DATA_DIR 폴더 안에서 이름이 무엇이든 확장자가 .pkl인
# 파일을 모두 찾아준다 (와일드카드 검색). list(...)로 감싸면 그 결과를
# 하나씩 꺼내볼 수 있는 리스트로 바꾼다.
# ------------------------------------------------------------------
pkl_files = list(DATA_DIR.glob("*.pkl"))

if not pkl_files:
    raise FileNotFoundError(
        f"'{DATA_DIR}' 폴더에 .pkl 파일이 없습니다.\n"
        "Kaggle(https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)에서 "
        "LSWMD.pkl을 내려받아 이 폴더에 넣은 뒤 다시 실행하세요."
    )

DATA_PATH = pkl_files[0]
print(f"[0] 사용할 데이터 파일: {DATA_PATH.name}  (크기: {DATA_PATH.stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ## 1. 데이터 로드
#
# `pd.read_pickle`은 파이썬 객체(여기서는 pandas DataFrame)를 그대로 파일에서
# 복원한다. csv와 달리 문자열로 저장하는 게 아니라 파이썬 객체 구조 그대로
# 저장/복원하기 때문에, numpy 배열이 들어있는 컬럼도 그대로 살아 돌아온다.
#
# **주의**: 원본 파일이 꽤 크기 때문에 로드에 1~2분 걸릴 수 있다.
#
# ### 왜 호환성 패치가 필요한가
# `LSWMD.pkl`은 2018년경 **아주 오래된 pandas(0.2x대)**로 만들어진 파일이다.
# 그 이후 pandas가 내부 구조를 여러 번 재편하면서(특히 `Index` 관련 클래스들의
# 위치), 지금 설치된 최신 pandas는 파일 속에 적힌 옛 모듈 경로
# (`pandas.indexes`)를 더 이상 갖고 있지 않다. 데이터가 손상된 게 아니라
# **순전히 "이삿짐 주소가 바뀐" 문제**다.

# %%
# ------------------------------------------------------------------
# 문법 설명 2.5: sys.modules, importlib.import_module(), dict.items()
# ------------------------------------------------------------------
# sys.modules는 "지금까지 import된 모듈 이름 -> 모듈 객체"를 저장해 둔
# 파이썬 내부 딕셔너리다. pickle이 파일 속의 옛 이름(예: "pandas.indexes.range")을
# import하려 할 때, 이 딕셔너리에 그 이름을 미리 등록해 두면 실제로 없는
# 모듈 대신 우리가 지정한 대체 모듈을 돌려받는다 — 옛 주소로 온 우편을
# 새 주소로 돌려보내는 것과 같다.
#
# importlib.import_module("경로.문자열")은 import 문을 문자열로 대신 실행하는
# 함수다. 여러 모듈을 반복문으로 한꺼번에 등록해야 해서 `import a.b.c` 문법
# 대신 이 함수를 쓴다.
#
# 딕셔너리(dict)는 "옛 이름: 새 이름" 쌍을 여러 개 저장하고,
# .items()로 (키, 값) 쌍을 하나씩 꺼내 for문을 돌릴 수 있다.
# ------------------------------------------------------------------
import sys
import importlib

# WM-811K가 만들어진 시절(pandas 0.2x대)의 Index 하위 모듈들을
# 지금 pandas의 대응 위치로 미리 연결해 둔다.
# Int64Index/Float64Index 등(numeric.py)은 pandas 2.0에서 아예 삭제되고
# 일반 Index로 통합되었기 때문에 base 모듈로 대신 연결한다.
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

for _old_name, _new_name in _OLD_TO_NEW_INDEX_MODULES.items():
    try:
        sys.modules.setdefault(_old_name, importlib.import_module(_new_name))
    except ImportError:
        pass  # 지금 pandas 버전에 아예 없는 모듈이면 건너뛴다

# ------------------------------------------------------------------
# 문법 설명 2.6: 왜 pd.read_pickle 대신 pickle_compat.Unpickler를 직접 쓰는가
# ------------------------------------------------------------------
# 이 파일은 Python 2 시절에 만들어졌다. Python 2는 "문자열"과 "이진 데이터"를
# 구분하지 않고 둘 다 str 타입 하나로 저장했는데, pickle 안의 numpy 배열
# 실제 픽셀 데이터도 이 방식으로 들어있다. Python 3의 pickle이 이걸 읽을 때
# 기본값(encoding="ASCII")으로 "사람이 읽는 텍스트"인 것처럼 해석하려다가,
# 텍스트가 아닌 순수 이진 바이트를 만나 UnicodeDecodeError가 났다.
#
# pd.read_pickle()은 이 encoding 옵션을 바깥에서 바꿀 방법을 제공하지 않으므로,
# pandas가 내부적으로 쓰는 호환 유틸리티(pickle_compat.Unpickler)를 직접
# 불러와서 encoding="latin1"을 지정한다. latin1은 0~255 바이트 전부에
# 대응하는 문자가 있어 디코딩 실패가 나지 않는다 — 오래된 이진 pickle을
# 읽을 때 쓰는 표준적인 우회법이다. (앞서 등록한 sys.modules 패치는 이
# 경로를 써도 그대로 적용된다 — __import__가 sys.modules 캐시를 먼저 보기 때문.)
# ------------------------------------------------------------------
import pandas.compat.pickle_compat as pickle_compat

print("[1] 데이터 로드 중... (시간이 걸릴 수 있습니다)")
try:
    with open(DATA_PATH, "rb") as f:
        df = pickle_compat.Unpickler(f, encoding="latin1").load()
except ModuleNotFoundError as e:
    print(f"    로드 실패: {e}")
    print(
        "    -> 패치를 더 추가해야 합니다. 이 에러 메시지 전체를 그대로 "
        "복사해서 알려주세요 (오래된 파일이라 이런 호환성 에러가 한 번 더 "
        "나올 수 있습니다 — 정상적인 과정입니다)."
    )
    raise

print(f"    전체 행(웨이퍼 맵) 개수: {len(df):,}")
print(f"    컬럼 목록: {list(df.columns)}")
print(df.head())

# %% [markdown]
# ## 2. 라벨 컬럼 정리
#
# WM-811K는 원래 MATLAB 구조체를 파이썬으로 옮긴 데이터라서, `failureType`과
# `trainTestLabel` 컬럼의 각 값이 문자열이 아니라 **배열 속에 배열이 중첩된 형태**로
# 들어있다. 예를 들어:
#
# ```
# 라벨 있음:  array([['Center']], dtype='<U6')   <- 2차원 배열 속에 문자열 1개
# 라벨 없음:  array([], dtype=float64)            <- 빈 배열
# ```
#
# 이 상태로는 `value_counts()` 같은 집계가 안 되므로, 중첩을 풀어 순수 문자열
# 또는 `None`으로 바꾸는 함수를 만든다.

# %%
def extract_label(x):
    """
    중첩된 numpy 배열에서 실제 라벨 문자열 하나를 꺼낸다.
    배열이 비어 있으면(=라벨 없음) None을 반환한다.
    """
    # -----------------------------------------------------------
    # 문법 설명 3: np.array(x).flatten()
    # -----------------------------------------------------------
    # flatten()은 몇 겹으로 중첩되어 있든 상관없이 모든 원소를
    # 1차원으로 평평하게 펴준다. [['Center']] -> ['Center']
    # 빈 배열이면 flatten해도 크기가 0인 빈 배열 그대로다.
    # -----------------------------------------------------------
    arr = np.asarray(x)
    flat = arr.flatten()
    if flat.size == 0:
        return None
    return str(flat[0])


# 주의: 원본 데이터셋 자체에 오타가 있다 — 컬럼명이 "trainTestLabel"이 아니라
# "trianTestLabel"(train -> trian)로 잘못 저장되어 있다. 우리가 만드는
# 정리된 컬럼 이름은 우리 마음대로 정하는 것이므로 올바른 철자(trainTestLabel_clean)를
# 쓰고, 원본에서 읽어올 때만 오타 그대로("trianTestLabel")를 참조한다.
df["failureType_clean"] = df["failureType"].apply(extract_label)
df["trainTestLabel_clean"] = df["trianTestLabel"].apply(extract_label)

print("[2] 라벨 정리 완료")
print(f"    failureType 고유값: {sorted(df['failureType_clean'].dropna().unique())}")
print(f"    trainTestLabel 고유값: {sorted(df['trainTestLabel_clean'].dropna().unique())}")

# %% [markdown]
# **여기서 헷갈리기 쉬운 지점**: `failureType_clean`이 문자열 `"none"`인 행과,
# 값이 아예 `None`(=NaN)인 행은 다른 의미다.
#
# - `"none"` 문자열 = 전문가가 **"불량 패턴 없음"이라고 라벨을 붙인** 정상 웨이퍼 (라벨된 데이터)
# - `None`/NaN = 애초에 **라벨 작업 자체를 안 한** 미라벨 데이터
#
# CLAUDE.md 4.2절의 "None 147,431장(85.2%)"은 전자, "미라벨 638,570장"은 후자다.
# 아래 집계에서 이 둘을 반드시 구분해서 본다.

# %%
# ------------------------------------------------------------------
# 문법 설명 4: value_counts(dropna=False)
# ------------------------------------------------------------------
# value_counts()는 기본적으로 결측치(NaN/None)를 세지 않고 무시한다.
# dropna=False를 주면 결측치도 하나의 항목처럼 개수를 세어 보여준다.
# 여기서는 '라벨 없음'이 몇 장인지도 같이 보고 싶어서 이 옵션을 켰다.
# ------------------------------------------------------------------
labeled_mask = df["failureType_clean"].notna()
n_labeled = labeled_mask.sum()
n_unlabeled = len(df) - n_labeled

print(f"[2] 라벨된 웨이퍼: {n_labeled:,}장 / 미라벨: {n_unlabeled:,}장 / 전체: {len(df):,}장")

# %% [markdown]
# ## 3. 클래스별 실제 장수 집계 (CLAUDE.md 4.2절 표와 대조)

# %%
class_counts = df.loc[labeled_mask, "failureType_clean"].value_counts()
class_ratio = (class_counts / n_labeled * 100).round(2)

class_summary = pd.DataFrame({"count": class_counts, "ratio(%)": class_ratio})
class_summary.index.name = "class"

print("[3] 클래스별 실제 장수 (라벨된 데이터 기준)")
print(class_summary)

class_summary.to_csv(OUTPUT_DIR / "class_distribution.csv", encoding="utf-8-sig")
print(f"    -> 저장됨: {OUTPUT_DIR / 'class_distribution.csv'}")

# %% [markdown]
# ## 4. 웨이퍼 맵 크기 분포 (Phase 1 최대 난관)
#
# `waferMap` 컬럼의 각 값은 2차원 numpy 배열(웨이퍼 맵 이미지)이고,
# `.shape`은 그 배열의 (세로, 가로) 크기를 튜플로 알려준다.
# 예: `(45, 48)` = 세로 45픽셀 × 가로 48픽셀짜리 웨이퍼 맵.

# %%
# ------------------------------------------------------------------
# 문법 설명 5: lambda와 .apply()
# ------------------------------------------------------------------
# lambda m: m.shape 는 "m을 받아서 m.shape를 돌려주는 이름 없는 함수"다.
# def로 함수를 따로 정의하기엔 너무 짧고 한 번만 쓸 때 lambda를 쓴다.
# df["col"].apply(함수)는 그 컬럼의 모든 행에 함수를 하나씩 적용한다.
# ------------------------------------------------------------------
df["map_shape"] = df["waferMap"].apply(lambda m: m.shape)

shape_counts = df["map_shape"].value_counts()
n_unique_shapes = len(shape_counts)

print(f"[4] 전체 고유 크기 종류: {n_unique_shapes:,}가지")

top_n = 20
top_shapes = shape_counts.head(top_n)
top_shapes_ratio = top_shapes.sum() / len(df) * 100

print(f"    상위 {top_n}개 크기가 전체의 {top_shapes_ratio:.1f}%를 차지")
print(top_shapes)

shape_summary = shape_counts.reset_index()
shape_summary.columns = ["shape(H,W)", "count"]
shape_summary["ratio(%)"] = (shape_summary["count"] / len(df) * 100).round(3)
shape_summary.to_csv(OUTPUT_DIR / "shape_distribution.csv", index=False, encoding="utf-8-sig")
print(f"    -> 저장됨: {OUTPUT_DIR / 'shape_distribution.csv'}")

# 상위 20개 크기를 막대그래프로 저장 (한눈에 보기 위함)
plt.figure(figsize=(10, 5))
labels = [str(s) for s in top_shapes.index]
plt.bar(labels, top_shapes.values)
plt.xticks(rotation=60, ha="right")
plt.ylabel("웨이퍼 맵 개수")
plt.title(f"상위 {top_n}개 웨이퍼 맵 크기 분포")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shape_distribution_top20.png", dpi=120)
plt.close()
print(f"    -> 그래프 저장됨: {OUTPUT_DIR / 'shape_distribution_top20.png'}")

# %% [markdown]
# ## 5. Lot 정보 확인 (Phase 6 관리도 가능 여부)

# %%
if "lotName" in df.columns:
    n_lots = df["lotName"].nunique()
    lot_sizes = df.groupby("lotName").size()

    print(f"[5] 고유 Lot 개수: {n_lots:,}")
    print("    Lot당 웨이퍼 장수 통계:")
    print(lot_sizes.describe())

    if "waferIndex" in df.columns:
        print("    waferIndex 예시 (Lot 내 웨이퍼 순번으로 추정):")
        print(df[["lotName", "waferIndex"]].head(10))
else:
    print("[5] lotName 컬럼이 없습니다 — 실제 컬럼명을 확인해 코드를 수정해야 합니다.")

# %% [markdown]
# ## 6. 픽셀 값 분포 확인 (0/1/2 외의 값이 있는지)
#
# 전체 81만 장을 순회하며 각 웨이퍼 맵에 어떤 픽셀 값이 등장하는지 모은다.
# 데이터가 많아 몇 분 걸릴 수 있어 10만 장마다 진행 상황을 출력한다.

# %%
# ------------------------------------------------------------------
# 문법 설명 6: set과 |= 연산자
# ------------------------------------------------------------------
# set()은 중복을 허용하지 않는 자료형이다. {0, 1, 2}에 이미 0이 있는 채로
# 0을 또 넣어도 여전히 {0, 1, 2}로 유지된다 — "지금까지 등장한 값의 종류"를
# 누적하기에 적합하다. a |= b는 a = a | b(합집합)의 축약형이다.
# ------------------------------------------------------------------
all_pixel_values = set()
for i, wafer_map in enumerate(df["waferMap"]):
    all_pixel_values |= set(np.unique(wafer_map))
    if (i + 1) % 100_000 == 0:
        print(f"    ...{i + 1:,} / {len(df):,} 처리 중")

print(f"[6] 데이터 전체에서 등장한 픽셀 값: {sorted(all_pixel_values)}")
expected = {0, 1, 2}
unexpected = all_pixel_values - expected
if unexpected:
    print(f"    주의: 예상 밖 픽셀 값 발견 -> {unexpected}")
else:
    print("    확인: 0(다이없음)/1(정상)/2(불량) 세 값만 존재함")

# %% [markdown]
# ## 7. 요약 리포트 저장
#
# 지금까지 확인한 내용을 텍스트 파일 하나로 정리해 저장한다.
# 다음 세션에서 CLAUDE.md 4절을 갱신할 때 이 파일을 참고하면 된다.

# %%
summary_lines = [
    "=== Phase 1 데이터 실태 조사 요약 ===",
    f"데이터 파일: {DATA_PATH.name}",
    f"전체 웨이퍼 맵: {len(df):,}장",
    f"라벨된 웨이퍼: {n_labeled:,}장 ({n_labeled / len(df) * 100:.1f}%)",
    f"미라벨 웨이퍼: {n_unlabeled:,}장 ({n_unlabeled / len(df) * 100:.1f}%)",
    "",
    "--- 클래스별 분포(라벨된 데이터 기준) ---",
    class_summary.to_string(),
    "",
    f"--- 웨이퍼 맵 크기 ---",
    f"고유 크기 종류: {n_unique_shapes:,}가지",
    f"상위 {top_n}개 크기가 전체의 {top_shapes_ratio:.1f}%를 차지",
    "",
    f"--- Lot 정보 ---",
    f"고유 Lot 개수: {n_lots if 'lotName' in df.columns else '컬럼 없음'}",
    "",
    f"--- 픽셀 값 ---",
    f"등장한 값: {sorted(all_pixel_values)}",
]

summary_text = "\n".join(summary_lines)
with open(OUTPUT_DIR / "phase1_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

print("\n" + summary_text)
print(f"\n[7] 전체 요약 저장됨: {OUTPUT_DIR / 'phase1_summary.txt'}")
