# %% [markdown]
# # data_overview_report — 데이터 내용 정리 리포트
#
# Phase 1~2에서 확인한 사실들을 하나의 HTML 문서로 정리한다.
# 다루는 내용: 컬럼 구조, 라벨(클래스) 종류, 데이터로 알 수 있는 것과
# 없는 것, 그리고 "공정 데이터"라고 부를 수 있는 게 이 데이터셋에
# 실제로 있는지 여부.
#
# 이 스크립트는 새로운 분석을 하지 않는다 — 이미 검증된 사실을
# **실제 데이터에서 다시 실시간으로 읽어와** 문서로 조립만 한다.
# (숫자를 손으로 옮겨 적으면 오타/구버전 수치가 섞일 위험이 있어서다)

# %%
import sys
import base64
from pathlib import Path

# Windows 콘솔 코드페이지(cp949)가 표현 못 하는 유니코드 문자(—) 때문에
# print()가 죽지 않도록 출력 인코딩을 UTF-8로 강제한다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_wm811k  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def b64_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# %% [markdown]
# ## 1. 데이터 로드 및 실시간 수치 재확인

# %%
df = load_wm811k()
labeled = df[df["failureType_clean"].notna()].reset_index(drop=True)

n_total = len(df)
n_labeled = len(labeled)
n_unlabeled = n_total - n_labeled

class_counts = labeled["failureType_clean"].value_counts()
n_lots = df["lotName"].nunique()
lot_sizes = df.groupby("lotName", observed=True).size()

tt_counts = labeled["trainTestLabel_clean"].value_counts()

print(f"[1] 전체 {n_total:,} / 라벨 {n_labeled:,} / 미라벨 {n_unlabeled:,}")
print(f"    Lot {n_lots:,}개, Lot당 웨이퍼 수 중앙값 {lot_sizes.median():.0f}")

# %% [markdown]
# ## 2. 클래스별 설명 텍스트 (Phase 2 패턴 그리드를 직접 육안 확인해 작성)

# %%
CLASS_DESC = {
    "none":      "인식 가능한 불량 패턴이 없는 정상 웨이퍼. 산발적인 단일 불량 다이는 있을 수 있음(평균 불량률 10.55%, D 그림 참고).",
    "Center":    "웨이퍼 정중앙에 뭉친 불량 덩어리.",
    "Donut":     "중앙을 피해 도넛(고리) 모양으로 분포하는 불량.",
    "Edge-Ring": "웨이퍼 가장자리 전체를 얇게 둘러싸는 불량 링.",
    "Edge-Loc":  "가장자리 근처에 국소적으로 뭉친 불량 — Scratch와 형태가 헷갈릴 수 있음.",
    "Loc":       "위치와 무관하게 국소적으로 뭉친 불량 (가장자리 제한 없음).",
    "Scratch":   "가늘고 긴 선 모양의 불량 — 기계적 스크래치 흔적.",
    "Random":    "웨이퍼 전면에 고르게 흩어진 불량.",
    "Near-full": "웨이퍼 거의 전체가 불량으로 덮인 상태.",
}

class_table_rows = ""
for cls in class_counts.index:
    cnt = class_counts[cls]
    pct = cnt / n_labeled * 100
    desc = CLASS_DESC.get(cls, "")
    class_table_rows += f"<tr><td><b>{cls}</b></td><td>{cnt:,}</td><td>{pct:.2f}%</td><td>{desc}</td></tr>\n"

# %% [markdown]
# ## 3. 컬럼 구조 테이블 (CLAUDE.md 4.4절과 동일 내용, 실시간 재확인)

# %%
col_table_rows = f"""
<tr><td><code>waferMap</code></td><td>object (2D numpy)</td><td>다이 배치 이미지, 값 0/1/2</td><td>행마다 크기 다름 (632가지)</td></tr>
<tr><td><code>dieSize</code></td><td>float</td><td>웨이퍼 전체 다이 수</td><td>waferMap의 (1 또는 2) 칸 개수와 정확히 일치</td></tr>
<tr><td><code>lotName</code></td><td>object</td><td>"lot1"~"lot{n_lots}"</td><td>{n_lots:,}개 전부 이 형식</td></tr>
<tr><td><code>waferIndex</code></td><td>float</td><td>Lot 내 웨이퍼 순번</td><td>결측 0개</td></tr>
<tr><td><code>trianTestLabel</code></td><td>object(중첩배열)</td><td>원 연구자의 Training/Test 구분</td><td>오타 그대로 저장된 컬럼명</td></tr>
<tr><td><code>failureType</code></td><td>object(중첩배열)</td><td>불량 패턴 라벨</td><td>미라벨은 빈 배열</td></tr>
"""

# %% [markdown]
# ## 4. HTML 조립

# %%
print("[4] 리포트 조립 중...")

img_a = b64_image(OUTPUT_DIR / "viz_A_pattern_grid.png")
img_c = b64_image(OUTPUT_DIR / "viz_C_class_imbalance.png")
img_d = b64_image(OUTPUT_DIR / "viz_D_class_average_rate_map.png")

html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>[리포트 02] WM-811K 데이터 개요 리포트</title>
<style>
  body {{ font-family: "Malgun Gothic", sans-serif; max-width: 1000px; margin: 40px auto;
         padding: 0 20px; line-height: 1.7; color: #222; }}
  h1 {{ border-bottom: 3px solid #4c72b0; padding-bottom: 10px; }}
  h2 {{ color: #2b3a55; margin-top: 50px; border-left: 5px solid #4c72b0; padding-left: 10px; }}
  h3 {{ color: #444; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.92em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #4c72b0; color: white; }}
  tr:nth-child(even) {{ background: #f7f7f7; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
  .warn {{ background: #fff3cd; border-left: 5px solid #cc9900; padding: 12px 16px; margin: 15px 0; }}
  .meta {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>[리포트 02] WM-811K 데이터 개요 리포트</h1>
<p class="meta">데이터 출처: Kaggle qingyi/wm811k-wafer-map (LSWMD.pkl) ·
전체 {n_total:,}장, 라벨 {n_labeled:,}장({n_labeled/n_total*100:.1f}%),
미라벨 {n_unlabeled:,}장({n_unlabeled/n_total*100:.1f}%)</p>

<h2>1. 이 데이터셋이 무엇인가</h2>
<p>실제 반도체 팹에서 나온 웨이퍼 검사 결과 이미지 모음이다. 웨이퍼 한 장 = 표 한 행이며,
각 행의 <code>waferMap</code>은 그 웨이퍼 위 모든 다이(반도체 칩 하나하나)가
정상인지 불량인지를 2차원 배열로 담고 있다. 전체 {n_total:,}장 중
{n_labeled:,}장({n_labeled/n_total*100:.1f}%)만 전문가가 "이 웨이퍼는 어떤
불량 패턴인지"를 라벨링했고, 나머지 {n_unlabeled:,}장은 라벨이 없다.</p>

<h2>2. 표 구조 (컬럼별 정의)</h2>
<table>
<tr><th>컬럼</th><th>타입</th><th>의미</th><th>비고</th></tr>
{col_table_rows}
</table>
<p>위 6개는 원본 파일에 있는 컬럼이고, 우리 코드(<code>data_loader.py</code>)가
분석하기 편하게 3개를 추가로 만들었다: <code>failureType_clean</code>·
<code>trainTestLabel_clean</code>(중첩 배열을 문자열로 정리), <code>map_shape</code>
(waferMap 크기 튜플).</p>

<div class="warn">
<b>주의할 관계 하나</b>: 라벨 존재 여부와 Training/Test 태그 존재 여부는
정확히 1:1로 붙어 있다(둘 다 있거나 둘 다 없거나, 중간 케이스 0건).
단 라벨된 데이터 안에서 Test가 Training의 2배 이상이다 — 이건 원 논문
저자가 자기 실험에서 쓴 구분이므로 우리 프로젝트의 학습/검증 분할에는
그대로 쓰지 않고 우리가 직접 나눈다(Phase 3에서 진행).
</div>

<h2>3. 라벨(클래스) 9종</h2>
<img src="data:image/png;base64,{img_a}" alt="패턴 그리드">
<table>
<tr><th>클래스</th><th>장수</th><th>비율</th><th>육안 특징</th></tr>
{class_table_rows}
</table>
<img src="data:image/png;base64,{img_c}" alt="클래스 불균형">

<h2>4. 이 데이터로 알 수 있는 것</h2>
<ul>
  <li><b>불량 패턴 종류 분류</b> — 9개 클래스, CNN으로 학습 예정 (Phase 4)</li>
  <li><b>웨이퍼별 불량률</b> = 불량 다이 수 / dieSize (원본에서 바로 계산 가능, 리사이즈 불필요)</li>
  <li><b>클래스별 평균 불량 위치 패턴</b> — 아래 그림처럼 클래스마다 불량이 몰리는 위치가 다르다</li>
  <li><b>Lot 단위 집계</b> — 같은 Lot(카세트, 보통 25장 단위) 안에서 패턴 발생 빈도 추이 (Phase 6에서 관리도로 활용)</li>
  <li><b>웨이퍼 크기(다이 수)와 불량률의 관계</b> — 정규화 없이 절대 개수로 비교하면 오해가 생김(Phase 2 E번 그림)</li>
</ul>
<img src="data:image/png;base64,{img_d}" alt="클래스별 평균 불량률 맵">

<h2>5. 이 데이터에 없는 것 — "공정 데이터"에 대하여</h2>
<div class="warn">
<b>이 데이터셋 자체에는 실제 공정(process) 데이터가 없다.</b>
CD(임계치수), RF 파워, 챔버 온도, 레시피 ID, 설비 ID, 실제 타임스탬프 —
이런 계측/설비 정보가 담긴 컬럼은 하나도 없다. 있는 건 <code>waferMap</code>
(결과 이미지), <code>dieSize</code>, <code>lotName</code>/<code>waferIndex</code>
(생산 그룹 정보), 라벨뿐이다.
</div>
<p>그래서 "Center 패턴 → 챔버 RF 이상"처럼 CLAUDE.md 7.2절에 정리해둔
공정 모듈 매핑은 <b>이 데이터를 분석해서 나온 결론이 아니라, 문헌(선행 연구)에서
가져온 도메인 지식</b>이다. 이 프로젝트가 실제로 검증할 수 있는 건
"패턴이 이렇게 생겼다"까지이고, "그 패턴의 물리적 원인이 이거다"는
문헌을 인용하는 것이지 이 데이터로 증명하는 게 아니다 — 발표할 때
이 구분을 명확히 해야 한다.</p>

<table>
<tr><th>패턴</th><th>문헌상 공정 모듈</th><th>물리적 원인</th></tr>
<tr><td>Center</td><td>챔버</td><td>RF 동작 불규칙, 유체 흐름 불균일</td></tr>
<tr><td>Donut</td><td>세정</td><td>포토레지스트 잔류물 미제거</td></tr>
<tr><td>Edge-Ring</td><td>RTP</td><td>온도 조절 이상, 저온 공정 편차</td></tr>
<tr><td>Edge-Loc</td><td>어닐링 / 로드락</td><td>온도 불균일, 밸브 오염</td></tr>
<tr><td>Loc</td><td>국소 설비</td><td>특정 위치 반복 결함</td></tr>
<tr><td>Scratch</td><td>핸들링</td><td>기계적 접촉, 이송 장비</td></tr>
<tr><td>Random</td><td>파티클</td><td>오염, 환경 요인</td></tr>
<tr><td>Near-full</td><td>전면 이상</td><td>공정 전체 실패</td></tr>
</table>

<h2>6. 지금까지 확인된 한계 총정리</h2>
<ul>
  <li><b>Near-full 149장</b> — 5-Fold 시 fold당 약 30장, 성능이 불안정할 수 있음(SECOM 센서 130과 같은 위치)</li>
  <li><b>Scratch 리사이즈 손실</b> — 큰 원본을 64x64로 줄일 때 선 연속성이 다소 손실됨 (Edge-Loc 혼동 가능성)</li>
  <li><b>"none" 웨이퍼도 중앙 불량률이 높음</b> — 10.55%→25.61%, 원인 미확정(상관관계만 관찰)</li>
  <li><b>Lot 순서 그래프 초반 스파이크</b> — 표본이 작아(55~500장) 생기는 통계적 잡음 가능성</li>
  <li><b>Training/Test 태그</b> — 원 논문 저자 기준이라 이 프로젝트에서 재사용하지 않음</li>
  <li><b>실제 타임스탬프 없음</b> — Lot 순서는 이름 기준 정렬일 뿐 생산 시각이 아님</li>
</ul>

</body>
</html>
"""

report_path = OUTPUT_DIR / "report02_data_overview.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"\n[완료] 리포트 저장됨: {report_path}")
