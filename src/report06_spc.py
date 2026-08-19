# %% [markdown]
# # report06_spc — [리포트 06] Phase 6 SPC 관리도 결과
#
# 무거운 연산 없이, 이미 저장된 Phase 6 결과물을 읽어와 HTML로 정리한다.
# 패턴별 수율 손실 기여도에 CLAUDE.md 7.2절의 공정 매핑(문헌 기반)을
# 결합해 "어느 공정 모듈이 수율 손실에 가장 크게 기여하는가"를 보여준다
# (PLAN.md 6.2절).

# %%
import sys
import base64
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def read_text(name):
    return (OUTPUT_DIR / name).read_text(encoding="utf-8")


def b64_image(name):
    with open(OUTPUT_DIR / name, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# CLAUDE.md 7.2절 — 문헌 기반 공정 매핑 (이 데이터로 검증된 것 아님, 반드시 명시)
PROCESS_MAP = {
    "Center": ("챔버", "RF 동작 불규칙, 유체 흐름 불균일"),
    "Donut": ("세정", "포토레지스트 잔류물 미제거"),
    "Edge-Ring": ("RTP", "온도 조절 이상, 저온 공정 편차"),
    "Edge-Loc": ("어닐링 / 로드락", "온도 불균일, 밸브 오염"),
    "Loc": ("국소 설비", "특정 위치 반복 결함"),
    "Scratch": ("핸들링", "기계적 접촉, 이송 장비"),
    "Random": ("파티클", "오염, 환경 요인"),
    "Near-full": ("전면 이상", "공정 전체 실패"),
}

summary = read_text("phase6_spc_summary.txt")
contribution_df = pd.read_csv(OUTPUT_DIR / "phase6_yield_contribution.csv")
img_chart = b64_image("phase6_spc_control_chart.png")
img_sizes = b64_image("phase6_group_sizes.png")

contrib_rows = ""
for _, row in contribution_df.iterrows():
    module, cause = PROCESS_MAP.get(row["패턴"], ("-", "-"))
    contrib_rows += (
        f"<tr><td>{int(row['기여도 순위'])}</td><td><b>{row['패턴']}</b></td>"
        f"<td>{row['발생률(%)']:.3f}%</td><td>{row['평균 불량다이비율(%)']:.2f}%</td>"
        f"<td>{row['수율손실 기여도']:.5f}</td><td>{module}</td><td>{cause}</td></tr>\n"
    )

html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>[리포트 06] Phase 6 SPC 관리도 결과</title>
<style>
  body {{ font-family: "Malgun Gothic", sans-serif; max-width: 1100px; margin: 40px auto;
         padding: 0 20px; line-height: 1.7; color: #222; }}
  h1 {{ border-bottom: 3px solid #4c72b0; padding-bottom: 10px; }}
  h2 {{ color: #2b3a55; margin-top: 45px; border-left: 5px solid #4c72b0; padding-left: 10px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 10px; text-align: left; }}
  th {{ background: #4c72b0; color: white; }}
  tr:nth-child(even) {{ background: #f7f7f7; }}
  pre {{ background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px; padding: 12px 16px;
         overflow-x: auto; font-size: 0.85em; white-space: pre-wrap; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
  .warn {{ background: #fff3cd; border-left: 5px solid #cc9900; padding: 12px 16px; margin: 15px 0; }}
  .good {{ background: #d4edda; border-left: 5px solid #28a745; padding: 12px 16px; margin: 15px 0; }}
  .navlist {{ background: #f0f4f8; padding: 12px 16px; border-radius: 6px; }}
  .meta {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>[리포트 06] Phase 6 — Lot 단위 SPC 관리도</h1>
<p class="meta">SECOM p-관리도 로직 재사용 (`1. 반도체_수율_프로젝트/main2.py`) · "Test" 라벨만 사용(아래 설명)</p>

<div class="navlist">
<b>리포트 목록</b><br>
01 시각화 · 02 데이터 개요 · 03 진행 스냅샷(P1~3) · 04 진행 스냅샷(P4) ·
05 Grad-CAM · <b>06 SPC 관리도(현재 문서)</b>
</div>

<h2>중요 발견 — 원 논문 Training/Test 분할이 Lot 순서 분석을 왜곡시킴</h2>
<div class="good">
관리도를 처음 그렸을 때, 거의 모든 불량 패턴이 Lot 순서 초반 구간에서
동시에 극단적으로 치솟는 현상이 나타났다. 원인을 파고든 결과, <b>그 구간
웨이퍼의 100%가 원 연구자의 "Training" 라벨</b>이었다 — 원 연구자가 자기
모델 학습용으로 다양한 불량 유형을 일부러 골고루 뽑아 만든 세트가 Lot
순서상 우연히 뭉쳐 있어 생긴 착시였지, 실제 생산 이상이 아니었다.
CLAUDE.md엔 이미 "Training/Test 태그를 재사용하지 않는다"고 적혀 있었지만,
Lot 순서 분석에서 이렇게 극적으로 드러날 줄은 예상 못 했다. <b>발견 즉시
"Test" 라벨(자연스러운 생산 비율에 더 가까움, 118,595장)만 남기고
다시 계산</b>했다 — 이 리포트의 모든 수치는 Test 라벨 기준이다.
</div>

<h2>Lot 묶음 구성 방식</h2>
<p>전체 Lot 46,293개 중 라벨이 하나라도 있는 Lot은 10,762개(23.2%)뿐이라,
Lot 하나 단위로는 대부분 표본이 너무 적어 관리도 판단이 불가능하다.
그래서 Phase 2에서 썼던 "Lot 번호 범위로 균등 분할"(구간마다 표본
크기가 크게 들쭉날쭉했음) 대신, <b>"라벨된 웨이퍼 수가 1,000장에 도달할
때까지 연속된 Lot을 묶는" 방식</b>을 썼다 — 구간마다 표본 크기가 고르게
맞춰져 관리한계선(UCL) 폭을 공정하게 비교할 수 있다.</p>
<img src="data:image/png;base64,{img_sizes}" alt="구간별 표본 크기">

<h2>패턴별 p-관리도</h2>
<img src="data:image/png;base64,{img_chart}" alt="SPC 관리도">
<pre>{html_escape(summary)}</pre>

<div class="warn">
<b>참고로 남겨둘 관찰</b>: Edge-Ring·Near-full이 그룹#54에서, Center·Random이
그룹#76에서 동시에 관리한계를 초과했다. 이게 실제로 여러 불량 유형을
한꺼번에 유발한 공정 이상(예: 특정 시기의 복합적 설비 문제)인지, 아니면
또 다른 데이터 구조상의 우연인지는 이 프로젝트 범위에서 확정할 수 없다 —
후속 분석 대상으로 남겨둔다.
</div>

<h2>수율 손실 기여도 (PLAN.md 6.2절) — 문헌 기반 공정 매핑 결합</h2>
<p>수율 손실 기여도 = 발생률 × 평균 불량 다이 비율. 공정 모듈 열은
<b>문헌 기반 도메인 지식이며 이 데이터로 검증된 결론이 아니다</b>
(CLAUDE.md 7.2절 · 리포트 02 참고).</p>
<table>
<tr><th>순위</th><th>패턴</th><th>발생률</th><th>평균 불량다이비율</th><th>기여도</th><th>문헌상 공정 모듈</th><th>물리적 원인</th></tr>
{contrib_rows}
</table>
<p><b>해석</b>: Edge-Loc과 Loc이 발생률(2.3%, 1.7%)과 다이당 불량 비율(16% 안팎)이
둘 다 상당해서 수율 손실 기여도 1·2위를 차지했다 — 문헌상 어닐링/로드락과
국소 설비 쪽 점검이 우선순위가 될 수 있다는 뜻이다(단, 위 단서대로 문헌
기반 매핑일 뿐 이 데이터로 원인을 증명한 것은 아니다). Near-full은
발생률은 가장 낮지만(0.08%) 다이당 불량 비율이 압도적으로 높아(87.6%)
장 발생 시 피해가 매우 크다는 점도 같이 봐야 한다.</p>

</body>
</html>
"""

report_path = OUTPUT_DIR / "report06_spc.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"[완료] 저장됨: {report_path}")
