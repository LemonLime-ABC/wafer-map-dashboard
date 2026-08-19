# %% [markdown]
# # report05_gradcam — [리포트 05] Phase 5 Grad-CAM 결과
#
# 무거운 연산(모델 재학습 등) 없이, 이미 저장된 Grad-CAM 결과물을
# 읽어와 HTML로 정리만 한다.

# %%
import sys
import base64
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def read_text(name):
    return (OUTPUT_DIR / name).read_text(encoding="utf-8")


def b64_image(name):
    with open(OUTPUT_DIR / name, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


summary = read_text("phase5_gradcam_summary.txt")
img_grid = b64_image("phase5_gradcam_grid.png")

html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>[리포트 05] Phase 5 Grad-CAM 결과</title>
<style>
  body {{ font-family: "Malgun Gothic", sans-serif; max-width: 1050px; margin: 40px auto;
         padding: 0 20px; line-height: 1.7; color: #222; }}
  h1 {{ border-bottom: 3px solid #4c72b0; padding-bottom: 10px; }}
  h2 {{ color: #2b3a55; margin-top: 45px; border-left: 5px solid #4c72b0; padding-left: 10px; }}
  pre {{ background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px; padding: 12px 16px;
         overflow-x: auto; font-size: 0.88em; white-space: pre-wrap; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
  .warn {{ background: #fff3cd; border-left: 5px solid #cc9900; padding: 12px 16px; margin: 15px 0; }}
  .navlist {{ background: #f0f4f8; padding: 12px 16px; border-radius: 6px; }}
  .meta {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>[리포트 05] Phase 5 — Grad-CAM 예측 근거 시각화</h1>
<p class="meta">fold 0 모델(Flatten head) · fold 0 검증셋(학습에 안 쓰인 2,553장) 기준</p>

<div class="navlist">
<b>리포트 목록</b><br>
01 — Phase 2 시각화 · 02 — 데이터 개요 · 03 — 진행 스냅샷(Phase 1~3) ·
04 — 진행 스냅샷(Phase 4 개선 중) · <b>05 — Phase 5 Grad-CAM(현재 문서)</b>
</div>

<h2>클래스별 예시 (원본 / Grad-CAM 오버레이)</h2>
<img src="data:image/png;base64,{img_grid}" alt="Grad-CAM 그리드">
<p><b>육안으로 확인된 것</b>: Donut은 히트맵이 고리 모양을 거의 그대로 따라 그리고,
Scratch는 선 모양을 정확히 따라간다. Loc은 흥미롭게도 덩어리 자체가 아니라
그 <b>윤곽선(경계)</b>을 고리 모양으로 감싸는 경우가 있다 — 모델이 "정확한
위치"보다 "형태의 경계"를 판단 근거로 쓰는 것으로 보인다.</p>

<h2>정량 평가 (통제군 대비)</h2>
<pre>{html_escape(summary)}</pre>

<div class="warn">
<b>정량 지표의 한계</b>: "히트맵이 불량 칸 위에 정확히 겹치는 비율"로 쟀는데,
이 지표는 Loc처럼 "윤곽선"을 보는 전략을 과소평가한다 — 정당한 판단 근거인데도
칸이 정확히 안 겹치면 낮게 나온다. 전체 평균 배율(1.0배, 사실상 통제군과 비슷)만
보면 실망스럽지만, 그림으로 직접 확인한 결과 모델이 각 클래스에 맞는 의미 있는
구조(고리·선·윤곽)를 실제로 보고 있다는 정황이 있다. 숫자와 그림을 같이 봐야
정확한 판단이 가능하다는 사례로 남긴다.
</div>

<h2>클래스별 요약</h2>
<ul>
  <li><b>Scratch(1.6배), Loc(1.4배), Near-full(1.4배), Random(1.3배)</b>: 통제군보다 뚜렷이 높음 — 실제 불량 위치를 근거로 판단</li>
  <li><b>Edge-Loc(1.0배), none(1.1배)</b>: 통제군과 비슷한 수준</li>
  <li><b>Donut(0.9배), Edge-Ring(0.8배), Center(0.6배)</b>: 통제군보다 낮음 — 다만 Donut은 육안으로는 고리를 잘 따라감(지표 한계로 추정)</li>
</ul>

</body>
</html>
"""

report_path = OUTPUT_DIR / "report05_gradcam.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"[완료] 저장됨: {report_path}")
