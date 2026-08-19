# %% [markdown]
# # progress_report — 프로젝트 진행 상황 종합 보고서
#
# Phase 1~4(진행 중)까지 지금까지 한 일을 하나의 HTML로 정리한다.
# 이미 저장된 요약 파일들(phase1_summary.txt 등)을 그대로 읽어와 인용하고,
# 손으로 옮겨 적지 않는다 — 옮겨 적으면 오타/구버전 수치가 섞일 위험이 있다.
#
# **주의**: 이 시점엔 Phase 4(CNN 학습)가 아직 진행 중이다. 최종 성능
# 수치는 없으므로 이 보고서에 넣지 않는다 — 학습이 끝나면 별도로 보고한다.

# %%
import sys
import base64
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def read_text(name: str) -> str:
    return (OUTPUT_DIR / name).read_text(encoding="utf-8")


def b64_image(name: str) -> str:
    with open(OUTPUT_DIR / name, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# %% [markdown]
# ## 1. 저장된 요약 파일들 로드 (실제 산출물을 그대로 인용)

# %%
phase1_summary = read_text("phase1_summary.txt")
resize_summary = read_text("resize_check_summary.txt")
phase3_summary = read_text("phase3_preprocessing_summary.txt")
input_rep_summary = read_text("phase4_input_representation_check.txt")

n_fold_models_done = len(list(ARTIFACTS_DIR.glob("model_fold*.pt")))
print(f"[1] 지금까지 저장된 fold 모델: {n_fold_models_done}/5")

img_pattern_grid = b64_image("viz_A_pattern_grid.png")
img_size_dist = b64_image("viz_B_size_distribution.png")
img_class_imbalance = b64_image("viz_C_class_imbalance.png")
img_rate_map = b64_image("viz_D_class_average_rate_map.png")
img_defect_vs_diesize = b64_image("viz_E_defect_rate_vs_diesize.png")
img_lot_trend = b64_image("viz_F_lot_order_trend.png")

# %% [markdown]
# ## 2. HTML 조립

# %%
print("[2] 리포트 조립 중...")

html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>[리포트 03] 진행 상황 스냅샷 (Phase 1~3)</title>
<style>
  body {{ font-family: "Malgun Gothic", sans-serif; max-width: 1000px; margin: 40px auto;
         padding: 0 20px; line-height: 1.7; color: #222; }}
  h1 {{ border-bottom: 3px solid #4c72b0; padding-bottom: 10px; }}
  h2 {{ color: #2b3a55; margin-top: 55px; border-left: 5px solid #4c72b0; padding-left: 10px; }}
  h3 {{ color: #444; margin-top: 25px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.92em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #4c72b0; color: white; }}
  tr:nth-child(even) {{ background: #f7f7f7; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
  pre {{ background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px; padding: 12px 16px;
         overflow-x: auto; font-size: 0.85em; white-space: pre-wrap; }}
  .warn {{ background: #fff3cd; border-left: 5px solid #cc9900; padding: 12px 16px; margin: 15px 0; }}
  .status-progress {{ background: #d1ecf1; border-left: 5px solid #0c5460; padding: 12px 16px; margin: 15px 0; }}
  .status-done {{ background: #d4edda; border-left: 5px solid #28a745; padding: 4px 10px;
                  border-radius: 3px; font-size: 0.85em; display: inline-block; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  .finding {{ margin: 8px 0; }}
</style>
</head>
<body>

<h1>[리포트 03] 웨이퍼 맵 불량 패턴 분류 프로젝트 — 진행 상황 스냅샷</h1>
<p class="meta">WM-811K 데이터셋 (Kaggle qingyi/wm811k-wafer-map) · SECOM 후속 프로젝트</p>

<div class="status-progress">
<b>현재 상태</b>: Phase 1~3 완료, <b>Phase 4(CNN 학습)는 현재 진행 중</b>
(5-Fold 중 {n_fold_models_done}/5 fold 완료 확인). 이 보고서는 Phase 4의
최종 성능 수치를 포함하지 않는다 — 학습이 끝나면 별도로 보고한다.
</div>

<h2>핵심 발견 요약</h2>
<ul>
  <li class="finding"><b>클래스 분포가 문헌 수치와 완전히 일치</b> — 데이터 무결성 확인 (Phase 1)</li>
  <li class="finding"><b>웨이퍼 맵 크기 632가지, 지배적 크기 없음</b> (상위 20개 합쳐도 57.5%) → 전체 64×64 리사이즈로 결정 (Phase 1)</li>
  <li class="finding"><b>Scratch 클래스는 다운샘플링 시 선 연속성이 다소 손실</b>됨 — Edge-Loc 혼동에 기여 가능성 (Phase 1)</li>
  <li class="finding"><b>"none"(정상) 웨이퍼도 정중앙 불량률이 2배 이상 높음</b>(10.55%→25.61%) — 원인 미확정, 상관관계만 관찰 (Phase 2)</li>
  <li class="finding"><b>Lot 순서 그래프 초반 스파이크는 표본이 작아(55~500장) 생기는 통계적 잡음일 가능성</b> — 후반은 표본이 1만 장 이상으로 훨씬 안정적 (Phase 2)</li>
  <li class="finding"><b>이 데이터셋엔 실제 공정(CD/RF파워/챔버온도/설비ID) 데이터가 없음</b> — 공정 모듈 매핑은 문헌 기반 도메인 지식이지 데이터로 검증된 결론이 아님</li>
  <li class="finding"><b>증강 누수 차단 구조를 실제로 검증</b> — 5개 fold 전부 검증셋 불변, 학습 fold 소수 클래스만 정확히 8배 증강 확인 (Phase 3)</li>
  <li class="finding"><b>로컬 GPU는 Intel Arc(비CUDA)</b> — 벤치마크 결과 로컬 CPU로도 폴드당 15~35분 수준이라 Colab 없이 로컬 진행 결정 (Phase 4)</li>
  <li class="finding"><b>입력 표현 비교에서 원리적 예상(onehot)과 반대로 raw가 우세</b>(macro-F1 0.579 vs 0.380, 5epoch 비교) → raw 채택 (Phase 4)</li>
</ul>

<h2>Phase 1 — 데이터 실태 조사 <span class="status-done">완료</span></h2>
<p>Kaggle에서 받은 <code>LSWMD.pkl</code>은 2018년경 Python2+pandas 0.2x대로 만들어져
호환성 문제 3종(옛 pandas 모듈 경로, Python2 문자열 인코딩, <code>trianTestLabel</code> 오타)을
해결해야 로드됐다. 이후 클래스 분포·크기 분포·Lot 정보·픽셀 값을 실측했다.</p>
<pre>{html_escape(phase1_summary)}</pre>

<h3>리사이즈 방식 검증</h3>
<p>범주형 데이터(0=없음/1=정상/2=불량)이므로 보간이 섞이지 않는
<b>최근접 이웃(nearest neighbor)</b> 방식만 사용, 실제 데이터로 검증했다.</p>
<pre>{html_escape(resize_summary)}</pre>
<img src="data:image/png;base64,{b64_image('resize_before_after.png')}" alt="리사이즈 전후 비교">

<h2>Phase 2 — 시각화 <span class="status-done">완료</span></h2>
<p>패턴 그리드, 크기 분포, 클래스 불균형, 클래스별 평균 불량률 맵,
불량률-dieSize 산점도, Lot 순서별 발생 비율 6종을 만들고
<code>outputs/02_visualization_report.html</code>로 통합했다. 아래는 요약.</p>

<img src="data:image/png;base64,{img_pattern_grid}" alt="패턴 그리드">
<img src="data:image/png;base64,{img_size_dist}" alt="크기 분포">
<img src="data:image/png;base64,{img_class_imbalance}" alt="클래스 불균형">
<img src="data:image/png;base64,{img_rate_map}" alt="클래스별 평균 불량률 맵">
<img src="data:image/png;base64,{img_defect_vs_diesize}" alt="불량률 vs dieSize">
<img src="data:image/png;base64,{img_lot_trend}" alt="Lot 순서별 발생 비율">

<h2>데이터 개요 (컬럼/라벨/공정 데이터 유무)</h2>
<p>표 구조, 라벨 9종의 의미, "데이터로 알 수 있는 것/없는 것", 그리고
<b>이 데이터셋에 실제 공정 데이터가 없다는 점</b>을 별도 문서로 정리했다.
전체 내용은 <code>outputs/data_overview_report.html</code> 참고.</p>
<div class="warn">
가장 중요한 요지만 다시 적으면: CLAUDE.md의 "패턴→공정 모듈" 매핑표는
<b>이 데이터를 분석해서 나온 결론이 아니라 문헌에서 가져온 도메인 지식</b>이다.
발표 시 "이 데이터가 증명한 것"과 "문헌을 인용한 것"을 구분해야 한다.
</div>

<h2>Phase 3 — 데이터 준비 <span class="status-done">완료</span></h2>
<p>다수 클래스(none/Edge-Ring/Edge-Loc/Center/Loc)는 2,000장으로 축소,
소수 클래스(Scratch/Random/Donut/Near-full)는 전량 사용해 12,763장을
구성하고 64×64로 리사이즈해 저장했다. 이어서 5-Fold 분할 후 <b>증강이
학습 fold 안에서만 적용되고 검증 fold는 원본 그대로 남는지</b>를
코드로 직접 검증했다.</p>
<pre>{html_escape(phase3_summary)}</pre>

<h2>Phase 4 — CNN 모델 학습 <span class="status-progress">진행 중</span></h2>
<p>PyTorch(CPU 버전) 설치 완료. 로컬 GPU가 Intel Arc(비CUDA)임을 확인하고
실측 벤치마크(배치당 121ms, 12,763장 기준 1epoch 약 24초) 결과를 근거로
Colab 대신 로컬 CPU로 진행하기로 결정했다.</p>

<h3>입력 표현 비교 (raw vs onehot)</h3>
<p>0/1/2를 그대로 1채널로 넣는 방식(raw)과 3채널 원-핫으로 분리하는
방식(onehot)을 2-Fold×5epoch로 빠르게 비교했다. 범주형 데이터라 onehot이
원리상 더 타당해 보였으나, 실측 결과 raw가 뚜렷하게 우세해 raw를 채택했다
(onehot은 입력 채널이 3배라 짧은 epoch에서 수렴이 느렸을 가능성 — 근본적
열등함이 아닐 수 있다는 단서를 남겨둔다).</p>
<pre>{html_escape(input_rep_summary)}</pre>

<h3>본 학습 — 진행 상황</h3>
<p>5-Fold Stratified 교차검증, fold당 최대 40epoch(macro-F1 기준 조기 종료,
patience=6), 학습 fold의 소수 클래스만 8배 증강, 클래스 가중치 적용
CrossEntropyLoss 사용. 현재 <b>{n_fold_models_done}/5 fold 완료</b>.
최종 성능(OOF 기준 macro-F1·balanced accuracy·9×9 혼동 행렬,
특히 Edge-Loc/Scratch 혼동률)은 학습이 끝난 뒤 별도로 보고한다.</p>

<h2>지금까지 확인된 한계 총정리</h2>
<table>
<tr><th>한계</th><th>내용</th></tr>
<tr><td>Near-full 149장</td><td>5-Fold 시 fold당 검증 29~30장, 3장 차이로 Recall이 크게 흔들릴 수 있음(실제 확인됨)</td></tr>
<tr><td>Scratch 리사이즈 손실</td><td>큰 원본을 64×64로 줄일 때 선 연속성 손실 — Edge-Loc 혼동 가능성</td></tr>
<tr><td>"none" 중앙 불량률</td><td>10.55%→25.61%, 원인 미확정(상관관계만 관찰)</td></tr>
<tr><td>Lot 순서 그래프 초반 스파이크</td><td>표본이 작아(55~500장) 생기는 통계적 잡음 가능성</td></tr>
<tr><td>Training/Test 태그</td><td>원 논문 저자 기준이라 이 프로젝트에서 재사용하지 않음</td></tr>
<tr><td>실제 타임스탬프 없음</td><td>Lot 순서는 이름 기준 정렬일 뿐 생산 시각 아님</td></tr>
<tr><td>공정 데이터 없음</td><td>패턴→공정 모듈 매핑은 문헌 기반, 데이터로 검증된 것 아님</td></tr>
<tr><td>입력 표현 비교의 한계</td><td>5epoch짜리 빠른 비교로 결정 — 전체 학습 규모에서 재확인 안 됨</td></tr>
</table>

<h2>다음 단계</h2>
<ol>
  <li>Phase 4 학습 완료 대기 → OOF 성능, 혼동 행렬 보고</li>
  <li>Phase 5 — Grad-CAM으로 예측 근거 시각화, 공정 원인 매핑과 연결</li>
  <li>Phase 6 — Lot 단위 SPC 관리도로 수율 이상 감지 (SECOM p-관리도 로직 재사용)</li>
  <li>Phase 7 — Streamlit 대시보드 통합</li>
</ol>

</body>
</html>
"""

report_path = OUTPUT_DIR / "report03_progress_snapshot_phase1to3.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_doc)

print(f"\n[완료] 리포트 저장됨: {report_path}")
