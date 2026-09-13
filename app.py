# ============================================
# app.py — 웨이퍼 맵 불량 패턴 분류 대시보드
# ============================================
# 실행:  python -m streamlit run app.py
#
# SECOM 프로젝트(1. 반도체_수율_프로젝트/app.py)와 같은 원칙:
# 1. 무거운 계산은 여기서 하지 않는다. 전부 src/08_precompute_dashboard.py가
#    미리 만들어둔 artifacts/dashboard_bundle.joblib에서 읽어온다.
#    (Grad-CAM만 예외 — 선택한 웨이퍼 1장에 대해서만 실시간 계산, 1초 미만)
# 2. 성능 숫자는 하드코딩하지 않는다. 번들을 다시 만들면 화면이 자동 갱신된다.
# 3. 문헌 기반 지식과 데이터로 검증된 사실을 화면에서도 구분해서 표시한다.

import os
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "src"))
from model import WaferCNN  # noqa: E402
from gradcam import GradCAM  # noqa: E402
from live_inference import (  # noqa: E402
    RULE_WM, RULE_BIN, parse_wafer_file, input_warnings, map_summary,
    to_model_input, load_fold_models, predict_probs,
)
from resize_utils import resize_nearest  # noqa: E402

ART = os.path.join(BASE, "artifacts")

# page_icon은 브라우저 탭 아이콘이자, 폰에서 "홈 화면에 추가"했을 때의
# 홈 화면 아이콘이 된다. 지정하지 않으면 Streamlit 기본 아이콘이 떠서
# 홈 화면에서 우리 앱인지 구분되지 않는다.
# (src/10_make_app_icon.py가 실제 Edge-Ring 웨이퍼 맵으로 생성)
_ICON = os.path.join(BASE, "assets", "app_icon.png")
st.set_page_config(
    page_title="웨이퍼 맵 불량 패턴 분류",
    page_icon=_ICON if os.path.exists(_ICON) else "🔬",
    layout="wide",
)

# --------------------------------------------
# 데이터·모델 로드 (캐시)
# --------------------------------------------
@st.cache_resource
def load_all():
    bundle = joblib.load(os.path.join(ART, "dashboard_bundle.joblib"))
    model = WaferCNN(in_channels=1, n_classes=len(bundle["class_names"]), head="flatten")
    state = torch.load(os.path.join(ART, "model_v2_flatten_fold0.pt"),
                        map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    gradcam = GradCAM(model, model.features[10])
    return bundle, model, gradcam


bundle = load_all()[0]
model = load_all()[1]
gradcam = load_all()[2]

class_names = bundle["class_names"]
wafer_table = bundle["wafer_table"]
X_val0 = bundle["X_val0"]
# 9개 클래스 전체 확률. 예전 번들에는 이 키가 없으므로 .get()으로 안전하게 꺼낸다
# (없으면 None이 되고, 화면 1에서 해당 그래프만 건너뛴다).
probs_all = bundle.get("probs_all")
# Lot 응집도(화면 3). 예전 번들에는 없는 키이므로 .get()으로 안전하게 꺼내고,
# 없으면 해당 섹션만 건너뛴다.
lot_pattern_df = bundle.get("lot_pattern_df")
lot_summary = bundle.get("lot_summary")
lot_gallery = bundle.get("lot_gallery")
cm = bundle["cm"]
per_class_df = bundle["per_class_df"]
overall = bundle["overall_metrics"]
metrics_v1 = bundle["metrics_v1"]
spc_bundle = bundle["spc_bundle"]
contribution_df = bundle["contribution_df"]
journey = bundle["phase4_journey"]

# 범주형 컬러(0=배경,1=정상,2=불량) — Phase 2에서 확정한 팔레트 재사용
# 문법 설명: 왜 구간을 3개로 나눴는가
# 값은 항상 정확히 0/1/2 중 하나이고 zmin=0,zmax=2라 각각 분수 0/0.5/1.0에
# 대응한다. 예전 코드는 0.5 지점에 색을 두 개(파랑·빨강) 겹쳐놨는데,
# Plotly가 이 경계값을 어느 쪽으로 처리할지 애매해서 실제로는 정상(1)도
# 불량(2)과 같은 빨간색으로 나와버렸다(정상/불량이 안 구분됨).
# 그래서 0.5 근처에 아주 좁은 "빈 구간"(0.49~0.51)을 둬서, 0/0.5/1.0 세
# 지점이 각자 확실히 자기 색 구간 안에 떨어지게 만들었다 — 실제 데이터가
# 이 좁은 빈 구간 값을 가질 일이 없으므로 안전하다.
DIE_COLORSCALE = [
    [0.0, "#f0f0f0"], [0.24, "#f0f0f0"],
    [0.26, "#4c72b0"], [0.74, "#4c72b0"],
    [0.76, "#c44e52"], [1.0, "#c44e52"],
]

PROCESS_MAP = {
    "Center": ("챔버 / 포토리소그래피", "RF 동작 불규칙·유체 흐름 불균일 또는 CD 중심-가장자리 불균일", "챔버 정비 점검 또는 CD 계측 확인"),
    "Donut": ("세정", "포토레지스트 잔류물 미제거 (스핀 코팅 유체역학 가능성도 있음)", "세정 공정 재점검"),
    "Edge-Ring": ("RTP / ESC", "온도 조절 이상, 저온 공정 편차, 척 가장자리 접촉 불량", "RTP 온도 프로파일·척 접촉 점검"),
    "Edge-Loc": ("챔버(가장자리 구조물)", "파티클(포커스링·클램프링에서 박리) 또는 어닐링/로드락 온도·밸브 문제", "챔버 가장자리 구조물·파티클 점검"),
    "Loc": ("챔버(중심부·상부 구조물)", "파티클(샤워헤드·챔버 돔에서 박리) 또는 특정 위치 반복 결함", "챔버 상부 구조물·파티클 점검"),
    "Scratch": ("핸들링", "기계적 접촉, 이송 장비 (또는 프로브 접촉 자국)", "이송 장비 점검"),
    "Random": ("파티클", "오염, 환경 요인 — Loc/Edge-Loc과 같은 파티클 축, 다수의 작은 입자가 산발적으로 낙하한 경우", "클린룸 환경 점검"),
    "Near-full": ("전면 이상", "설비 미가동, 잘못된 레시피, 웨이퍼 오식별 중 하나", "설비·레시피·취급 SOP 점검"),
}


# 공정 원인 매핑의 근거 문헌.
# 원문을 그대로 옮기지 않고 메커니즘을 우리말로 요약한 뒤 출처를 밝힌다
# (논문 본문·그림을 그대로 게시하면 저작권 문제가 되고, 유료 논문은
#  본문 접근도 안 된다. 출처를 밝히고 링크로 확인하게 하는 것이 정석이다).
#
# 여기에는 공정 문헌에서 메커니즘이 명확히 확인된 항목만 싣는다.
EVIDENCE = [
    {
        "patterns": ["Loc", "Edge-Loc", "Random"],
        "claim": "챔버 내부에 쌓인 부산물이 박리되어 웨이퍼에 떨어지면 국소 결함이 된다",
        "summary": (
            "플라즈마 장비 문헌은 파티클 발생 경로를 이렇게 기술한다 — 챔버 내벽이 "
            "플라즈마에 노출되면 반응 부산물 막이 쌓이고, 이 막이 시간이 지나며 "
            "박리(flaking)되어 공정 환경을 오염시킨다. 식각 장비의 파티클 발생원으로는 "
            "챔버 표면 증착막의 박리와 정전척에서 떨어져 나오는 물질이 함께 지목된다. "
            "떨어진 입자는 웨이퍼 표면에 낙하해 결함을 만들고 수율을 떨어뜨린다. "
            "'쌓임 → 박리 → 낙하 → 국소 결함'이라는 인과 사슬이 문서화되어 있다."
        ),
        "sources": [
            ("Reduction of Particle Contamination in Plasma-Etching Equipment by "
             "Dehydration of Chamber Wall (Jpn. J. Appl. Phys. 47, 3630)",
             "https://iopscience.iop.org/article/10.1143/JJAP.47.3630"),
            ("Investigation of contamination particles generation and surface chemical "
             "reactions on Al2O3, Y2O3, and YF3 coatings in F-based plasma "
             "(Applied Surface Science)",
             "https://www.sciencedirect.com/science/article/abs/pii/S0169433223010450"),
        ],
    },
    {
        "patterns": ["Center"],
        "claim": "웨이퍼 반경 방향 CD 편차는 웨이퍼 레벨 특성 맵으로 나타난다",
        "summary": (
            "식각 속도와 증착 두께가 반경 방향으로 균일하지 않으면 CD(선폭)가 중심과 "
            "가장자리에서 다르게 형성되고, 이것이 웨이퍼 전체의 특성 분포 맵으로 "
            "드러난다고 보고된다. 웨이퍼 중심-가장자리 CD 편차 문제가 수율에 직접 "
            "영향을 준다는 서술도 함께 확인된다. 실제로 이 편차를 잡기 위해 식각 "
            "장비의 온도 제어 영역이 1개에서 반경 방향 4개 존으로 늘어나는 방향으로 "
            "발전해 왔다."
        ),
        "sources": [
            ("Across-wafer CD uniformity control through lithography and etch process: "
             "Experimental verification (SPIE 6518)",
             "https://www.researchgate.net/publication/228984337_Across-wafer_CD_uniformity_control_through_lithography_and_etch_process_Experimental_verification_-_art_no_65182C"),
            ("Evolution of across-wafer uniformity control in plasma etch "
             "(Semiconductor Digest)",
             "https://sst.semiconductor-digest.com/2016/08/evolution-of-across-wafer-uniformity-control-in-plasma-etch/"),
        ],
    },
    {
        "patterns": ["Edge-Ring", "Edge-Loc"],
        "claim": "웨이퍼 가장자리 구조물(정전척·포커스링)이 링 형태 결함 맵을 만든다",
        "summary": (
            "이온주입 장비 연구는 웨이퍼 가장자리에 분포한 결함이 링 형태의 맵을 "
            "형성하며, 정전척의 프린지 필드가 링 형태 손상의 핵심 원인이라고 보고한다. "
            "포커스링은 정전척 위에서 웨이퍼 가장자리를 둘러싸도록 배치되어 플라즈마를 "
            "가두는 부품이므로, 웨이퍼 가장자리 바로 옆에 위치한다. 가장자리 조건은 "
            "이런 부품을 두고도 중심부보다 떨어지는 경향이 있어 별도 튜닝 대상이 된다."
        ),
        "sources": [
            ("Ring-type ESD damage caused by electrostatic chuck of ion implanter "
             "(SPIE Proceedings 3743)",
             "https://www.spiedigitallibrary.org/conference-proceedings-of-spie/3743/1/Ring-type-ESD-damage-caused-by-electrostatic-chuck-of-ion/10.1117/12.346916.short"),
            ("Defect Challenges Grow At The Wafer Edge (Semiconductor Engineering)",
             "https://semiengineering.com/defect-challenges-grow-at-the-wafer-edge/"),
        ],
    },
    {
        "patterns": ["Scratch"],
        "claim": "선형 스크래치는 장비 취급 과정의 기계적 접촉에서 발생한다",
        "summary": (
            "웨이퍼 맵 결함 분류 문헌에서 스크래치는 장비 핸들링 문제의 결과로 "
            "일관되게 기술된다. 이 항목은 조사한 자료들 사이에 이견이 없었다."
        ),
        "sources": [
            ("Wafer defect recognition method based on multi-scale feature fusion "
             "(Frontiers in Neuroscience)",
             "https://pmc.ncbi.nlm.nih.gov/articles/PMC10272367/"),
        ],
    },
    {
        "patterns": ["Donut"],
        "claim": "스핀 코팅의 가장자리 비드와 그 제거(EBR)가 가장자리 결함률을 좌우한다",
        "summary": (
            "스핀 코팅에서는 표면장력 때문에 기판 가장자리에 두꺼운 비드가 형성되며, "
            "이 과잉 두께가 가장자리 결함률 증가와 수율 저하의 원인이 된다. 그래서 "
            "코팅 직후 EBR로 폭 1~5mm의 환형 영역을 제거하는 것이 표준 절차다. "
            "즉 웨이퍼 최외곽의 상태는 EBR이 규정하므로, 가장자리가 정상으로 보인다는 "
            "사실만으로 앞 공정이 정상이라고 판단할 수 없다."
        ),
        "sources": [
            ("Spin Coating of Photoresists (MicroChemicals, Application Note)",
             "https://www.microchemicals.com/dokumente/application_notes/spin_coating_photoresist.pdf"),
        ],
    },
]


# 웨이퍼 맵은 64x64 정사각 데이터다. Plotly는 기본적으로 그림을 컨테이너 폭에
# 맞춰 늘리므로, 화면이 넓을수록 다이가 가로로 찌그러진다 — PC와 휴대폰에서
# 다르게 보이던 원인이 이것이다. y축을 x축에 1:1로 묶어(scaleanchor) 칸이 항상
# 정사각이 되게 하고, constrain="domain"으로 범위를 늘리는 대신 그림 영역 자체를
# 줄여 가운데 정렬시킨다.
SQUARE_AXES = dict(
    xaxis=dict(visible=False, constrain="domain"),
    yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, constrain="domain"),
)


def wafer_heatmap_fig(img: np.ndarray, title: str, height: int = 300) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=img[::-1], colorscale=DIE_COLORSCALE,
                                     zmin=0, zmax=2, showscale=False))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=35, b=10),
                       **SQUARE_AXES)
    return fig


def gradcam_fig(img64: np.ndarray, heatmap: np.ndarray, title: str) -> go.Figure:
    """64x64 웨이퍼 맵(회색) 위에 Grad-CAM 히트맵을 겹친다. 화면 1·6 공용."""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=img64[::-1], colorscale="gray", zmin=0, zmax=2,
                             showscale=False, opacity=0.5))
    fig.add_trace(go.Heatmap(z=heatmap[::-1], colorscale="Jet",
                             zmin=0, zmax=1, showscale=False, opacity=0.55))
    fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=35, b=10),
                      **SQUARE_AXES)
    return fig


def prob_bar_fig(p: np.ndarray, pred_lab: str, true_lab: str | None = None):
    """9개 클래스 확률 막대. 화면 1·6 공용.

    색으로 역할을 구분한다: 예측한 것(빨강) / 실제 정답(초록) / 나머지(회색).
    화면 6처럼 정답을 모르는 경우 true_lab=None이라 초록 막대가 없다.
    반환: (그림, 확률 높은 순 이름, 확률 높은 순 값)
    """
    order = np.argsort(p)[::-1]          # 확률 높은 순으로 정렬
    names_sorted = [class_names[i] for i in order]
    vals_sorted = p[order]
    bar_colors = []
    for nm in names_sorted:
        if nm == pred_lab:
            bar_colors.append("#c44e52")
        elif true_lab is not None and nm == true_lab:
            bar_colors.append("#55a868")
        else:
            bar_colors.append("#c9ccd1")
    fig = go.Figure(go.Bar(
        x=names_sorted, y=vals_sorted, marker_color=bar_colors,
        text=[f"{v*100:.1f}%" if v >= 0.001 else "<0.1%" for v in vals_sorted],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="확률", range=[0, 1.15], tickformat=".0%"),
        xaxis=dict(title=None),
    )
    return fig, names_sorted, vals_sorted


@st.cache_data
def gap_reference():
    """fold 0 검증셋에서 '1-2등 격차 20%p 미만'이 실제로 오분류를 가려내는지.

    하드코딩하면 나중에 모델이 바뀔 때 조용히 거짓말이 되므로 매번 계산한다.
    반환: (격차 좁은 웨이퍼 수, 그중 오분류율, 격차 넓은 웨이퍼의 오분류율)
    """
    srt = np.sort(probs_all, axis=1)[:, ::-1]
    narrow = (srt[:, 0] - srt[:, 1]) < 0.20
    wrong = ~wafer_table["correct"].values
    return int(narrow.sum()), float(wrong[narrow].mean()), float(wrong[~narrow].mean())


def _grid(maps, labels, wafer_idx, cols: int = 6, label_is_lot: bool = False):
    """웨이퍼 맵 여러 장을 격자로 늘어놓는다.

    Lot 응집도를 눈으로 확인시키는 용도라, 개별 맵을 크게 보여주기보다
    한 화면에 여러 장을 나란히 놓아 '비슷한가/다른가'가 바로 보이게 한다.
    """
    for start in range(0, len(maps), cols):
        row = st.columns(cols)
        for slot, k in enumerate(range(start, min(start + cols, len(maps)))):
            with row[slot]:
                fig = go.Figure(go.Heatmap(z=maps[k][::-1], colorscale=DIE_COLORSCALE,
                                           zmin=0, zmax=2, showscale=False))
                fig.update_layout(height=150,
                                  margin=dict(l=2, r=2, t=2, b=2),
                                  **SQUARE_AXES)
                # key를 안 주면 같은 화면에 차트가 여러 개일 때 Streamlit이
                # 중복 ID로 오류를 낸다.
                st.plotly_chart(fig, width='stretch',
                                key=f"grid_{label_is_lot}_{start}_{slot}_{labels[k]}",
                                config={"displayModeBar": False})
                cap = labels[k]
                sub = f"slot {wafer_idx[k]}" if wafer_idx[k] > 0 else ""
                # 정상(none)은 흐리게 — 불량이 어느 슬롯에 몰렸는지가 한눈에 보이도록
                if not label_is_lot and cap == "none":
                    st.caption(f":gray[정상]  \n:gray[{sub}]")
                else:
                    st.caption(f"**{cap}**  \n{sub}")


# --------------------------------------------
# 화면 선택
# --------------------------------------------
PAGES = ["웨이퍼 맵 진단", "패턴별 성능", "Lot 수율 모니터링", "공정 원인 매핑", "모델 방법론"]
# 화면 6은 기존 번호(PAGES[0~4])를 건드리지 않으려고 따로 이름을 두고,
# 메뉴에서는 '웨이퍼 맵 진단' 바로 다음에 보이게 끼워 넣는다.
PAGE_LIVE = "새 웨이퍼 즉석 판정"
page = st.sidebar.radio("화면 선택", [PAGES[0], PAGE_LIVE] + PAGES[1:])
st.sidebar.divider()
st.sidebar.caption(
    f"WM-811K · 라벨 172,950장 중 12,763장 샘플 학습\n\n"
    f"최종 모델: CNN(Flatten head) · macro-F1 {overall['oof_macro_f1']:.4f}"
)

st.title("웨이퍼 맵 불량 패턴 분류 대시보드")

# ============================================
# 화면 1: 웨이퍼 맵 진단
# ============================================
if page == PAGES[0]:
    st.subheader("웨이퍼 단위 예측 및 Grad-CAM 근거 확인")
    st.caption(
        "아래 목록은 fold 0 모델이 **학습에 쓰지 않은** 검증셋(2,553장)입니다. "
        "즉 이 모델 입장에서 전부 '처음 보는' 웨이퍼입니다."
    )

    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        view = st.radio("웨이퍼 목록", ["전체", "오분류만", "클래스로 보기"])
        if view == "오분류만":
            cand_df = wafer_table[~wafer_table["correct"]]
        elif view == "클래스로 보기":
            cls_pick = st.selectbox("클래스", sorted(class_names))
            cand_df = wafer_table[wafer_table["true_label"] == cls_pick]
        else:
            cand_df = wafer_table
        if len(cand_df) == 0:
            st.warning("조건에 맞는 웨이퍼가 없습니다.")
            st.stop()
        row_pos = st.selectbox(
            "웨이퍼 선택", cand_df.index,
            format_func=lambda i: f"#{i} (실제:{wafer_table.loc[i,'true_label']})",
        )

    row = wafer_table.loc[row_pos]
    local_pos = wafer_table.index.get_loc(row_pos)
    img = X_val0[local_pos]

    with col_info:
        c1, c2, c3 = st.columns(3)
        c1.metric("실제 라벨", row["true_label"])
        c2.metric("예측 라벨", row["pred_label"],
                   delta="일치" if row["correct"] else "불일치",
                   delta_color="normal" if row["correct"] else "inverse")
        c3.metric("예측 확률", f"{row['pred_prob']:.3f}")

    st.divider()
    x_t = torch.from_numpy((img.astype(np.float32) / 2.0)[None, None, :, :])
    heatmap, target_cls, prob = gradcam.generate(x_t, target_class=None)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(wafer_heatmap_fig(img, "원본 웨이퍼 맵"), width='stretch')
    with g2:
        fig_cam = gradcam_fig(img, heatmap,
                              f"Grad-CAM (근거: {class_names[target_cls]}, p={prob:.2f})")
        st.plotly_chart(fig_cam, width='stretch')

    st.caption(
        "히트맵이 밝을수록 모델이 그 위치를 근거로 판단했다는 뜻입니다. "
        "리포트 05에서 확인했듯, 정량 지표만으로는 판단이 혼재되어 있어 "
        "육안으로 같이 확인하는 것이 좋습니다 — 예: Donut/Scratch는 형태를 "
        "잘 따라가지만, Center/Edge-Ring은 지표상 통제군보다 약합니다."
    )

    # ---- 9개 클래스 전체 확률 ----
    # 최댓값 하나만 보면 "0.99로 확신한 예측"과 "0.35 vs 0.33으로 간신히
    # 이긴 예측"이 구분되지 않는다. 실무에서 이 둘은 신뢰도가 전혀 다르므로
    # 2등 이하까지 전부 보여준다.
    if probs_all is not None:
        st.divider()
        st.markdown("##### 9개 클래스 전체 예측 확률")

        # 맞힌 경우엔 예측과 정답이 같은 막대이므로 빨강 하나만 보인다.
        fig_p, names_sorted, vals_sorted = prob_bar_fig(
            probs_all[local_pos], row["pred_label"], row["true_label"])
        st.plotly_chart(fig_p, width='stretch')

        # 1등과 2등의 격차 = 모델이 얼마나 확신했는가.
        # 이 임계값(20%p)이 실제로 의미가 있는지는 검증셋 전체로 확인할 수 있다.
        # (하드코딩하면 나중에 모델이 바뀔 때 조용히 거짓말이 되므로 매번 계산한다)
        gap = vals_sorted[0] - vals_sorted[1]
        n_narrow, err_narrow, err_wide = gap_reference()

        msg = (f"1등 **{names_sorted[0]}** {vals_sorted[0]*100:.1f}% · "
               f"2등 **{names_sorted[1]}** {vals_sorted[1]*100:.1f}% · "
               f"격차 **{gap*100:.1f}%p**")
        if gap < 0.20:
            st.warning(msg + " — 격차가 좁습니다. 모델이 두 패턴 사이에서 망설인 사례로, "
                             "사람이 재확인할 가치가 있습니다.")
        else:
            st.info(msg)

        st.caption(
            "빨강=모델이 예측한 클래스, 초록=실제 라벨, 회색=나머지. "
            "9개 확률의 합은 항상 1입니다(softmax). "
            "정확도만 보면 맞고 틀림밖에 안 보이지만, 이 분포를 보면 "
            "**모델이 무엇과 헷갈렸는지**까지 알 수 있습니다."
        )
        st.caption(
            f"이 검증셋({len(wafer_table):,}장)에서 **1-2등 격차가 20%p 미만인 웨이퍼는 "
            f"{n_narrow}장이고, 그중 {err_narrow*100:.1f}%가 오분류**입니다 — "
            f"격차가 20%p 이상인 웨이퍼의 오분류율 {err_wide*100:.1f}%의 "
            f"약 {err_narrow/err_wide:.0f}배입니다. 즉 확률 분포의 격차는 "
            "'이 예측을 사람이 다시 봐야 하는가'를 실제로 가려냅니다."
        )

# ============================================
# 화면 2: 패턴별 성능
# ============================================
elif page == PAGES[1]:
    st.subheader("클래스별 성능 (5-Fold 교차검증 OOF 기준)")
    st.caption(
        "모든 수치는 out-of-fold 예측 기준입니다 — 각 웨이퍼는 자신을 "
        "학습에 쓰지 않은 fold의 모델에게 평가받았습니다(CLAUDE.md 원칙)."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("macro-F1 (주 지표)", f"{overall['oof_macro_f1']:.4f}",
              delta=f"{overall['oof_macro_f1'] - metrics_v1['oof_macro_f1']:+.4f} (1차 대비)")
    m2.metric("balanced accuracy", f"{overall['oof_balanced_accuracy']:.4f}")
    m3.metric("accuracy (참고용)", f"{overall['oof_accuracy']:.4f}",
              help="클래스별 편차를 가릴 수 있어 macro-F1을 우선한다")

    st.divider()
    st.markdown("#### 클래스별 정밀도·재현율·F1")
    disp = per_class_df.copy()
    disp["목표(F1≥0.80)"] = disp["f1"].apply(lambda v: "✅" if v >= 0.80 else "❌")
    a, b = st.columns([2, 1])
    with a:
        st.dataframe(disp.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
                     hide_index=True, width='stretch')
    with b:
        fig_bar = go.Figure(go.Bar(x=disp["f1"], y=disp["class"], orientation="h",
                                    marker_color=["#28a745" if v >= 0.8 else "#dc3545" for v in disp["f1"]]))
        fig_bar.add_vline(x=0.80, line_dash="dash", line_color="gray")
        fig_bar.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="F1")
        st.plotly_chart(fig_bar, width='stretch')

    st.divider()
    st.markdown("#### 9x9 혼동 행렬")
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig_cm = go.Figure(go.Heatmap(z=cm_norm, x=list(class_names), y=list(class_names),
                                   colorscale="Blues", zmin=0, zmax=1,
                                   text=np.round(cm_norm, 2), texttemplate="%{text}"))
    fig_cm.update_layout(height=500, xaxis_title="예측", yaxis_title="실제",
                          yaxis_autorange="reversed")
    st.plotly_chart(fig_cm, width='stretch')

    idx_el, idx_sc, idx_loc = [list(class_names).index(c) for c in ["Edge-Loc", "Scratch", "Loc"]]
    el_to_sc = cm[idx_el, idx_sc] / cm[idx_el].sum()
    center_to_loc = cm[list(class_names).index("Center"), idx_loc] / cm[list(class_names).index("Center")].sum()
    st.success(
        f"**원래 걱정했던 Edge-Loc↔Scratch 혼동은 목표(10%)를 달성**했습니다 "
        f"(Edge-Loc→Scratch {el_to_sc:.1%}). 대신 **예상 못 했던 'Loc'이 혼동 허브**였는데 "
        f"(1차 시도에서 Center→Loc 24.5%), Flatten head 개선 후 {center_to_loc:.1%}로 급감했습니다."
    )

# ============================================
# 화면 3: Lot 수율 모니터링
# ============================================
elif page == PAGES[2]:
    st.subheader("Lot 단위 SPC 관리도")
    st.caption(
        "SECOM의 주간 p-관리도 로직을 Lot 단위로 재구성했습니다. "
        "**주의**: 원 논문 Training 라벨을 제외한 Test 라벨(118,595장)만 사용합니다 — "
        "Training 라벨이 Lot 순서 분석을 심하게 왜곡시킨다는 것을 발견했습니다(하단 설명 참고)."
    )

    chart_data = spc_bundle["chart_data"]
    pattern = st.selectbox("패턴 선택", list(chart_data.keys()))
    g = chart_data[pattern]

    n_alert = int(g["over_ucl"].sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("전체 발생률(중심선)", f"{g['p_bar'].iloc[0]*100:.3f}%")
    k2.metric("관리한계 초과 구간", f"{n_alert} / {len(g)}개")
    if n_alert > 0:
        worst = g[g["over_ucl"]].sort_values("z", ascending=False).iloc[0]
        k3.metric("가장 심한 초과", f"Z={worst['z']:+.1f}, p={worst['pval']:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g["rate"] * 100, mode="lines+markers",
                              name="발생률(%)", line=dict(color="#4c72b0")))
    fig.add_trace(go.Scatter(x=g.index, y=g["ucl"] * 100, mode="lines", name="UCL",
                              line=dict(color="#cc3333", dash="dash")))
    fig.add_hline(y=g["p_bar"].iloc[0] * 100, line_dash="dot", line_color="gray",
                  annotation_text="중심선")
    over = g[g["over_ucl"]]
    if len(over) > 0:
        fig.add_trace(go.Scatter(x=over.index, y=over["rate"] * 100, mode="markers",
                                  name="관리한계 초과", marker=dict(color="red", size=10)))
    fig.update_layout(height=380, xaxis_title=f"Lot 묶음 순서 (그룹당 라벨 약 {spc_bundle['target_group_size']}장 — 실제 생산 시각 아님)",
                       yaxis_title="발생률(%)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.markdown("#### 구간별 표본 크기")
    gs = spc_bundle["group_sizes"]
    fig_n = go.Figure(go.Bar(x=gs.index, y=gs.values, marker_color="#888888"))
    fig_n.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="라벨 수")
    st.plotly_chart(fig_n, width='stretch')

    with st.expander("왜 Training 라벨을 제외했는가 (중요한 발견)"):
        st.markdown(
            "관리도를 처음 그렸을 때, 거의 모든 패턴이 Lot 순서 초반 구간에서 "
            "동시에 극단적으로 치솟았습니다. 원인을 파고든 결과, **그 구간 웨이퍼의 "
            "100%가 원 연구자의 'Training' 라벨**이었습니다 — 원 연구자가 균형 잡힌 "
            "학습셋을 만들려고 다양한 불량 유형을 골고루 뽑은 세트가 Lot 순서상 "
            "우연히 뭉쳐 있어 생긴 착시였지, 실제 생산 이상이 아니었습니다. "
            "발견 즉시 Test 라벨만 남기고 다시 계산했습니다."
        )

    # ------------------------------------------------------------
    # Lot 응집도 — "같은 Lot이면 같은 불량이 나오는가"
    # Lot은 같은 카세트로 같은 공정 경로를 함께 통과한 묶음이므로,
    # 원인이 '공정 조건'이면 Lot 전체가 같이 영향받아 뭉쳐야 하고,
    # '웨이퍼 개별 확률 사건'이면 뭉치지 않아야 한다. 그 차이를 측정한다.
    # ------------------------------------------------------------
    if lot_summary is not None and lot_pattern_df is not None:
        st.divider()
        st.subheader("같은 Lot의 웨이퍼는 같은 불량 양상을 보이는가")
        st.caption(
            "Lot은 같은 카세트로 **같은 공정 경로를 함께 통과한** 묶음입니다. "
            "원인이 공정 조건이면 Lot 전체가 같이 영향을 받아 패턴이 뭉쳐야 하고, "
            "웨이퍼 한 장 단위의 확률 사건(파티클 낙하 등)이면 뭉치지 않아야 합니다."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("실제 일치율", f"{lot_summary['observed']:.2f}%")
        m2.metric("통제군(무작위)", f"{lot_summary['control']:.2f}%")
        m3.metric("배율", f"{lot_summary['lift']:.2f}x")
        m4.metric("Z", f"+{lot_summary['z']:.1f}")
        st.caption(
            f"Test 라벨 · none 제외 조건 · 웨이퍼 {lot_summary['wafers']:,}장 / "
            f"비교 쌍 {lot_summary['pairs']:,}개. "
            "**통제군**은 라벨을 전체에서 무작위로 섞어 Lot 구조를 파괴한 뒤 "
            "같은 계산을 한 값(20회 평균)입니다 — 'Lot이 아무 의미 없었다면 나왔을 값'이며, "
            "이것보다 높아야만 '뭉친다'고 말할 수 있습니다."
        )

        st.markdown("##### 패턴별 응집도")
        lp = lot_pattern_df.sort_values("실제(%)", ascending=False)
        fig_lot = go.Figure()
        fig_lot.add_trace(go.Bar(
            x=lp["패턴"], y=lp["실제(%)"], name="실제 일치율",
            marker_color="#c44e52",
            text=[f"{v:.1f}%" for v in lp["실제(%)"]], textposition="outside"))
        fig_lot.add_trace(go.Bar(
            x=lp["패턴"], y=lp["통제군(%)"], name="통제군(무작위)",
            marker_color="#c9ccd1"))
        fig_lot.update_layout(
            barmode="group", height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="같은 Lot의 다른 불량 웨이퍼도 같은 패턴일 확률 (%)",
                       range=[0, 95]),
            legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig_lot, width='stretch')

        show = lp.copy()
        show["배율"] = show["배율"].map(lambda v: f"{v:.2f}x")
        for c in ("실제(%)", "통제군(%)"):
            show[c] = show[c].map(lambda v: f"{v:.2f}")
        st.dataframe(show, width='stretch', hide_index=True)

        st.markdown(
            "**읽는 법** — Edge-Ring(8.35x)·Center(7.70x)처럼 **Lot 전체에 걸리는 "
            "공정 조건**(RTP 온도 프로파일, CD 불균일)이 원인으로 추정되는 패턴은 강하게 "
            "뭉칩니다. 반대로 Edge-Loc(1.84x)·Loc(1.82x)·Scratch(2.07x)는 8개 패턴 중 "
            "**응집도가 가장 낮습니다** — 파티클 낙하나 핸들링 접촉은 웨이퍼 한 장 단위의 "
            "확률 사건이라 Lot 전체에 걸리지 않는다는 가설과 맞습니다."
        )

        with st.expander("이 분석의 한계 (반드시 함께 읽을 것)"):
            st.markdown(
                "- **인과가 아닙니다.** 설비ID·레시피·타임스탬프가 없어 "
                "'공정 조건이 원인'을 증명하지 못합니다. "
                "'같은 Lot이면 패턴이 비슷하다'는 통계적 연관까지만 말할 수 있습니다.\n"
                "- **라벨링 절차 자체가 교란일 수 있습니다.** 원 연구자가 Lot 단위로 "
                "몰아 보며 라벨링했다면 일치율이 인위적으로 오릅니다. Lot당 라벨 수 "
                "중앙값이 23/25장이라 'Lot 통째 라벨링'에 가까운 정황이 실제로 있고, "
                "**배제할 방법이 없습니다.**\n"
                "- **Random(22.25x)은 위 설명에 맞지 않습니다.** 파티클 계열로 분류했는데 "
                "강하게 뭉칩니다. 지속적 오염(필터 열화 등)이면 설명은 되지만 확인 불가라, "
                "**가설과 어긋나는 관찰로 그대로 기록합니다.**\n"
                "- **Donut(436쌍)·Near-full(259쌍)은 표본이 작습니다.**\n"
                "- **배율과 절대값은 순위가 다릅니다.** 배율은 희귀한 패턴일수록 기계적으로 "
                "커집니다(Random이 배율 1위인 이유). '얼마나 뭉치는가'는 절대값으로, "
                "'우연이 아닌가'는 배율로 봐야 합니다.\n"
                "- **Training 라벨을 포함하면 92.73%**로 부풀려집니다(Test만 쓰면 65.52%). "
                "위 SPC 관리도에서 겪은 것과 **같은 함정이 두 번째로 재발**한 사례입니다."
            )

    # ------------------------------------------------------------
    # Lot별 웨이퍼 맵 갤러리 — 위 응집도 수치를 눈으로 확인
    # 숫자(2.92배)만으로는 "실제로 얼마나 비슷한지"가 안 와닿는다.
    # 같은 Lot의 맵을 나란히 놓으면 바로 보인다.
    # ------------------------------------------------------------
    if lot_gallery:
        st.divider()
        st.subheader("Lot별 웨이퍼 맵 — 직접 보기")
        st.caption(
            "위 수치를 눈으로 확인하는 화면입니다. "
            "**단일 패턴 Lot과 혼재 Lot을 둘 다** 담았습니다 — "
            "뭉친 것만 보여주면 '좋은 것만 골랐다'는 반박을 받기 때문입니다."
        )

        gview = st.radio("보기 방식",
                         ["Lot 하나씩 보기", "패턴별로 모아 보기"],
                         horizontal=True, key="gallery_view")

        if gview == "Lot 하나씩 보기":
            # Lot이 900개가 넘으므로 드롭다운에 그냥 넣으면 못 고른다. 먼저 좁힌다.
            all_pats = sorted({v["dominant"] for v in lot_gallery.values()})
            g1, g2, g3 = st.columns([1, 1, 1])
            with g1:
                grp = st.selectbox("Lot 유형", ["전체", "단일 패턴", "혼재"],
                                   key="lot_grp")
            with g2:
                pat_f = st.selectbox("주 불량 패턴", ["전체"] + all_pats,
                                     key="lot_pat_f")
            with g3:
                min_def = st.slider("최소 불량 장수", 3, 25, 3, key="lot_min_def")

            cand = {
                k: v for k, v in lot_gallery.items()
                if (grp == "전체" or v["group"] == grp)
                and (pat_f == "전체" or v["dominant"] == pat_f)
                and v["n_defect"] >= min_def
            }
            st.caption(
                f"전체 **{len(lot_gallery):,}개** Lot 중 조건에 맞는 Lot "
                f"**{len(cand):,}개** — Test 라벨 · 불량 3장 이상인 Lot을 전부 담았습니다."
            )
            if not cand:
                st.warning("조건에 맞는 Lot이 없습니다. 필터를 완화해 주세요.")
                st.stop()
            lot_pick = st.selectbox(
                "Lot 선택", list(cand.keys()),
                format_func=lambda k: (f"{k} · 불량 {cand[k]['n_defect']}장 / "
                                       f"{cand[k]['n_shown']}장 · "
                                       f"패턴 {'·'.join(cand[k]['kind_list'])}"),
                key="lot_pick")
            info = cand[lot_pick]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("라벨된 웨이퍼", f"{info['n_shown']}장")
            c2.metric("불량", f"{info['n_defect']}장")
            c3.metric("정상(none)", f"{info['n_normal']}장")
            c4.metric("불량 패턴 종류", f"{info['n_kinds']}종")
            st.caption(
                f"**{lot_pick}** · 카세트 슬롯 번호 순서로 배열했습니다. "
                f"최빈 불량 패턴 비중 **{info['top_share']*100:.0f}%** · "
                f"패턴 {' / '.join(info['kind_list'])}"
            )
            _grid(info["maps"], info["patterns"], info["wafer_idx"])
            if info["n_kinds"] == 1:
                st.success(
                    f"이 Lot의 불량 {info['n_defect']}장이 **전부 "
                    f"{info['kind_list'][0]}** 입니다. 같은 공정 경로를 통과한 묶음이 "
                    "같은 방식으로 영향받았다는 가설과 맞는 모습입니다.")
            else:
                st.warning(
                    f"이 Lot에는 불량 패턴이 **{info['n_kinds']}종** 섞여 있습니다. "
                    "모든 Lot이 뭉치는 것은 아니며, 응집도 65.52%는 "
                    "**평균값**이라는 점을 보여주는 사례입니다.")
        else:
            # 패턴별: 갤러리 전체를 패턴 기준으로 다시 묶는다
            by_pat = {}
            for lot, v in lot_gallery.items():
                for m, p, wi in zip(v["maps"], v["patterns"], v["wafer_idx"]):
                    by_pat.setdefault(p, []).append((m, lot, wi))
            pat_pick = st.selectbox(
                "패턴 선택", sorted(by_pat.keys()),
                format_func=lambda p: f"{p} · {len(by_pat[p])}장", key="pat_pick")
            n_show = st.slider("표시 장수", 6, min(48, len(by_pat[pat_pick])),
                               min(24, len(by_pat[pat_pick])), step=6, key="pat_n")
            items = by_pat[pat_pick][:n_show]
            st.markdown(
                f"**{pat_pick}** — 갤러리에 담긴 {len(by_pat[pat_pick])}장 중 "
                f"{len(items)}장 표시. 캡션은 이 웨이퍼가 속한 Lot입니다."
            )
            _grid([i[0] for i in items],
                  [i[1] for i in items],
                  [i[2] for i in items], label_is_lot=True)
            st.caption(
                "같은 패턴이라도 Lot이 다르면 모양의 세부는 제각각입니다. "
                "모델은 이 변형들을 하나의 패턴으로 묶어내야 합니다."
            )

# ============================================
# 화면 4: 공정 원인 매핑
# ============================================
elif page == PAGES[3]:
    st.subheader("패턴별 공정 원인 매핑 및 수율 손실 기여도")
    st.error(
        "**이 데이터셋엔 실제 공정 데이터(CD·RF파워·챔버온도·설비ID)가 없습니다.** "
        "아래 '문헌상 공정 모듈' 열은 문헌 기반 도메인 지식이며, "
        "이 프로젝트의 데이터로 검증된 결론이 아닙니다."
    )

    disp = contribution_df.copy()
    disp["문헌상 공정 모듈"] = disp["패턴"].map(lambda p: PROCESS_MAP.get(p, ("-", "-", "-"))[0])
    disp["물리적 원인"] = disp["패턴"].map(lambda p: PROCESS_MAP.get(p, ("-", "-", "-"))[1])
    disp["권장 조치"] = disp["패턴"].map(lambda p: PROCESS_MAP.get(p, ("-", "-", "-"))[2])
    st.dataframe(
        disp[["기여도 순위", "패턴", "발생률(%)", "평균 불량다이비율(%)", "수율손실 기여도",
              "문헌상 공정 모듈", "물리적 원인", "권장 조치"]]
        .style.format({"발생률(%)": "{:.3f}", "평균 불량다이비율(%)": "{:.2f}", "수율손실 기여도": "{:.5f}"}),
        hide_index=True, width='stretch',
    )

    fig = go.Figure(go.Bar(x=disp["수율손실 기여도"], y=disp["패턴"], orientation="h",
                            marker_color="#4c72b0"))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="수율 손실 기여도 (발생률 × 평균 불량다이비율)")
    st.plotly_chart(fig, width='stretch')

    st.info(
        "Edge-Loc과 Loc이 발생률·다이당 불량 비율 둘 다 상당해 기여도 1·2위입니다. "
        "Near-full은 발생률은 최하위(0.08%)지만 다이당 불량 비율이 87.6%로 압도적이라 "
        "**발생 시 피해가 매우 큽니다** — 발생률만으로 우선순위를 정하면 놓칠 수 있는 지점입니다."
    )

    # ------------------------------------------------------------
    # 근거 문헌 — 위 매핑이 어디에 기반했는지 확인 가능하게 한다
    # ------------------------------------------------------------
    st.divider()
    st.subheader("공정 원인 매핑의 근거 문헌")
    st.caption(
        "위 표의 공정 모듈·원인은 반도체 공정 문헌에 기술된 메커니즘에 기반합니다. "
        "아래는 각 메커니즘이 어떤 자료에서 확인되는지 정리한 것입니다. "
        "원문을 그대로 옮기지 않고 메커니즘을 요약했으며, 출처 링크로 직접 확인하실 수 있습니다."
    )

    for ev in EVIDENCE:
        with st.expander(
            f"{' · '.join(ev['patterns'])}  —  {ev['claim']}", expanded=False
        ):
            st.markdown(ev["summary"])
            st.markdown("**출처**")
            for title, url in ev["sources"]:
                st.markdown(f"- [{title}]({url})")

    st.warning(
        "**이 데이터로 검증한 것은 아닙니다.** WM-811K에는 설비 ID·레시피·공정 "
        "파라미터·타임스탬프가 없습니다(컬럼은 waferMap · dieSize · lotName · "
        "waferIndex · trainTestLabel · failureType 6개뿐이고, 라벨은 패턴 이름 9종). "
        "따라서 위 대응 관계는 **공정 지식에 근거한 원인 후보**이지, "
        "이 데이터셋에서 인과를 확인한 결과가 아닙니다."
    )

# ============================================
# 화면 5: 모델 방법론
# ============================================
elif page == PAGES[4]:
    st.subheader("모델 학습 방법론 및 개선 과정")

    st.markdown("#### 데이터 누수 차단 (최우선 원칙)")
    st.code(
        "for train_idx, val_idx in skf.split(X, y):\n"
        "    X_tr = augment(X[train_idx])   # 학습 fold만 증강\n"
        "    X_val = X[val_idx]             # 검증 fold는 원본 그대로",
        language="python",
    )
    st.caption(
        "회전·뒤집기 증강은 라벨을 바꾸지 않아 물리적으로 타당하지만, "
        "분할 전에 적용하면 같은 원본의 변형이 학습셋과 검증셋에 나뉘어 들어가는 "
        "누수가 생긴다(SECOM의 SMOTE 누수와 동일한 문제). Phase 3에서 이 구조를 "
        "assert로 직접 검증했다."
    )

    st.divider()
    st.markdown("#### Phase 4 반복 개선 과정 — 실패를 포함해 전부 기록")

    with st.expander("1차 시도 (GAP 구조) — 실패", expanded=False):
        st.text(journey["v1_summary"][:1200])
        st.error(f"macro-F1 {metrics_v1['oof_macro_f1']:.4f} — 목표(9개 중 7개 F1≥0.80) 미달")

    with st.expander("원인 분석: GAP이 위치·모양 정보를 지움"):
        st.markdown(
            "클래스를 '밀도로 구분되는 클래스'(Edge-Ring·Random·Near-full·none)와 "
            "'위치·모양으로 구분되는 클래스'(Center·Loc·Scratch·Donut)로 나누면 "
            "성능이 정확히 갈렸다. Global Average Pooling이 마지막에 위치 정보를 "
            "평균 내 지워버리는 것이 원인으로 추정됐다."
        )

    with st.expander("개선 Step 1: Flatten head — 성공"):
        st.text(journey["head_check"])
        st.success(f"최종 macro-F1 {overall['oof_macro_f1']:.4f} — 목표 달성. 채택 확정.")

    with st.expander("개선 Step 2: 해상도 128x128 — 실패, 폐기"):
        st.text(journey["resolution_check"])
        st.error("전 클래스 악화(macro-F1 -0.052). 입력만 키우고 층수를 안 늘린 것이 원인으로 추정.")

    with st.expander("개선 Step 3: 모양(선형성) 보조 특징 — 실패, 폐기"):
        st.text(journey["aux_check"])
        st.error("Scratch는 개선(+0.031)됐지만 다른 클래스가 대부분 악화되어 순손실(-0.016).")

    st.divider()
    st.markdown("#### 최종 결정")
    st.info(
        "두 번의 추가 개선 시도가 모두 실패한 뒤, Flatten 단독 결과를 Phase 4 "
        "최종으로 확정했다. Scratch(F1 0.697)는 근본 원인이 Phase 1의 리사이즈 "
        "단계(정보 손실)에 있다고 판단해 추가 모델 구조 개선을 중단했다 — "
        "이미 최소·우수 목표(macro-F1 기준)를 달성했다는 점을 근거로 삼았다."
    )

    st.divider()
    st.markdown("#### 정직하게 남겨둔 한계")
    st.warning(
        "- Scratch(F1 0.697): 목표(0.70) 근소 미달, 리사이즈 단계 정보 손실이 근본 원인으로 추정\n"
        "- Near-full(149장): fold당 검증 29~30장, 결과가 시드에 따라 흔들릴 수 있는 구조적 불안정성\n"
        "- Grad-CAM 정량 지표: '칸이 정확히 겹쳐야 인정'하는 방식이라 윤곽선 기반 판단 근거를 과소평가\n"
        "- 공정 데이터 없음: 패턴→공정 모듈 매핑은 문헌 기반, 이 데이터로 검증된 것 아님\n"
        "- Training/Test 분할 왜곡: 원 논문 저자의 라벨 curation이 Lot 순서 분석을 왜곡시킴 (Phase 6에서 발견·보정)"
    )

# ============================================
# 화면 6: 새 웨이퍼 즉석 판정
# ============================================
# 다른 화면은 미리 계산해 둔 결과를 보여주지만, 이 화면만은 사용자가 넣은
# 웨이퍼 맵을 그 자리에서 모델에 통과시킨다. 계산 부분(파일 해석·입력 검사·예측)은
# src/live_inference.py에 있고, 여기는 화면 배치만 담당한다.
@st.cache_resource
def load_live():
    models5 = load_fold_models(ART, len(class_names))
    path = os.path.join(ART, "unlabeled_samples.joblib")
    live = joblib.load(path) if os.path.exists(path) else None
    hpath = os.path.join(ART, "heldout_eval.joblib")
    held = joblib.load(hpath) if os.path.exists(hpath) else None
    return models5, live, held


if page == PAGE_LIVE:
    MAX_FILES = 100
    st.subheader("새 웨이퍼 즉석 판정")
    st.caption(
        "웨이퍼 맵 파일을 넣으면 **그 자리에서** 모델이 판정합니다. 결과를 미리 저장해 둔 "
        "다른 화면과 달리, 여기서는 넣는 순간 64×64 리사이즈 → CNN → Grad-CAM이 실제로 실행됩니다. "
        "학습 때와 전처리가 같은지는 화면 1의 저장된 확률과 **오차 0으로 일치**하는 것을 확인했습니다."
    )
    models5, live, held = load_live()
    ref = live["ref"] if live else None

    # ---- 이 화면의 판정을 어디까지 믿을 수 있는가 (먼저 보여준다) ----
    if held is not None:
        with st.expander(held["headline"], expanded=False):
            st.markdown(held["summary_md"])

    # ---- 입력 ----
    src_opts = ["파일 올리기", "예시 Lot 한 통 (25장)", "미라벨 웨이퍼 무작위 추출"]
    src_mode = st.radio("입력 방법", src_opts, horizontal=True)
    items, errors = [], []          # items: (이름, 웨이퍼 맵) / errors: (이름, 이유)

    if src_mode == src_opts[0]:
        c1, c2 = st.columns([3, 1])
        rule = c1.radio("값 규칙", [RULE_WM, RULE_BIN])
        pass_bin = c2.number_input("Pass bin 번호", value=1, step=1,
                                   disabled=(rule == RULE_WM),
                                   help="Bin 코드 형식일 때만 씁니다. 이 번호만 정상, 나머지 번호는 모두 불량으로 봅니다.")
        files = st.file_uploader(
            "웨이퍼 맵 파일을 끌어다 놓으세요 — 여러 장을 한 번에 넣으면 Lot 단위로 요약합니다",
            type=["csv", "txt", "npy"], accept_multiple_files=True)
        with st.expander("파일 형식 안내"):
            st.markdown(
                "- **한 파일 = 웨이퍼 한 장**, 격자 한 칸 = 다이 한 개. 크기는 자유(모델 입력 시 64×64로 줄임)\n"
                "- **WM-811K 형식**: 모든 칸이 0(다이 없음) / 1(정상) / 2(불량)\n"
                "- **Bin 코드 형식**: 빈칸 또는 음수 = 다이 없음, Pass bin 번호 = 정상, 그 외 번호 = 불량. "
                "테스트 장비가 내는 bin map을 그대로 넣을 때 씁니다\n"
                "- CSV(쉼표), TXT(공백 구분), NPY(2차원 숫자 배열)\n\n"
                "시연용 파일이 저장소 samples 폴더에 있습니다 — 전부 **라벨이 없는** 원본 웨이퍼이고, "
                "줄무늬처럼 9개 패턴에 속하지 않는 모양은 빼고 모양이 서로 다른 것을 사람이 골랐습니다."
            )
            st.code("0,0,1,1,1,0,0\n0,1,1,2,1,1,0\n1,1,2,2,1,1,1\n0,1,1,1,1,1,0\n0,0,1,1,1,0,0",
                    language="text")
        if files and len(files) > MAX_FILES:
            st.warning(f"한 번에 {MAX_FILES}장까지만 판정합니다. 앞의 {MAX_FILES}장만 사용합니다.")
        for f in (files or [])[:MAX_FILES]:
            m, err = parse_wafer_file(f.getvalue(), f.name, rule, int(pass_bin))
            if err is None:
                items.append((f.name, m))
            else:
                errors.append((f.name, err))

    elif src_mode == src_opts[1]:
        if live is None:
            st.error("unlabeled_samples.joblib이 없습니다. src/13_unlabeled_samples.py를 먼저 실행하세요.")
            st.stop()
        lot_dir = os.path.join(BASE, "samples", live["demo_lot"])
        # 저장소에 있는 CSV를 '파일 올리기'와 똑같은 해석 함수로 읽는다 —
        # 폰처럼 파일을 끌어다 놓기 어려운 환경에서도 같은 경로를 시연하기 위해서다.
        for fn in sorted(os.listdir(lot_dir)):
            with open(os.path.join(lot_dir, fn), "rb") as fh:
                m, err = parse_wafer_file(fh.read(), fn, RULE_WM)
            if err is None:
                items.append((fn, m))
            else:
                errors.append((fn, err))
        st.caption(
            f"samples/{live['demo_lot']} 폴더 — 원본에 25장이 있고 **그중 라벨 있는 웨이퍼가 0장**인 Lot입니다. "
            "단, 줄무늬 없이 불량 덩어리가 뚜렷한 Lot을 **사람이 그림을 보고 시연용으로 고른 것**이라 "
            "일반적인 Lot을 대표하지 않습니다(모델 판정은 고를 때 보지 않았습니다). "
            "있는 그대로의 모습은 '미라벨 웨이퍼 무작위 추출'에서 보세요."
        )

    else:
        if live is None:
            st.error("unlabeled_samples.joblib이 없습니다. src/13_unlabeled_samples.py를 먼저 실행하세요.")
            st.stop()
        pool = live["pool"]
        c1, c2, c3 = st.columns([2, 1, 1])
        min_dr = c1.slider("최소 불량 다이 비율", 0.0, 0.8, 0.0, 0.05,
                           help="0이면 완전 무작위. 올리면 불량이 많은 웨이퍼만 남깁니다.")
        n_pick = c2.number_input("장수", 1, 25, 6)
        # 문법 설명: st.session_state
        # Streamlit은 버튼을 누를 때마다 스크립트를 처음부터 다시 실행해서 일반 변수는
        # 매번 초기화된다. session_state는 새로고침 전까지 값이 유지되는 저장소라,
        # '다시 뽑기'를 누른 횟수를 여기에 담아 난수 시드로 쓴다.
        if "live_seed" not in st.session_state:
            st.session_state["live_seed"] = 0
        if c3.button("다시 뽑기"):
            st.session_state["live_seed"] += 1
        dr = np.array(pool["defect_ratio"])
        cand = np.flatnonzero(dr >= min_dr)
        if len(cand) == 0:
            st.warning("조건에 맞는 웨이퍼가 없습니다.")
            st.stop()
        rng = np.random.default_rng(st.session_state["live_seed"])
        picks = rng.choice(cand, size=min(int(n_pick), len(cand)), replace=False)
        items = [(f"{pool['lot'][i]} · slot {pool['wafer_index'][i]}", pool["maps"][i]) for i in picks]
        st.caption(
            f"원본 81만 장 중 **라벨이 없는 638,507장**에서 무작위로 뽑아 둔 {len(pool['maps']):,}장 중 "
            f"조건에 맞는 {len(cand):,}장에서 추출합니다. 원 연구자도 답을 붙이지 않았고, "
            "12,763장 학습 샘플에도 들어간 적이 없어 5개 모델 모두 처음 보는 웨이퍼입니다."
        )

    for name, err in errors:
        st.error(f"**{name}** — {err}")
    if not items:
        st.info("판정할 웨이퍼를 넣어주세요.")
        st.stop()

    # ---- 판정 ----
    names = [n for n, _ in items]
    maps = [m for _, m in items]
    pr = predict_probs(maps, models5)          # (5개 모델, 웨이퍼 수, 9)
    # 대표 판정은 fold 0 모델로 낸다 — 화면 1과 같은 모델이고, '격차 20%p' 기준의
    # 오분류율이 이 모델의 검증셋에서 실측돼 있기 때문이다. 나머지 4개는 참고용 일치도.
    p0 = pr[0]
    pred = p0.argmax(1)
    srt = np.sort(p0, axis=1)[:, ::-1]
    gaps = srt[:, 0] - srt[:, 1]
    votes = (pr.argmax(2) == pred[None, :]).sum(0)
    warns = [input_warnings(m, ref) if ref else [] for m in maps]
    summ = [map_summary(m) for m in maps]

    res = pd.DataFrame({
        "웨이퍼": names,
        "원본 크기": [f"{s['shape'][0]}×{s['shape'][1]}" for s in summ],
        "불량 다이 비율(%)": [s["defect_ratio"] * 100 for s in summ],
        "판정": [class_names[k] for k in pred],
        "확률(%)": p0.max(1) * 100,
        "1-2등 격차(%p)": gaps * 100,
        "5개 모델 일치": [f"{v}/5" for v in votes],
        # 기준은 src/14_heldout_eval.py에서 본 적 없는 라벨 웨이퍼로 검증했다 —
        # 격차 기준만으로는 정상 웨이퍼 오경보를 거의 못 걸러서 만장일치 기준을 더했다.
        # 입력 경고는 이 기준에 넣지 않는다 — 검증되지 않은 기준을 섞으면 캡션의 수치가 거짓이 된다.
        "재확인 권장": ["예" if (v < 5 or g < 0.20) else "" for v, g in zip(votes, gaps)],
        "입력 경고": [f"{len(w)}건" if w else "" for w in warns],
    })

    st.divider()
    if len(items) > 1:
        st.markdown(f"#### 판정 요약 — {len(items)}장")
        cnt = res["판정"].value_counts()
        s1, s2 = st.columns([1, 2])
        with s1:
            is_def = res["판정"] != "none"
            k1, k2, k3 = st.columns(3)
            k1.metric("불량 판정", f"{int(is_def.sum())}장")
            k2.metric("그중 만장일치", f"{int((is_def & (votes == 5)).sum())}장")
            k3.metric("재확인 권장", f"{int((res['재확인 권장'] == '예').sum())}장")
            st.caption(held["flag_caption"] if held is not None else
                       "재확인 권장 = 5개 모델 만장일치가 아니거나 1-2등 격차 20%p 미만인 웨이퍼")
        with s2:
            fig_c = go.Figure(go.Bar(
                x=cnt.index.tolist(), y=cnt.values,
                marker_color=["#c9ccd1" if c == "none" else "#c44e52" for c in cnt.index],
                text=cnt.values, textposition="outside", cliponaxis=False))
            fig_c.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                                yaxis=dict(title="장수"))
            st.plotly_chart(fig_c, width='stretch', config={"displayModeBar": False})

        st.dataframe(res, hide_index=True, width='stretch', column_config={
            "불량 다이 비율(%)": st.column_config.NumberColumn(format="%.1f"),
            "확률(%)": st.column_config.NumberColumn(format="%.1f"),
            "1-2등 격차(%p)": st.column_config.NumberColumn(format="%.1f"),
        })
        st.download_button("판정 결과 CSV 내려받기", res.to_csv(index=False).encode("utf-8-sig"),
                           file_name="wafer_judgement.csv", mime="text/csv")

        # 넣은 순서대로 격자 — Lot 안에서 같은 판정이 몰리는지 한눈에 보이게
        with st.expander("웨이퍼 맵 격자로 보기"):
            _grid([resize_nearest(m, 64) for m in maps],
                  [class_names[k] for k in pred], [0] * len(maps), cols=5)
        st.divider()

    sel = st.selectbox(
        "자세히 볼 웨이퍼", range(len(items)),
        format_func=lambda i: f"{names[i]} — {class_names[pred[i]]} {p0[i].max()*100:.1f}%"
                              + ("  (재확인 권장)" if res["재확인 권장"][i] else ""),
    )
    m = maps[sel]
    pred_lab = class_names[pred[sel]]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("판정", pred_lab)
    d2.metric("확률", f"{p0[sel].max()*100:.1f}%")
    d3.metric("1-2등 격차", f"{gaps[sel]*100:.1f}%p")
    d4.metric("5개 모델 일치", f"{votes[sel]}/5")
    for w in warns[sel]:
        st.warning(w)
    if votes[sel] < 5:
        err_txt = ""
        if held is not None and int(votes[sel]) in held["votes_error"]:
            err_txt = (f" 본 적 없는 라벨 웨이퍼에서 {votes[sel]}/5 일치였던 판정은 "
                       f"{held['votes_error'][int(votes[sel])]*100:.0f}%가 오분류였습니다.")
        st.warning(f"5개 모델 중 {votes[sel]}개만 이 판정에 동의했습니다.{err_txt} 사람이 재확인할 대상입니다.")

    x_t = torch.from_numpy(to_model_input(m)[None, None, :, :])
    heatmap, _, _ = gradcam.generate(x_t, target_class=int(pred[sel]))
    img64 = resize_nearest(m, 64)
    v1, v2, v3 = st.columns(3)
    with v1:
        st.plotly_chart(wafer_heatmap_fig(m, f"원본 ({m.shape[0]}×{m.shape[1]})"),
                        width='stretch', key="live_orig")
    with v2:
        st.plotly_chart(wafer_heatmap_fig(img64, "모델이 본 입력 (64×64)"),
                        width='stretch', key="live_64")
    with v3:
        st.plotly_chart(gradcam_fig(img64, heatmap, f"Grad-CAM (근거: {pred_lab})"),
                        width='stretch', key="live_cam")
    st.caption(
        "가운데는 최근접 이웃 방식으로 64×64로 줄인 실제 모델 입력입니다. 원본이 크면 가는 선이 "
        "끊겨 보일 수 있습니다 — Scratch 성능 한계(F1 0.697)의 원인으로 추정한 바로 그 단계입니다."
    )

    st.markdown("##### 9개 클래스 전체 예측 확률")
    fig_p, names_sorted, vals_sorted = prob_bar_fig(p0[sel], pred_lab)
    st.plotly_chart(fig_p, width='stretch', key="live_prob")
    n_narrow, err_narrow, err_wide = gap_reference()
    msg = (f"1등 **{names_sorted[0]}** {vals_sorted[0]*100:.1f}% · "
           f"2등 **{names_sorted[1]}** {vals_sorted[1]*100:.1f}% · 격차 **{gaps[sel]*100:.1f}%p**")
    if gaps[sel] < 0.20:
        st.warning(msg + f" — 격차가 좁습니다. 검증셋에서 이런 웨이퍼는 {err_narrow*100:.0f}%가 "
                         f"오분류였습니다(넓은 경우 {err_wide*100:.0f}%).")
    else:
        st.info(msg)
    st.caption("라벨이 없는 웨이퍼이므로 정답 막대(초록)는 없고, 이 판정이 맞았는지는 이 화면에서 알 수 없습니다.")

    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("##### 5개 fold 모델의 판정")
        st.dataframe(pd.DataFrame({
            "모델": [f"fold {k}" for k in range(5)],
            "판정": [class_names[pr[k, sel].argmax()] for k in range(5)],
            "확률(%)": [pr[k, sel].max() * 100 for k in range(5)],
        }), hide_index=True, width='stretch',
            column_config={"확률(%)": st.column_config.NumberColumn(format="%.1f")})
        if held is not None:
            st.caption(held["votes_caption"])
        else:
            st.caption("5개 모델은 서로 다른 80%로 학습됐습니다. 이 일치도가 오분류를 가려내는지는 따로 검증하지 않았습니다.")
    with c_r:
        st.markdown("##### 문헌상 공정 모듈")
        if pred_lab == "none":
            st.success("정상 판정 — 공정 조치 대상이 아닙니다.")
        else:
            mod, cause, action = PROCESS_MAP.get(pred_lab, ("-", "-", "-"))
            st.markdown(f"**공정 모듈** {mod}\n\n**물리적 원인** {cause}\n\n**점검 방향** {action}")
            st.caption("근거 문헌과 출처는 '공정 원인 매핑' 화면에 있습니다.")
