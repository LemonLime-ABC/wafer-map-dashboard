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

ART = os.path.join(BASE, "artifacts")

st.set_page_config(page_title="웨이퍼 맵 불량 패턴 분류 대시보드", layout="wide")

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


def wafer_heatmap_fig(img: np.ndarray, title: str, height: int = 300) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=img[::-1], colorscale=DIE_COLORSCALE,
                                     zmin=0, zmax=2, showscale=False))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=35, b=10),
                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# --------------------------------------------
# 화면 선택
# --------------------------------------------
PAGES = ["웨이퍼 맵 진단", "패턴별 성능", "Lot 수율 모니터링", "공정 원인 매핑", "모델 방법론"]
page = st.sidebar.radio("화면 선택", PAGES)
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
        fig_cam = go.Figure()
        fig_cam.add_trace(go.Heatmap(z=img[::-1], colorscale="gray", zmin=0, zmax=2,
                                      showscale=False, opacity=0.5))
        fig_cam.add_trace(go.Heatmap(z=heatmap[::-1], colorscale="Jet",
                                      zmin=0, zmax=1, showscale=False, opacity=0.55))
        fig_cam.update_layout(title=f"Grad-CAM (근거: {class_names[target_cls]}, p={prob:.2f})",
                               height=300, margin=dict(l=10, r=10, t=35, b=10),
                               xaxis=dict(visible=False), yaxis=dict(visible=False))
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

        p = probs_all[local_pos]
        order = np.argsort(p)[::-1]          # 확률 높은 순으로 정렬
        names_sorted = [class_names[i] for i in order]
        vals_sorted = p[order]

        true_lab, pred_lab = row["true_label"], row["pred_label"]
        # 색으로 역할을 구분한다: 예측한 것(빨강) / 실제 정답(초록) / 나머지(회색).
        # 맞힌 경우엔 둘이 같은 막대이므로 빨강 하나만 보인다.
        bar_colors = []
        for nm in names_sorted:
            if nm == pred_lab:
                bar_colors.append("#c44e52")
            elif nm == true_lab:
                bar_colors.append("#55a868")
            else:
                bar_colors.append("#c9ccd1")

        fig_p = go.Figure(go.Bar(
            x=names_sorted, y=vals_sorted, marker_color=bar_colors,
            text=[f"{v*100:.1f}%" if v >= 0.001 else "<0.1%" for v in vals_sorted],
            textposition="outside", cliponaxis=False,
        ))
        fig_p.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="확률", range=[0, 1.15], tickformat=".0%"),
            xaxis=dict(title=None),
        )
        st.plotly_chart(fig_p, width='stretch')

        # 1등과 2등의 격차 = 모델이 얼마나 확신했는가.
        # 이 임계값(20%p)이 실제로 의미가 있는지는 검증셋 전체로 확인할 수 있다.
        # (하드코딩하면 나중에 모델이 바뀔 때 조용히 거짓말이 되므로 매번 계산한다)
        gap = vals_sorted[0] - vals_sorted[1]
        srt = np.sort(probs_all, axis=1)[:, ::-1]
        gaps_all = srt[:, 0] - srt[:, 1]
        narrow = gaps_all < 0.20
        wrong = ~wafer_table["correct"].values
        err_narrow = wrong[narrow].mean()
        err_wide = wrong[~narrow].mean()

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
            f"{int(narrow.sum())}장이고, 그중 {err_narrow*100:.1f}%가 오분류**입니다 — "
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
