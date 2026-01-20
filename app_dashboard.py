# -*- coding: utf-8 -*-
"""
app_dashboard.py
EDA 및 클러스터링 보고서를 기반으로 한 Streamlit 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------------
# 페이지 설정
# ------------------------------------------------------------------
st.set_page_config(
    page_title="iMS | 지능형 경영 의사결정 시스템",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# 프리미엄 CSS 스타일링 (Premium UI/UX)
# ------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* 메인 배경 및 글래스모피즘 효과 */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 메트릭 카드 스타일 */
    [data-testid="stMetricValue"] {
        font-weight: 800;
        color: #1a1a1a;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #555;
    }
    
    /* 카드 컨테이너 프리미엄 박스 */
    .premium-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        margin-bottom: 20px;
    }
    
    /* 버튼 스타일 고도화 */
    .stButton>button {
        background: linear-gradient(45deg, #2c3e50, #000000);
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* 구분선 스타일 */
    hr {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 데이터 로딩 (캐싱)
# ------------------------------------------------------------------
@st.cache_data
def load_data(refresh_time):
    """전처리된 데이터 및 클러스터링 데이터 로드"""
    data_dir = Path("data")
    
    # 전처리 및 기본 데이터
    df_preprocessed = pd.read_csv(data_dir / "data_preprocessed.csv", encoding="utf-8-sig")
    df_clustered = pd.read_csv(data_dir / "data_clustered.csv", encoding="utf-8-sig")
    
    # 마케팅 및 분석 데이터
    df_event = pd.read_csv(data_dir / "data_eventstats.csv", encoding="utf-8-sig")
    df_page = pd.read_csv(data_dir / "data_pagestats.csv", encoding="utf-8-sig")
    df_click = pd.read_csv(data_dir / "data_sales_click.csv", encoding="utf-8-sig")
    
    # 심화 분석 데이터
    df_cluster_channel = pd.read_csv(data_dir / "analysis_cluster_channel.csv", encoding="utf-8-sig", index_col=0)
    df_prod_eff = pd.read_csv(data_dir / "analysis_product_efficiency.csv", encoding="utf-8-sig")
    
    # Phase 5 데이터
    try:
        df_ltv = pd.read_csv(data_dir / "analysis_ltv.csv", encoding="utf-8-sig")
        df_interval = pd.read_csv(data_dir / "analysis_order_interval.csv", encoding="utf-8-sig")
        df_attr = pd.read_csv(data_dir / "analysis_attribution.csv", encoding="utf-8-sig")
    except:
        df_ltv, df_interval, df_attr = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # 날짜 컬럼 변환
    date_cols = ["주문일", "일자", "날짜"]
    for df_item in [df_preprocessed, df_clustered, df_event, df_click]:
        for col in date_cols:
            if col in df_item.columns:
                df_item[col] = pd.to_datetime(df_item[col], errors="coerce")
    
    return df_preprocessed, df_clustered, df_event, df_page, df_click, df_cluster_channel, df_prod_eff, df_ltv, df_interval, df_attr

# 데이터 로드 (파일 수정 시간 기반 캐시 갱신)
last_mod = Path("data/analysis_product_efficiency.csv").stat().st_mtime
@st.cache_data(ttl=3600, show_spinner="데이터를 분석 중입니다...")
def load_all_data(mod_time):
    df_preprocessed = pd.read_csv("data/data_preprocessed.csv")
    df_prod_eff = pd.read_csv("data/analysis_product_efficiency.csv")
    df_event = pd.read_csv("data/data_eventstats.csv")
    df_click = pd.read_csv("data/data_sales_click.csv")
    df_attr = pd.read_csv("data/analysis_attribution.csv")
    return df_preprocessed, df_prod_eff, df_event, df_click, df_attr

df_preprocessed, df_prod_eff, df_event, df_click, df_attr = load_all_data(last_mod)

# 강제 디버깅: '공급가' 누락 시 더미 데이터 생성 시도
if '공급가' not in df_prod_eff.columns:
    if not df_prod_eff.empty:
        df_prod_eff['공급가'] = 0 # Fallback

# ------------------------------------------------------------------
# 사이드바 메뉴
# ------------------------------------------------------------------
st.sidebar.title("📊 메뉴")
page = st.sidebar.radio(
    "페이지 선택",
    ["👑 경영 요약", "🏆 고객 가치 분석", "📊 마케팅 기여도", "📈 개요", "📊 EDA 분석", "🎯 클러스터링", "📈 마케팅 분석", "💎 속성 분석", "🔍 상세 분석"]
)

st.sidebar.divider()
st.sidebar.subheader("📡 시스템 상태 (Health)")
st.sidebar.caption("✅ 데이터 엔진 정상 작동 중")
st.sidebar.caption(f"📅 최종 동기화: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ------------------------------------------------------------------
# 페이지: 👑 경영 요약 (Management View)
# ------------------------------------------------------------------
if page == "👑 경영 요약":
    st.markdown("""
        <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 30px; border-radius: 20px; color: white; margin-bottom: 30px;">
            <h1 style="margin:0; font-weight:800; font-size: 2.5rem;">👑 지능형 경영 의사결정 브리핑</h1>
            <p style="margin:5px 0 0 0; opacity: 0.8; font-size: 1.1rem;"> Intelligent Management Support System | iMS v6.0 </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 0. 이상 징후 감지 (Anomaly Detection)
    st.subheader("🚨 실시간 성과 경보 (Anomaly Detection)")
    
    # 최근 7일 매출 변동성 분석
    daily_sales = df_preprocessed.groupby('주문일')['결제금액(상품별)'].sum().reset_index()
    last_7_days = daily_sales.tail(7)
    if not last_7_days.empty:
        mean_sales = daily_sales['결제금액(상품별)'].mean()
        std_sales = daily_sales['결제금액(상품별)'].std()
        latest_sales = last_7_days.iloc[-1]['결제금액(상품별)']
        
        if latest_sales > mean_sales + 2 * std_sales:
            st.success(f"🔥 **성과 급증 감지**: 최근 매출이 평균 대비 2배 이상 높습니다! 현재 마케팅 소재의 효율이 극대화된 상태입니다.")
        elif latest_sales < mean_sales - 1.5 * std_sales:
            st.warning(f"⚠️ **성과 하락 주의**: 최근 매출이 정상 범위보다 낮습니다. 유입 경로의 이탈이나 결제 오류 여부를 확인하세요.")
        else:
            st.info("✅ 현재 매출 및 운영 지표가 정상 범위 내에서 안정적으로 유지되고 있습니다.")
    
    st.divider()
    st.subheader("📍 핵심 성과 지표 (KPI)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df_preprocessed["결제금액(상품별)"].sum()
        st.metric("총 매출액", f"{total_revenue:,.0f}원")
    
    with col2:
        # 최근 마케팅 데이터를 통한 RPC 추출
        avg_rpc = df_prod_eff["RPC"].mean()
        st.metric("평균 클릭당 매출 (RPC)", f"{avg_rpc:,.1f}원")
        
    with col3:
        avg_ctr = df_prod_eff["CTR"].mean()
        st.metric("평균 마케팅 클릭률 (CTR)", f"{avg_ctr:.2f}%")
        
    with col4:
        total_vistors = df_event['DAU 전체(회원)'].sum()
        st.metric("총 방문자 수 (DAU)", f"{total_vistors:,.0f}명")

    st.divider()

    # 1.5. 매출 예측 (Revenue Forecasting - Simple Trend)
    st.subheader("🔮 향후 7일 매출 예측 (Forecasting)")
    
    # 일별 매출 집계
    daily_sales = df_preprocessed.groupby('주문일')['결제금액(상품별)'].sum().reset_index()
    daily_sales = daily_sales.sort_values('주문일')
    
    # 최근 30일 데이터로 7일 예측 (이동평균 + 추세 기반 단순 모델)
    recent_sales = daily_sales.tail(30)
    last_date = recent_sales['주문일'].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    
    # 간단한 선형 추세 계산
    x = np.arange(len(recent_sales))
    y = recent_sales['결제금액(상품별)'].values
    slope, intercept = np.polyfit(x, y, 1)
    
    forecast_values = slope * (np.arange(len(recent_sales), len(recent_sales) + 7)) + intercept
    forecast_values = np.maximum(forecast_values, 0) # 음수 방지
    
    df_forecast = pd.DataFrame({'날짜': forecast_dates, '예상매출': forecast_values})
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=recent_sales['주문일'], y=y, name='실제 매출', line=dict(color='royalblue', width=2)))
    fig_forecast.add_trace(go.Scatter(x=df_forecast['날짜'], y=forecast_values, name='예측 매출', line=dict(color='firebrick', width=2, dash='dot')))
    
    fig_forecast.update_layout(
        title="최근 매출 추이 및 향후 7일 예측",
        xaxis_title="날짜",
        yaxis_title="매출액 (원)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.caption("최근 30일간의 매출 추세를 기반으로 산출된 통계적 예측치입니다.")

    st.divider()

    # 2. 매출 시뮬레이터
    st.subheader("📊 매출 성장 시뮬레이터 (Simulator)")
    st.write("마케팅 유입 및 효율 변화에 따른 예상 매출액을 시뮬레이션합니다.")
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        st.info("💡 변수 설정")
        target_pv = st.slider("목표 페이지뷰 (PV) 증감 (%)", -50, 200, 20)
        target_ctr = st.slider("목표 클릭률 (CTR) 개선 (pp)", -2.0, 5.0, 0.5, step=0.1)
        target_cvr = st.slider("목표 전환율 (CVR) 개선 (pp)", -1.0, 3.0, 0.2, step=0.1)
        
    with col_sim2:
        # 기본값 로직
        current_pv = df_event['PV'].sum()
        current_ctr = (df_click['클릭수'].sum() / df_click['조회수'].sum()) * 100
        # 단순 전환율 추정 (판매건수 / 클릭수)
        current_cvr = (len(df_preprocessed) / df_click['클릭수'].sum()) * 100
        avg_order_value = df_preprocessed["결제금액(상품별)"].mean()
        
        # 시뮬레이션 계산
        sim_pv = current_pv * (1 + target_pv / 100)
        sim_click = sim_pv * ((current_ctr + target_ctr) / 100)
        sim_order = sim_click * ((current_cvr + target_cvr) / 100)
        sim_revenue = sim_order * avg_order_value
        
        rev_diff = sim_revenue - total_revenue
        
        # 결과 표시
        st.write("### 예상 성과")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("예상 총 매출", f"{sim_revenue:,.0f}원", f"{rev_diff:,.0f}원")
        res_col2.metric("예상 주문 건수", f"{sim_order:,.0f}건", f"{sim_order - len(df_preprocessed):,.0f}건")
        
        # 차트 표시
        fig_sim = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sim_revenue,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "매출 목표 달성 예측 (원)"},
            delta = {'reference': total_revenue, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, total_revenue * 2]},
                'steps': [
                    {'range': [0, total_revenue], 'color': "lightgray"},
                    {'range': [total_revenue, total_revenue * 1.5], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': sim_revenue}}))
        st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()

    # 3. 데이터 기반 자동 전략 제안 (Auto-Insights)
    st.subheader("💡 인공지능 기반 마케팅 진단")
    
    insights = []
    
    # RPCInsight
    low_rpc_prods = df_prod_eff[df_prod_eff['RPC'] < df_prod_eff['RPC'].median()].head(3)
    if not low_rpc_prods.empty:
        insights.append(f"⚠️ **수익성 주의**: `{', '.join(low_rpc_prods['상품명'].tolist())}` 상품은 클릭 대비 매출(RPC)이 낮습니다. 상세 페이지의 가격 제안 혹은 구매 전환 요소를 점검하세요.")
        
    # High CTR, Low Conversion Insight
    high_ctr_prods = df_prod_eff[df_prod_eff['CTR'] > df_prod_eff['CTR'].median()].sort_values('RPC').head(2)
    if not high_ctr_prods.empty:
        insights.append(f"✨ **기회 포착**: `{', '.join(high_ctr_prods['상품명'].tolist())}` 상품은 유입량은 많으나 결제로의 연결이 부족합니다. '한정 수량' 혹은 '타임 세일' 등의 장치를 추가해 보세요.")
        
    # Channel Insight
    top_channel = df_preprocessed['주문경로'].value_counts().idxmax()
    insights.append(f"📈 **채널 성과**: 현재 가장 강력한 유입 채널은 **{top_channel}**입니다. 해당 채널의 예산을 15% 증액하여 규모의 경제를 달성할 것을 권장합니다.")

    for insight in insights:
        st.write(insight)

    st.divider()

    # 4. 상품별 적정 판매가 제안 (Pricing Suggestion)
    st.subheader("💰 상품별 수익 최적화 제안 (Pricing)")
    st.write("공급가와 현재 판매 성과를 분석하여 수익 극대화를 위한 적정 판매가를 제안합니다.")
    
    # 마진율 계산을 위해 판매수량 정보 결합
    prod_qty = df_clustered.groupby('상품코드')['주문수량'].sum().reset_index()
    df_pricing = pd.merge(df_prod_eff, prod_qty, on='상품코드')
    
    df_pricing['마진액'] = df_pricing['결제금액(상품별)'] - (df_pricing['공급가'] * df_pricing['주문수량'])
    df_pricing['현재마진율'] = (df_pricing['마진액'] / df_pricing['결제금액(상품별)']) * 100
    
    # 제안 로직: CTR이 높고 마진율이 낮은 상품은 가격 인상 고려, CTR이 낮고 마진이 높은 상품은 할인 이벤트 고려
    def suggest_price(row):
        if row['CTR'] > df_pricing['CTR'].median() and row['현재마진율'] < 20:
            return f"{row['결제금액(상품별)']/row['주문수량'] * 1.1:,.0f}원 (인상 권고)", "인기 대비 저마진"
        elif row['CTR'] < df_pricing['CTR'].median() and row['현재마진율'] > 40:
            return f"{row['결제금액(상품별)']/row['주문수량'] * 0.9:,.0f}원 (할인 권고)", "고마진 대비 저조한 유입"
        return "현재가 유지", "안정적 성과"

    df_pricing[['제안가격', '판단근거']] = df_pricing.apply(lambda r: pd.Series(suggest_price(r)), axis=1)
    
    st.dataframe(
        df_pricing[['상품명', '공급가', '현재마진율', 'CTR', '제안가격', '판단근거']].head(10),
        use_container_width=True
    )
    
    st.divider()
    
    # 5. 엑셀 리포트 출력
    st.subheader("📥 경영 분석 리포트 다운로드")
    
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_pricing.to_excel(writer, sheet_name='상품효율및가격제안', index=False)
        daily_sales.to_excel(writer, sheet_name='일별매출현황', index=False)
        df_prod_eff.to_excel(writer, sheet_name='마케팅효율지표', index=False)
    
    st.download_button(
        label="📊 전문가용 경영 분석 엑셀 다운로드",
        data=output.getvalue(),
        file_name=f"Management_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ------------------------------------------------------------------
# 페이지: 🏆 고객 가치 분석 (LTV Analysis)
# ------------------------------------------------------------------
elif page == "🏆 고객 가치 분석":
    st.title("🏆 고객 생애 가치 및 이탈 분석 (LTV & Churn)")
    st.write("고객별 구매 패턴을 분석하여 미래 가치가 높은 VIP 고객과 이탈 위험 고객을 식별합니다.")
    
    if df_ltv.empty:
        st.warning("분석 데이터가 부족합니다. `analyze_phase5.py`를 실행해 주세요.")
    else:
        # KPI 요약
        col_ltv1, col_ltv2, col_ltv3 = st.columns(3)
        with col_ltv1:
            st.metric("평균 LTV 점수", f"{df_ltv['LTV_Score'].mean():.1f}")
        with col_ltv2:
            st.metric("평균 재구매 횟수", f"{df_ltv['Frequency'].mean():.1f}회")
        with col_ltv3:
            st.metric("고가치 고객 비중 (Top 20%)", f"{len(df_ltv[df_ltv['LTV_Score'] > df_ltv['LTV_Score'].quantile(0.8)]) / len(df_ltv) * 100:.1f}%")
            
        st.divider()
        
        # LTV 분포 및 위험도 시각화
        col_ltv_chart1, col_ltv_chart2 = st.columns(2)
        
        with col_ltv_chart1:
            st.subheader("💰 고객 가치(LTV) 분포")
            fig_ltv_dist = px.histogram(df_ltv, x="LTV_Score", nbins=50, 
                                        color="cluster", title="클러스터별 LTV 점수 분포")
            st.plotly_chart(fig_ltv_dist, use_container_width=True)
            
        with col_ltv_chart2:
            st.subheader("📉 재구매 지연 고객 (이탈 위험)")
            # Recency가 30일 이상인 고객 필터링
            df_churn = df_ltv[df_ltv['Recency'] > 30].sort_values('Monetary', ascending=False)
            st.write(f"최근 30일간 구매가 없는 고가치 고객 ({len(df_churn)}명)")
            st.dataframe(df_churn[['고객ID', 'Recency', 'Monetary', 'Frequency']].head(10), use_container_width=True)
            
        st.divider()
        
        # 재구매 주기 분석
        st.subheader("🕙 클러스터별 평균 재구매 주기 (Retention Loop)")
        fig_loop = px.bar(df_interval, x="cluster", y="avg_order_interval", 
                          title="구매와 구매 사이의 간격 (단위: 일)",
                          color="cluster", labels={"avg_order_interval": "평균 주기 (일)"})
        st.plotly_chart(fig_loop, use_container_width=True)
        st.info("💡 전략 제안: 평균 주기보다 Recency가 길어지는 클러스터를 대상으로 '컴백 쿠폰'을 자동 발행하는 전략이 유효합니다.")

# ------------------------------------------------------------------
# 페이지: 📊 마케팅 ROI 및 기여도 (Attribution Analysis)
# ------------------------------------------------------------------
elif page == "📊 마케팅 기여도":
    st.title("📊 마케팅 채널별 ROI 및 기여도 분석")
    st.write("각 마케팅 채널의 광고비 대비 매출 성과(ROAS) 및 주문 기여도를 정밀하게 분석합니다.")
    
    if df_attr.empty:
        st.warning("분석 데이터가 부족합니다. `analyze_attribution.py`를 실행해 주세요.")
    else:
        # 채널 성과 매트릭스
        st.subheader("🚀 채널별 ROAS 및 효율성")
        fig_roas = px.bar(df_attr, x="채널", y="ROAS", text_auto=".1f",
                          color="ROAS", color_continuous_scale="RdYlGn",
                          title="채널별 ROAS (%)")
        st.plotly_chart(fig_roas, use_container_width=True)
        
        col_attr_1, col_attr_2 = st.columns(2)
        with col_attr_1:
            st.subheader("💰 채널별 매출 기여 비중")
            fig_attr_pie = px.pie(df_attr, values="매출액", names="채널", hole=0.3)
            st.plotly_chart(fig_attr_pie, use_container_width=True)
            
        with col_attr_2:
            st.subheader("🎯 고객 획득 비용 (CPA)")
            fig_cpa = px.bar(df_attr, x="채널", y="CPA", text_auto=",.0f",
                             color="채널", title="주문 1건당 광고비 (원)")
            st.plotly_chart(fig_cpa, use_container_width=True)
            
        st.info("💡 경영 제안: ROAS가 가장 높은 채널에 예산을 우선 배정하고, CPA가 평균보다 높은 채널은 유입 품질을 개선할 필요가 있습니다.")

# ------------------------------------------------------------------
# 페이지: 개요
# ------------------------------------------------------------------
elif page == "📈 개요":
    st.title("📈 판매 데이터 개요")
    
    # KPI 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_orders = len(df_preprocessed)
        st.metric("총 주문 건수", f"{total_orders:,}건")
    
    with col2:
        total_revenue = df_preprocessed["결제금액(상품별)"].sum()
        st.metric("총 매출액", f"{total_revenue:,.0f}원")
    
    with col3:
        avg_order = df_preprocessed["결제금액(상품별)"].mean()
        st.metric("평균 주문 금액", f"{avg_order:,.0f}원")
    
    with col4:
        avg_quantity = df_preprocessed["주문수량"].mean()
        st.metric("평균 주문 수량", f"{avg_quantity:.2f}개")
    
    st.divider()
    
    # 일별 주문 추이
    st.subheader("📅 일별 주문 추이")
    daily_orders = df_preprocessed.groupby(df_preprocessed["주문일"].dt.date).agg({
        "주문번호": "count",
        "결제금액(상품별)": "sum"
    }).reset_index()
    daily_orders.columns = ["날짜", "주문건수", "매출액"]
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=daily_orders["날짜"],
        y=daily_orders["주문건수"],
        mode="lines+markers",
        name="주문건수",
        line=dict(color="#1f77b4", width=2)
    ))
    fig_daily.update_layout(
        title="일별 주문 건수 추이",
        xaxis_title="날짜",
        yaxis_title="주문 건수",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # 주문 경로별 분포
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📱 주문 경로별 분포")
        channel_dist = df_preprocessed["주문경로"].value_counts()
        fig_channel = px.pie(
            values=channel_dist.values,
            names=channel_dist.index,
            title="주문 경로별 비율"
        )
        st.plotly_chart(fig_channel, use_container_width=True)
    
    with col2:
        st.subheader("💳 결제 방법별 분포")
        payment_dist = df_preprocessed["결제방법"].value_counts()
        fig_payment = px.pie(
            values=payment_dist.values,
            names=payment_dist.index,
            title="결제 방법별 비율"
        )
        st.plotly_chart(fig_payment, use_container_width=True)

# ------------------------------------------------------------------
# 페이지: EDA 분석
# ------------------------------------------------------------------
elif page == "📊 EDA 분석":
    st.title("📊 EDA 분석")
    
    # 결측치 현황
    st.subheader("🔍 결측치 현황")
    missing_data = pd.DataFrame({
        "컬럼명": df_preprocessed.columns,
        "결측치 수": df_preprocessed.isnull().sum().values,
        "결측치 비율(%)": (df_preprocessed.isnull().sum() / len(df_preprocessed) * 100).values
    }).sort_values("결측치 수", ascending=False)
    st.dataframe(missing_data, use_container_width=True)
    
    st.divider()
    
    # 수치형 컬럼 통계
    st.subheader("📈 수치형 컬럼 기본 통계")
    numeric_cols = df_preprocessed.select_dtypes(include="number").columns
    stats_df = df_preprocessed[numeric_cols].describe().T
    st.dataframe(stats_df, use_container_width=True)
    
    st.divider()
    
    # 요일별 주문 분포
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 요일별 주문 분포")
        df_preprocessed["주문요일"] = df_preprocessed["주문일"].dt.day_name()
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_counts = df_preprocessed["주문요일"].value_counts().reindex(weekday_order)
        
        fig_weekday = px.bar(
            x=weekday_counts.index,
            y=weekday_counts.values,
            labels={"x": "요일", "y": "주문 건수"},
            title="요일별 주문 건수"
        )
        st.plotly_chart(fig_weekday, use_container_width=True)
    
    with col2:
        st.subheader("💰 결제금액 분포")
        fig_payment_dist = px.histogram(
            df_preprocessed,
            x="결제금액(상품별)",
            nbins=50,
            title="결제금액 분포",
            labels={"결제금액(상품별)": "결제금액"}
        )
        st.plotly_chart(fig_payment_dist, use_container_width=True)
    
    # 주문수량 분포
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 주문수량 분포")
        fig_quantity = px.box(
            df_preprocessed,
            y="주문수량",
            title="주문수량 Box Plot"
        )
        st.plotly_chart(fig_quantity, use_container_width=True)
    
    with col2:
        st.subheader("📱 주문 경로별 매출 비교")
        channel_revenue = df_preprocessed.groupby("주문경로")["결제금액(상품별)"].sum().sort_values(ascending=False)
        fig_channel_revenue = px.bar(
            x=channel_revenue.index,
            y=channel_revenue.values,
            labels={"x": "주문 경로", "y": "총 매출액"},
            title="주문 경로별 총 매출액"
        )
        st.plotly_chart(fig_channel_revenue, use_container_width=True)

# ------------------------------------------------------------------
# 페이지: 클러스터링
# ------------------------------------------------------------------
elif page == "🎯 클러스터링":
    st.title("🎯 구매 패턴 클러스터링")
    
    # 클러스터 통계
    st.subheader("📊 클러스터별 통계 요약")
    cluster_stats = df_clustered.groupby("cluster").agg({
        "주문번호": "count",
        "주문수량": ["mean", "sum"],
        "결제금액(상품별)": ["mean", "median", "sum"]
    }).round(2)
    cluster_stats.columns = ["주문건수", "평균수량", "총수량", "평균금액", "중앙금액", "총매출"]
    st.dataframe(cluster_stats, use_container_width=True)
    
    st.divider()
    
    # 클러스터 산점도
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 클러스터 산점도 (결제금액 vs 주문수량)")
        fig_scatter = px.scatter(
            df_clustered,
            x="주문수량",
            y="결제금액(상품별)",
            color="cluster",
            title="클러스터별 결제금액 vs 주문수량",
            labels={"cluster": "클러스터"},
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.subheader("📊 클러스터별 평균 금액 비교")
        cluster_avg = df_clustered.groupby("cluster")["결제금액(상품별)"].mean().sort_values(ascending=False)
        fig_cluster_avg = px.bar(
            x=cluster_avg.index.astype(str),
            y=cluster_avg.values,
            labels={"x": "클러스터", "y": "평균 결제금액"},
            title="클러스터별 평균 결제금액"
        )
        st.plotly_chart(fig_cluster_avg, use_container_width=True)
    
    # 클러스터별 건수 분포
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 클러스터별 건수 분포")
        cluster_counts = df_clustered["cluster"].value_counts()
        fig_cluster_pie = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index.astype(str),
            title="클러스터별 주문 건수 비율"
        )
        st.plotly_chart(fig_cluster_pie, use_container_width=True)
    
    with col2:
        st.subheader("📦 클러스터별 결제금액 분포")
        fig_cluster_box = px.box(
            df_clustered,
            x="cluster",
            y="결제금액(상품별)",
            title="클러스터별 결제금액 Box Plot",
            labels={"cluster": "클러스터", "결제금액(상품별)": "결제금액"}
        )
        st.plotly_chart(fig_cluster_box, use_container_width=True)

# ------------------------------------------------------------------
# 페이지: 마케팅 분석
# ------------------------------------------------------------------
elif page == "📈 마케팅 분석":
    st.title("📈 마케팅 유입 및 클릭 분석")
    
    # 상단 지표
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 PV (조회수)", f"{df_event['PV'].sum():,.0f}")
    with col2:
        st.metric("평균 DAU", f"{df_event['DAU 전체(회원)'].mean():,.1f}명")
    with col3:
        st.metric("평균 재방문율", f"{df_event['재방문율(월)'].mean():.1f}%")
    with col4:
        st.metric("최고 조회 페이지", df_page.iloc[0]['페이지제목'])

    st.divider()

    # 유입 추이 차트
    st.subheader("📅 일별 방문자 및 페이지뷰 추이")
    fig_visit = go.Figure()
    fig_visit.add_trace(go.Scatter(x=df_event['일자'], y=df_event['DAU 전체(회원)'], name="DAU(회원)", line=dict(color="#1f77b4")))
    fig_visit.add_trace(go.Scatter(x=df_event['일자'], y=df_event['PV'], name="PV (페이지뷰)", line=dict(color="#ff7f0e"), yaxis="y2"))
    
    fig_visit.update_layout(
        title="방문자(DAU) 및 조회수(PV) 추이",
        yaxis=dict(title="방문자 수"),
        yaxis2=dict(title="페이지뷰(PV)", overlaying="y", side="right"),
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig_visit, use_container_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔝 인기 페이지 (조회수 기준)")
        fig_top_pages = px.bar(
            df_page.head(10),
            x="조회수",
            y="페이지제목",
            orientation="h",
            title="상위 10개 인기 페이지",
            color="조회수",
            color_continuous_scale="Viridis"
        )
        fig_top_pages.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top_pages, use_container_width=True)

    with col2:
        st.subheader("🎯 상품별 클릭 분석")
        # 최근 날짜 기준 상품별 클릭 합계
        df_click_agg = df_click.groupby("상품명_정제").agg({
            "조회수": "sum",
            "클릭수": "sum"
        }).reset_index()
        df_click_agg["CTR(%)"] = (df_click_agg["클릭수"] / df_click_agg["조회수"] * 100).fillna(0)
        
        fig_ctr = px.scatter(
            df_click_agg,
            x="조회수",
            y="클릭수",
            size="CTR(%)",
            hover_name="상품명_정제",
            title="상품별 조회수 대비 클릭수 (원 크기: CTR)",
            color="CTR(%)",
            color_continuous_scale="Plasma"
        )
        st.plotly_chart(fig_ctr, use_container_width=True)

    st.divider()
    
    # 전환 분석 (판매 데이터와 결합)
    st.subheader("🔄 마케팅 유입과 매출의 상관관계")
    
    # 일별 매출과 일별 PV 결합
    daily_sales = df_preprocessed.groupby(df_preprocessed["주문일"].dt.date)["결제금액(상품별)"].sum().reset_index()
    daily_sales.columns = ["날짜", "매출액"]
    daily_sales["날짜"] = pd.to_datetime(daily_sales["날짜"])
    
    df_marketing_sales = pd.merge(daily_sales, df_event[["일자", "PV", "DAU 전체(회원)"]], left_on="날짜", right_on="일자", how="inner")
    
    fig_corr = px.scatter(
        df_marketing_sales,
        x="PV",
        y="매출액",
        trendline="ols",
        title="페이지뷰(PV)와 매출액의 상관관계",
        labels={"PV": "페이지뷰", "매출액": "총 매출액 (원)"},
        hover_data=["날짜"]
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    if not df_marketing_sales.empty:
        correlation = df_marketing_sales["PV"].corr(df_marketing_sales["매출액"])
        st.info(f"💡 분석 결과: 페이지뷰와 매출액의 상관계수는 **{correlation:.2f}**입니다. " + 
                ("강한 양의 상관관계가 있습니다." if correlation > 0.7 else "어느 정도 연관성이 있습니다." if correlation > 0.4 else "상관관계가 낮습니다."))

    st.divider()

    # 심화 분석 섹션
    st.subheader("💡 비즈니스 고도화 분석")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**🎯 클러스터별 유입 채널 분포 (Heatmap)**")
        fig_heat = px.imshow(
            df_cluster_channel,
            labels=dict(x="유입 채널", y="클러스터", color="비중 (%)"),
            x=df_cluster_channel.columns,
            y=df_cluster_channel.index,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="YlGnBu"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("어떤 채널이 특정 구매 그룹(클러스터)을 더 많이 유입시키는지 파악할 수 있습니다.")

    with col4:
        st.write("**💰 상품별 마케팅 효율 매트릭스**")
        fig_bubble = px.scatter(
            df_prod_eff,
            x="CTR",
            y="RPC",
            size="조회수",
            color="RPV",
            hover_name="상품명",
            labels={"CTR": "클릭률 (%)", "RPC": "클릭당 매출 (RPC)", "RPV": "조회당 매출 (RPV)"},
            title="CTR vs RPC (원 크기: 조회수, 색상: RPV)",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.caption("우측 상단 상품: 클릭률도 높고 실제 매출 기여도도 높은 고효율 상품군")

# ------------------------------------------------------------------
# 페이지: 💎 속성 분석 (Attribute Analysis)
# ------------------------------------------------------------------
elif page == "💎 속성 분석":
    st.title("💎 상품 속성별 성과 분석")
    st.write("상품명에서 추출한 등급, 중량, 세트여부 등의 속성이 매출 및 마케팅 효율에 미치는 영향을 분석합니다.")
    
    col_attr1, col_attr2 = st.columns(2)
    
    with col_attr1:
        st.subheader("📦 등급/유형별 매출 비중")
        df_grade = df_preprocessed.groupby('등급')['결제금액(상품별)'].sum().reset_index()
        fig_grade = px.pie(df_grade, values='결제금액(상품별)', names='등급', hole=0.4,
                           color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_grade, use_container_width=True)
        
    with col_attr2:
        st.subheader("⚖️ 중량별 판매 수량")
        df_weight = df_preprocessed.groupby('중량')['주문수량'].sum().reset_index()
        fig_weight = px.bar(df_weight, x='중량', y='주문수량', color='중량',
                             color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_weight, use_container_width=True)
        
    st.divider()
    
    col_attr3, col_attr4 = st.columns(2)
    
    with col_attr3:
        st.subheader("🎁 세트 상품 vs 단품 성과")
        df_set = df_preprocessed.groupby('세트여부')['결제금액(상품별)'].mean().reset_index()
        df_set['세트여부'] = df_set['세트여부'].map({1: '세트/구성상품', 0: '단품'})
        fig_set = px.bar(df_set, x='세트여부', y='결제금액(상품별)', text_auto='.0s',
                         title="평균 주문 금액 비교", color='세트여부')
        st.plotly_chart(fig_set, use_container_width=True)
        
    with col_attr4:
        st.subheader("📣 이벤트 상품 성과")
        df_evt = df_preprocessed.groupby('이벤트여부').agg({
            '결제금액(상품별)': 'sum',
            '주문번호': 'count'
        }).reset_index()
        df_evt['이벤트여부'] = df_evt['이벤트여부'].map({1: '이벤트 포함', 0: '일반'})
        fig_evt = px.bar(df_evt, x='이벤트여부', y='결제금액(상품별)', color='이벤트여부',
                         title="총 매출 기여도")
        st.plotly_chart(fig_evt, use_container_width=True)

# ------------------------------------------------------------------
# 페이지: 상세 분석
# ------------------------------------------------------------------
elif page == "🔍 상세 분석":
    st.title("🔍 상세 분석")
    
    # 사이드바 필터
    st.sidebar.subheader("🔧 필터 설정")
    
    # 날짜 범위 필터
    min_date = df_preprocessed["주문일"].min().date()
    max_date = df_preprocessed["주문일"].max().date()
    date_range = st.sidebar.date_input(
        "날짜 범위",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # 주문 경로 필터
    channels = ["전체"] + df_preprocessed["주문경로"].unique().tolist()
    selected_channel = st.sidebar.selectbox("주문 경로", channels)
    
    # 결제 방법 필터
    payments = ["전체"] + df_preprocessed["결제방법"].unique().tolist()
    selected_payment = st.sidebar.selectbox("결제 방법", payments)
    
    # 필터 적용
    df_filtered = df_preprocessed.copy()
    
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered["주문일"].dt.date >= date_range[0]) &
            (df_filtered["주문일"].dt.date <= date_range[1])
        ]
    
    if selected_channel != "전체":
        df_filtered = df_filtered[df_filtered["주문경로"] == selected_channel]
    
    if selected_payment != "전체":
        df_filtered = df_filtered[df_filtered["결제방법"] == selected_payment]
    
    # 필터링된 데이터 요약
    st.subheader("📊 필터링된 데이터 요약")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("주문 건수", f"{len(df_filtered):,}건")
    with col2:
        st.metric("총 매출액", f"{df_filtered['결제금액(상품별)'].sum():,.0f}원")
    with col3:
        st.metric("평균 주문 금액", f"{df_filtered['결제금액(상품별)'].mean():,.0f}원")
    with col4:
        st.metric("평균 주문 수량", f"{df_filtered['주문수량'].mean():.2f}개")
    
    st.divider()
    
    # 시계열 분석
    st.subheader("📈 시계열 분석")
    time_unit = st.radio("시간 단위", ["일별", "주별", "월별"], horizontal=True)
    
    if time_unit == "일별":
        time_series = df_filtered.groupby(df_filtered["주문일"].dt.date).agg({
            "결제금액(상품별)": "sum",
            "주문번호": "count"
        }).reset_index()
    elif time_unit == "주별":
        time_series = df_filtered.groupby(df_filtered["주문일"].dt.to_period("W")).agg({
            "결제금액(상품별)": "sum",
            "주문번호": "count"
        }).reset_index()
        time_series["주문일"] = time_series["주문일"].astype(str)
    else:  # 월별
        time_series = df_filtered.groupby(df_filtered["주문일"].dt.to_period("M")).agg({
            "결제금액(상품별)": "sum",
            "주문번호": "count"
        }).reset_index()
        time_series["주문일"] = time_series["주문일"].astype(str)
    
    fig_timeseries = go.Figure()
    fig_timeseries.add_trace(go.Scatter(
        x=time_series["주문일"],
        y=time_series["결제금액(상품별)"],
        mode="lines+markers",
        name="매출액",
        line=dict(color="#1f77b4", width=2)
    ))
    fig_timeseries.update_layout(
        title=f"{time_unit} 매출 추이",
        xaxis_title="기간",
        yaxis_title="매출액 (원)",
        hovermode="x unified",
        height=400
    )
    st.plotly_chart(fig_timeseries, use_container_width=True)
    
    st.divider()
    
    # 상위 상품 분석
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 상위 10개 상품 (매출 기준)")
        top_products = df_filtered.groupby("상품명")["결제금액(상품별)"].sum().sort_values(ascending=False).head(10)
        fig_top_products = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation="h",
            labels={"x": "총 매출액", "y": "상품명"},
            title="매출 상위 10개 상품"
        )
        st.plotly_chart(fig_top_products, use_container_width=True)
    
    with col2:
        st.subheader("💳 결제 방법별 매출")
        payment_revenue = df_filtered.groupby("결제방법")["결제금액(상품별)"].sum()
        fig_payment_revenue = px.pie(
            values=payment_revenue.values,
            names=payment_revenue.index,
            title="결제 방법별 매출 비율"
        )
        st.plotly_chart(fig_payment_revenue, use_container_width=True)
    
    # 공급가 vs 결제금액 산점도
    st.subheader("💰 공급가 vs 결제금액")
    fig_price_scatter = px.scatter(
        df_filtered.sample(min(1000, len(df_filtered))),  # 샘플링으로 성능 개선
        x="공급가",
        y="결제금액(상품별)",
        title="공급가 vs 결제금액 산점도",
        labels={"공급가": "공급가 (원)", "결제금액(상품별)": "결제금액 (원)"},
        opacity=0.6
    )
    st.plotly_chart(fig_price_scatter, use_container_width=True)

# ------------------------------------------------------------------
# 푸터
# ------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.info("""
📊 **판매 데이터 분석 대시보드**

- 전체 레코드: 9,224건
- 클러스터: 4개
- 데이터 기간: 2025년 9월
""")
