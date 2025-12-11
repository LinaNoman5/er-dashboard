import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# PAGE CONFIGURATION + CUSTOM UI THEME
# ------------------------------------------------------------
st.set_page_config(
    page_title="ER Arrival Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(to bottom, #dcecff, #eaf3ff);
}

.card {
    padding: 22px;
    border-radius: 18px;
    background: white;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    margin-bottom: 22px;
}

.title-text {
    font-size: 34px;
    font-weight: 700;
    color: #083c6e;
    text-align: center;
    margin-bottom: 20px;
}

.section-label {
    font-size: 20px;
    font-weight: 600;
    color: #0d3c74;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
model = joblib.load("xgb_model.joblib")

FEATURES = [
    'hospital_id', 'is_weekend', 'is_public_holiday', 'month', 'day_of_year',
    'temperature_C', 'humidity_pct', 'expected_visits', 'critical_cases',
    'prev_day_visits', 'ma_7', 'ma_14', 'dow_num'
]


# ------------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------------
st.markdown("<div class='title-text'>🏥 ER Arrival Prediction Dashboard</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# SIDEBAR INPUTS
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Input Parameters")

    hospital_id = st.selectbox("Hospital ID", [1, 2, 3, 4])
    is_weekend = st.selectbox("Weekend?", [0, 1])
    is_public_holiday = st.selectbox("Public Holiday?", [0, 1])

    month = st.slider("Month", 1, 12, 5)
    day_of_year = st.slider("Day of Year", 1, 365, 120)

    temperature_C = st.slider("Temperature (°C)", 10, 50, 32)
    humidity_pct = st.slider("Humidity (%)", 5, 100, 60)

    expected_visits = st.slider("Expected Visits", 10, 300, 120)
    critical_cases = st.slider("Critical Cases", 0, 40, 5)

    prev_day_visits = st.slider("Yesterday's Visits", 10, 200, 80)
    ma_7 = st.slider("7-Day Moving Avg", 10, 200, 100)
    ma_14 = st.slider("14-Day Moving Avg", 10, 200, 90)

    dow_num = st.slider("Day of Week (0=Mon)", 0, 6, 2)

    submit = st.button("🔮 Predict ER Arrivals", use_container_width=True)


# ------------------------------------------------------------
# MAIN PREDICTION + VISUAL OUTPUT
# ------------------------------------------------------------
if submit:

    # 1️⃣ Prepare input
    input_vector = np.array([[hospital_id, is_weekend, is_public_holiday,
                              month, day_of_year, temperature_C, humidity_pct,
                              expected_visits, critical_cases, prev_day_visits,
                              ma_7, ma_14, dow_num]])

    # 2️⃣ Predict
    prediction = int(model.predict(input_vector)[0])

    # 3️⃣ Compute capacity
    capacity_limit = 150
    capacity_load = min(100, round((prediction / capacity_limit) * 100, 1))


    # --------------------------------------------------------
    # THREE MAIN CHARTS (FIXED — NO EXTRA WHITE BLOCKS)
    # --------------------------------------------------------
    col1, col2, col3 = st.columns([1.3, 1, 1.3])

    # ===== COLUMN 1: BAR CHART =====
    with col1:
        st.markdown("<div class='section-label'>📊 Prediction vs Yesterday</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        df_plot = pd.DataFrame({
            "Metric": ["Yesterday", "Predicted"],
            "Count": [prev_day_visits, prediction]
        })

        fig_bar = px.bar(
            df_plot, x="Metric", y="Count",
            color="Metric",
            color_discrete_sequence=["#2C67FF", "#0B4CC2"],
            text="Count",
            height=350
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=10))

        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ===== COLUMN 2: GAUGE =====
    with col2:
        st.markdown("<div class='section-label'>❤️ Capacity Load</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=capacity_load,
            number={'suffix': "%", 'font': {'size': 40, 'color': "#8B0000"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#B22222"},
                'bgcolor': "#ffffff",
                'borderwidth': 0,
                'steps': [{'range': [0, 100], 'color': 'rgba(0,82,204,0.12)'}]
            }
        ))
        gauge_fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=10))

        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ===== COLUMN 3: TREND CHART =====
    with col3:
        st.markdown("<div class='section-label'>📉 Trend</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        history = [ma_14, ma_7, prev_day_visits, prediction]

        fig_line = go.Figure(go.Scatter(
            y=history,
            mode='lines+markers',
            line=dict(color="#165BAA", width=4),
            marker=dict(size=10, color="#0B4CC2")
        ))
        fig_line.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=10))

        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # FINAL BIG NUMBER CARD
    # --------------------------------------------------------
    st.markdown(f"""
    <div class='card' style='text-align:center; padding: 35px;'>
        <h2 style='color:#0d3c74;'>Predicted ER Arrivals</h2>
        <h1 style='font-size:64px; font-weight:800; color:#0B4CC2;'>{prediction}</h1>
    </div>
    """, unsafe_allow_html=True)
