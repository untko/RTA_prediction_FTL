import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Page Config ---
st.set_page_config(
    page_title="Traffic Risk AI", 
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS (Targeting Native Metrics) ---
st.markdown("""
<style>
    /* Main container spacing */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Styling Streamlit's Native Metric to look like a Card */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Center text in metrics */
    div[data-testid="stMetricLabel"] { justify-content: center; }
    div[data-testid="stMetricValue"] { justify-content: center; color: #FFFFFF; }
    div[data-testid="stMetricDelta"] { justify-content: center; }
    
    /* Remove default chart margin */
    .js-plotly-plot { margin-top: -20px; }
</style>
""", unsafe_allow_html=True)

# --- 3. Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        return joblib.load('accident_model_artifacts.joblib')
    except FileNotFoundError:
        st.error("Model file not found. Please run 'train_pipeline.py' first.")
        st.stop()

artifacts = load_artifacts()
model = artifacts['model']
feature_names = artifacts['feature_names']
metrics = artifacts['metrics']

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 4. Sidebar Inputs ---
st.sidebar.title("🛠️ Input Parameters")
st.sidebar.markdown("Adjust conditions to predict risk.")

with st.sidebar.expander("📍 Road Conditions", expanded=True):
    road_type = st.selectbox("Road Type", ["highway", "urban", "rural"], help="Type of road infrastructure.")
    curvature = st.slider("Curvature", 0.0, 1.0, 0.5, help="0.0 = Straight, 1.0 = Very Winding")
    lighting = st.selectbox("Lighting", ["bright", "dim", "dark"], help="Visibility conditions.")
    weather = st.selectbox("Weather", ["clear", "rainy", "foggy"])

with st.sidebar.expander("🚗 Traffic Details", expanded=True):
    speed_limit = st.slider("Speed Limit", 20, 80, 45)
    num_lanes = st.slider("Lanes", 1, 4, 2)
    traffic_volume = st.slider("Traffic Density (0-10)", 0, 10, 5)

with st.sidebar.expander("⚠️ Context", expanded=False):
    time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "night"])
    road_signs = st.checkbox("Road Signs Present?", True)
    public_road = st.checkbox("Public Road?", True)
    holiday = st.checkbox("Holiday?", False)
    school_season = st.checkbox("School Season?", False)

# --- 5. Prediction Logic ---
def get_prediction_data(custom_speed=None):
    # Allow overriding speed for "What-If" analysis
    current_speed = custom_speed if custom_speed is not None else speed_limit
    
    data = {
        'road_type': road_type, 'num_lanes': num_lanes, 'curvature': curvature,
        'speed_limit': current_speed, 'lighting': lighting, 'weather': weather,
        'road_signs_present': int(road_signs), 'public_road': int(public_road),
        'time_of_day': time_of_day, 'holiday': int(holiday),
        'school_season': int(school_season), 'num_reported_accidents': traffic_volume
    }
    df_raw = pd.DataFrame([data])
    categorical = ['road_type', 'lighting', 'weather', 'time_of_day']
    df_enc = pd.get_dummies(df_raw, columns=categorical)
    df_final = df_enc.reindex(columns=feature_names, fill_value=0)
    
    prob = model.predict(df_final)[0]
    
    # Feature Importance
    global_imp = model.feature_importances_
    active_indices = np.where(df_final.values[0] > 0)[0]
    relevant_imp = global_imp[active_indices]
    relevant_names = np.array(feature_names)[active_indices]
    sorted_idx = np.argsort(relevant_imp)[::-1][:5]
    
    return prob, relevant_names[sorted_idx], relevant_imp[sorted_idx]

prob, top_feats, top_scores = get_prediction_data()
st.session_state['history'].append(prob)

# --- 6. Main Dashboard Layout ---

st.title("🚦 Traffic Accident Risk Predictor")
st.markdown("Real-time AI assessment of road safety conditions.")
st.markdown("---")

# === ROW 1: Gauge & Metrics ===
col1, col2 = st.columns([2, 1])

with col1:
    # GAUGE CHART
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Accident Probability (%)", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 30], 'color': '#00CC96'},
                {'range': [30, 60], 'color': '#FFA15A'},
                {'range': [60, 100], 'color': '#EF553B'}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': prob * 100}
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("### Model Trust")
    # Metric 1: Precision
    st.metric(
        label="High-Risk Precision",
        value=f"{metrics['precision']:.1%}",
        delta="Reliability",
        delta_color="off",
        help="✅ PRECISION: When the AI predicts 'High Risk', it is correct 84% of the time. (High precision = fewer false alarms)."
    )
    
    st.write("") # Spacer
    
    # Metric 2: Recall
    st.metric(
        label="Severe Accident Recall",
        value=f"{metrics['recall']:.1%}",
        delta="Safety Net",
        delta_color="off",
        help="🛡️ RECALL: Of all actual severe accidents in history, the AI successfully caught 75% of them. (High recall = fewer missed dangers)."
    )

st.markdown("---")

# === ROW 2: Drivers & Recommendation ===
col_mid1, col_mid2 = st.columns([1.5, 1])

with col_mid1:
    st.subheader("🔍 Key Risk Drivers")
    fig_bar = px.bar(x=top_scores, y=top_feats, orientation='h', color=top_scores, color_continuous_scale='Reds')
    fig_bar.update_layout(
        height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Impact Contribution", yaxis_title="", coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_mid2:
    st.subheader("💡 Recommendation")
    
    # Risk Level Box
    if prob > 0.6:
        st.error("🚨 **HIGH RISK DETECTED**\n\nRecommended: Delay travel or seek alternate route.")
    elif prob > 0.3:
        st.warning("⚠️ **MODERATE RISK**\n\nRecommended: Drive with increased caution.")
    else:
        st.success("✅ **SAFE CONDITIONS**\n\nStandard driving protocols apply.")
        
    # NEW FEATURE: Safety Optimizer
    st.markdown("#### 📉 Safety Optimizer")
    # Check if lowering speed helps
    if speed_limit > 30:
        lower_speed_prob, _, _ = get_prediction_data(custom_speed=speed_limit-10)
        risk_reduction = (prob - lower_speed_prob) * 100
        
        if risk_reduction > 5:
            st.info(f"⚡ **Tip:** Reducing speed by 10 mph would lower risk by **{risk_reduction:.1f}%**.")
        else:
            st.info("⚡ **Tip:** Speed is not the primary risk factor right now. Check weather/road type.")

st.markdown("---")

# === ROW 3: Stability (Full Width) ===
st.subheader("📉 Session Stability")
if len(st.session_state['history']) > 1:
    df_hist = pd.DataFrame(st.session_state['history'][-50:], columns=['risk']) # Show last 50
    fig_line = px.line(df_hist, y='risk', markers=True)
    fig_line.update_layout(
        height=250, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#333', range=[0, 1], title="Risk Score")
    )
    fig_line.update_traces(line_color='#00CC96', line_width=3)
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.caption("Interact with the sidebar controls to see how risk stability changes over time.")
