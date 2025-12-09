import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Page Config (Must be first) ---
st.set_page_config(
    page_title="Traffic Risk AI", 
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for "Modern" Look ---
st.markdown("""
<style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 1rem;
        color: #888;
        margin-bottom: 5px;
    }
    
    /* Remove default top margin from charts */
    .js-plotly-plot {
        margin-top: -20px;
    }
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

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 4. Sidebar Inputs (Kept clean) ---
st.sidebar.title("🛠️ Input Parameters")
st.sidebar.markdown("Adjust conditions to predict risk.")

with st.sidebar.expander("📍 Road Conditions", expanded=True):
    road_type = st.selectbox("Road Type", ["highway", "urban", "rural"])
    curvature = st.slider("Curvature", 0.0, 1.0, 0.5, help="0 = Straight, 1 = Very Curvy")
    lighting = st.selectbox("Lighting", ["bright", "dim", "dark"])
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

# --- 5. Logic ---
def get_prediction():
    # Build Input Data
    data = {
        'road_type': road_type, 'num_lanes': num_lanes, 'curvature': curvature,
        'speed_limit': speed_limit, 'lighting': lighting, 'weather': weather,
        'road_signs_present': int(road_signs), 'public_road': int(public_road),
        'time_of_day': time_of_day, 'holiday': int(holiday),
        'school_season': int(school_season), 'num_reported_accidents': traffic_volume
    }
    df_raw = pd.DataFrame([data])
    
    # Encode
    categorical = ['road_type', 'lighting', 'weather', 'time_of_day']
    df_enc = pd.get_dummies(df_raw, columns=categorical)
    
    # Align
    df_final = df_enc.reindex(columns=feature_names, fill_value=0)
    
    # Predict
    prob = model.predict(df_final)[0]
    
    # Feature Importance (Local Approximation)
    # We use global importance * feature value magnitude as a proxy for "local" contribution
    # For a real production app, use SHAP values. 
    global_imp = model.feature_importances_
    # Simple trick: Identify which features in this input are active (non-zero) and high importance
    active_indices = np.where(df_final.values[0] > 0)[0]
    relevant_imp = global_imp[active_indices]
    relevant_names = np.array(feature_names)[active_indices]
    
    # Sort
    sorted_idx = np.argsort(relevant_imp)[::-1][:5]
    
    return prob, relevant_names[sorted_idx], relevant_imp[sorted_idx]

# Run Prediction automatically or on button
prob, top_feats, top_scores = get_prediction()
st.session_state['history'].append(prob)

# --- 6. Main Dashboard Layout ---

# HEADER
st.title("🚦 Traffic Accident Risk Predictor")
st.markdown("Real-time AI assessment of road safety conditions.")
st.markdown("---")

# TOP ROW: The "Big Numbers"
col1, col2, col3 = st.columns([1.5, 1, 1])

with col1:
    # GAUGE CHART (Visual Impact)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Accident Probability (%)", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Transparent bar, we use steps/marker
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 30], 'color': '#00CC96'}, # Green
                {'range': [30, 60], 'color': '#FFA15A'}, # Orange
                {'range': [60, 100], 'color': '#EF553B'} # Red
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    # Metric Card 1
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">High-Risk Precision</div>
        <div class="metric-value" style="color: #00CC96;">{metrics['precision']:.1%}</div>
        <div style="font-size: 0.8rem; color: #666;">Model Reliability</div>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    # Metric Card 2
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">High-Risk Recall</div>
        <div class="metric-value" style="color: #AB63FA;">{metrics['recall']:.1%}</div>
        <div style="font-size: 0.8rem; color: #666;">Safety Net Coverage</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # STABILITY CHART (Small sparkline style)
    st.markdown("##### 📉 Session Stability")
    if len(st.session_state['history']) > 1:
        # Create a clean line chart
        df_hist = pd.DataFrame(st.session_state['history'][-20:], columns=['risk']) # Last 20 predictions
        fig_line = px.line(df_hist, y='risk', markers=True)
        fig_line.update_layout(
            height=200, 
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='#333', range=[0, 1])
        )
        fig_line.update_traces(line_color='#00CC96', line_width=3)
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Adjust inputs to generate history.")

# BOTTOM ROW: Analysis
st.markdown("---")
col_bottom_1, col_bottom_2 = st.columns([2, 1])

with col_bottom_1:
    st.subheader("🔍 Key Risk Drivers")
    # Clean Horizontal Bar Chart
    fig_bar = px.bar(
        x=top_scores, y=top_feats, orientation='h',
        color=top_scores, color_continuous_scale='Reds'
    )
    fig_bar.update_layout(
        height=300, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Impact Contribution",
        yaxis_title="",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_bottom_2:
    st.subheader("💡 Recommendation")
    if prob > 0.6:
        st.error("🚨 HIGH RISK: Consider alternate route or delaying travel.")
    elif prob > 0.3:
        st.warning("⚠️ CAUTION: Increased vigilance required.")
    else:
        st.success("✅ SAFE: Conditions appear favorable.")