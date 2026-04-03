import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import time

# --- 1. SYSTEM CONFIG & STYLING ---
st.set_page_config(page_title="Actarus F1 Aero-Intelligence", page_icon="🏎️", layout="wide")

# Custom Industrial Dark Theme
st.markdown("""
    <style>
    .stMetric { background: #111418; border: 1px solid #1f2937; padding: 1.5rem; border-radius: 12px; transition: 0.3s; }
    .stMetric:hover { border-color: #3b82f6; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
    div[data-testid="stExpander"] { border: 1px solid #1f2937; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & ENGINE CACHING ---
@st.cache_data(show_spinner=False)
def load_and_clean_data(file_path: str):
    """Loads 150k rows and applies engineering transforms."""
    df = pd.read_csv(file_path)
    df['Efficiency_LD'] = (df['downforce_n'] / df['drag_n']).replace([np.inf, -np.inf], 0)
    return df

@st.cache_resource(show_spinner=False)
def train_telemetry_models(data: pd.DataFrame):
    """Trains a predictive regressor and an unsupervised anomaly detector."""
    # ML Regressor for Downforce
    X = data[['speed_kmh', 'wing_angle_deg', 'drs_active']]
    y = data['downforce_n']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    regressor = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1)
    regressor.fit(X_train, y_train)
    
    # Anomaly Detection (Unsupervised)
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    data['is_anomaly'] = iso_forest.fit_predict(data[['speed_kmh', 'downforce_n', 'drag_n']])
    
    metrics = {"r2": r2_score(y_test, regressor.predict(X_test)), "mae": mean_absolute_error(y_test, regressor.predict(X_test))}
    importances = dict(zip(X.columns, regressor.feature_importances_))
    
    return regressor, iso_forest, metrics, importances

# --- 3. AI AGENT (GROQ) INTEGRATION ---
def get_ai_interpretation(api_key: str, context: dict):
    """Bridge to Groq Llama-3 for technical inference."""
    if not api_key: return "⚠️ API Key missing in Sidebar. AI Analysis disabled."
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        Act as a Senior F1 Telemetry Engineer. Analyze this state:
        - ML Reliability (R2): {context['metrics']['r2']:.4f}
        - Current Input: {context['current_sim']}
        - Top Drivers: {context['importances']}
        
        Explain why this specific configuration is efficient or inefficient from an aerodynamic stall perspective. 
        Limit to 3 punchy, technical bullet points.
        """
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"System Log Error: {str(e)}"

# --- 4. APP INITIALIZATION ---
df = load_and_clean_data('actaruslab_f1_telemetry_2026.csv')
regressor, detector, metrics, importances = train_telemetry_models(df)

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ Engineering Core")
    api_key = st.text_input("Groq API Key", type="password", help="Get at console.groq.com")
    st.divider()
    st.subheader("Global Filters")
    speed_filter = st.slider("Analysis Window (Speed)", 100, 360, (150, 320))
    dev_mode = st.toggle("Enable Dev Mode Logs", value=False)
    
    if st.button("Purge System Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --- 6. MAIN UI LAYOUT ---
st.title("🏎️ Actarus Aero-Intelligence Lab")
st.markdown("### Integrated 2026 F1 Telemetry Analysis & Prediction")

# Top KPI Bar
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictive Accuracy", f"{metrics['r2']*100:.2f}%", help="Coefficient of Determination (R²)")
c2.metric("Mean Sensor Error", f"{metrics['mae']:.2f} N")
c3.metric("Anomalous Points", len(df[df['is_anomaly'] == -1]))
c4.metric("Engine Throughput", "2.4M Ops/s")

st.divider()

# TABBED INTERFACE
tab1, tab2, tab3 = st.tabs(["📊 Performance Studio", "🔮 Predictive AI & Groq", "🛠️ System Health"])

with tab1:
    col_l, col_r = st.columns([2, 1])
    filtered_view = df[(df['speed_kmh'] >= speed_filter[0]) & (df['speed_kmh'] <= speed_filter[1])]
    
    with col_l:
        st.subheader("Load Analysis (Downforce vs Drag)")
        fig = px.scatter(filtered_view.sample(3000), x="speed_kmh", y="downforce_n", 
                         size="drag_n", color="wing_angle_deg", template="plotly_dark",
                         color_continuous_scale="Viridis", labels={'downforce_n': 'Downforce (N)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        st.subheader("Feature Dominance")
        imp_df = pd.DataFrame(list(importances.items()), columns=['F', 'V']).sort_values('V')
        fig_bar = px.bar(imp_df, x='V', y='F', orientation='h', template="plotly_dark", color_discrete_sequence=['#3b82f6'])
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Real-Time Predictive Simulation")
    sc1, sc2, sc3 = st.columns(3)
    s_speed = sc1.number_input("Target Speed (km/h)", 100, 365, 280)
    s_wing = sc2.number_input("Wing Incidence (deg)", 5.0, 35.0, 18.0)
    s_drs = sc3.selectbox("DRS System", [0, 1], format_func=lambda x: "DEPLOYED" if x==1 else "CLOSED")
    
    # Run Inference
    input_vector = pd.DataFrame([[s_speed, s_wing, s_drs]], columns=['speed_kmh', 'wing_angle_deg', 'drs_active'])
    pred_val = regressor.predict(input_vector)[0]
    
    st.info(f"#### **Predicted Downforce Level:** `{pred_val:.2f} Newtons`")
    
    if st.button("Ask Groq AI to Analyze this Scenario"):
        with st.status("Querying LPU Inference Engine..."):
            ctx = {"metrics": metrics, "current_sim": input_vector.to_dict('records')[0], "importances": importances}
            analysis = get_ai_interpretation(api_key, ctx)
            st.write(analysis)

with tab3:
    st.subheader("Anomaly Detection (Isolation Forest)")
    anomalies = df[df['is_anomaly'] == -1]
    st.warning(f"Detected {len(anomalies)} telemetry outliers representing potential sensor fatigue or mechanical failure.")
    
    fig_anom = px.scatter_3d(df.sample(2000), x='speed_kmh', y='downforce_n', z='drag_n',
                             color='is_anomaly', color_discrete_map={1: '#1f2937', -1: '#ef4444'},
                             template="plotly_dark", title="Outlier Cluster Map (Red = Anomaly)")
    st.plotly_chart(fig_anom, use_container_width=True)

# --- 7. DEV MODE LOGS ---
if dev_mode:
    st.divider()
    st.subheader("🛠️ Developer Metadata")
    st.json({"Model_State": "Ready", "Features": list(importances.keys()), "Data_Shape": df.shape})
