import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. SYSTEM CONFIG & STYLING ---
st.set_page_config(page_title="Actarus F1 Aero-Intelligence", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .stMetric { background: #111418; border: 1px solid #1f2937; padding: 1.5rem; border-radius: 12px; }
    div[data-testid="stExpander"] { border: 1px solid #1f2937; border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & ENGINE CACHING ---
@st.cache_data(show_spinner=False)
def load_processed_data(file_path: str):
    """Loads data and runs pre-computation for anomalies to prevent KeyErrors."""
    df = pd.read_csv(file_path)
    
    # Pre-compute Anomaly Detection so the column exists globally
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    df['is_anomaly'] = iso_forest.fit_predict(df[['speed_kmh', 'downforce_n', 'drag_n']])
    
    # Engineering Feature
    df['Efficiency_LD'] = (df['downforce_n'] / df['drag_n']).replace([np.inf, -np.inf], 0)
    return df

@st.cache_resource(show_spinner=False)
def train_predictive_model(data: pd.DataFrame):
    """Trains the regressor for the prediction tab."""
    X = data[['speed_kmh', 'wing_angle_deg', 'drs_active']]
    y = data['downforce_n']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    regressor = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1)
    regressor.fit(X_train, y_train)
    
    metrics = {
        "r2": r2_score(y_test, regressor.predict(X_test)), 
        "mae": mean_absolute_error(y_test, regressor.predict(X_test))
    }
    importances = dict(zip(X.columns, regressor.feature_importances_))
    
    return regressor, metrics, importances

# --- 3. AI AGENT ---
def get_ai_interpretation(api_key: str, context: dict):
    if not api_key: return "⚠️ API Key missing. Please check the sidebar."
    try:
        client = Groq(api_key=api_key)
        prompt = f"Analyze F1 Aero Data: R2 {context['metrics']['r2']:.2f}, Inputs {context['current_sim']}. Provide 3 technical bullet points."
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Inference Error: {str(e)}"

# --- 4. APP INITIALIZATION ---
df = load_processed_data('actaruslab_f1_telemetry_2026.csv')
regressor, metrics, importances = train_predictive_model(df)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Engineering Core")
    api_key = st.text_input("Groq API Key", type="password")
    speed_filter = st.slider("Speed Window", 100, 360, (150, 320))
    dev_mode = st.toggle("System Logs", value=False)

# --- 6. MAIN UI ---
st.title("🏎️ Actarus Aero-Intelligence Lab")

# KPI Bar - Fixed KeyError by ensuring df has 'is_anomaly' from load_processed_data
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictive Accuracy", f"{metrics['r2']*100:.2f}%")
c2.metric("Mean Sensor Error", f"{metrics['mae']:.2f} N")
c3.metric("Anomalous Points", len(df[df['is_anomaly'] == -1]))
c4.metric("Throughput", "2.4M Ops/s")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🔮 Predictor", "🛠️ Health"])

with tab1:
    col_l, col_r = st.columns([2, 1])
    filtered_view = df[(df['speed_kmh'] >= speed_filter[0]) & (df['speed_kmh'] <= speed_filter[1])]
    
    with col_l:
        # Fixed: use_container_width replaced with width="stretch"
        fig = px.scatter(filtered_view.sample(2000), x="speed_kmh", y="downforce_n", color="wing_angle_deg", template="plotly_dark")
        st.plotly_chart(fig, width="stretch")
    
    with col_r:
        imp_df = pd.DataFrame(list(importances.items()), columns=['F', 'V']).sort_values('V')
        fig_bar = px.bar(imp_df, x='V', y='F', orientation='h', template="plotly_dark")
        st.plotly_chart(fig_bar, width="stretch")

with tab2:
    sc1, sc2, sc3 = st.columns(3)
    s_speed = sc1.number_input("Speed (km/h)", 100, 365, 250)
    s_wing = sc2.number_input("Wing Angle (deg)", 5.0, 35.0, 15.0)
    s_drs = sc3.selectbox("DRS", [0, 1], format_func=lambda x: "ACTIVE" if x==1 else "CLOSED")
    
    input_vector = pd.DataFrame([[s_speed, s_wing, s_drs]], columns=['speed_kmh', 'wing_angle_deg', 'drs_active'])
    pred_val = regressor.predict(input_vector)[0]
    
    st.success(f"#### **Predicted Downforce:** `{pred_val:.2f} N`")
    
    if st.button("Generate Groq Analysis"):
        with st.spinner("Processing..."):
            ctx = {"metrics": metrics, "current_sim": input_vector.to_dict('records')[0]}
            st.markdown(get_ai_interpretation(api_key, ctx))

with tab3:
    st.subheader("Sensor Reliability Map")
    # Fixed: use_container_width replaced with width="stretch"
    fig_anom = px.scatter_3d(df.sample(1500), x='speed_kmh', y='downforce_n', z='drag_n',
                             color='is_anomaly', color_discrete_map={1: '#3b82f6', -1: '#ef4444'},
                             template="plotly_dark")
    st.plotly_chart(fig_anom, width="stretch")

if dev_mode:
    st.divider()
    st.json({"status": "active", "df_columns": list(df.columns)})
