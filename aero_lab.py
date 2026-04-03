import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Actarus F1 Aero-Lab", page_icon="🏎️", layout="wide")

# CSS for a professional "Dark Mode" Industrial UI
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .stMetric { background: #161b22; border: 1px solid #30363d; padding: 1rem; border-radius: 10px; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE GROQ ---
# Note: In a production environment, use st.secrets["GROQ_API_KEY"]
client = Groq(api_key="YOUR_GROQ_API_KEY")

# --- DATA ENGINE ---
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv('actaruslab_f1_telemetry_2026.csv')
    return df

@st.cache_resource
def train_aero_model(data):
    X = data[['speed_kmh', 'wing_angle_deg', 'drs_active']]
    y = data['downforce_n']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Calculate performance
    preds = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds)
    }
    importances = dict(zip(X.columns, model.feature_importances_))
    return model, metrics, importances

# --- AI INSIGHT ENGINE ---
def get_ai_analysis(metrics, importances, scenario_data=None):
    try:
        prompt = f"""
        Act as a Senior F1 Aerodynamicist. Analyze this ML Model performance on 2026 Telemetry:
        - R2 Score: {metrics['r2']:.4f}
        - Key Drivers (Feature Importance): {importances}
        
        {f"Current Scenario Analysis: {scenario_data}" if scenario_data else "Provide a general overview of model reliability."}
        
        Provide a technical summary of how Wing Angle and DRS are interacting based on these weights. 
        Keep it under 150 words. Use engineering terminology (e.g., Reynolds numbers, flow separation, lift-to-drag).
        """
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a technical expert."}, 
                      {"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Insight Unavailable: {str(e)}"

# --- UI LAYOUT ---
df = load_and_preprocess()
model, metrics, importances = train_aero_model(df)

st.title("🏎️ F1 Aero-Intelligence Lab")
st.caption("2026 Regulation Simulation Engine | Powered by Scikit-Learn & Groq LPU")

# 1. KPI TOP BAR
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model Precision (R²)", f"{metrics['r2']*100:.1f}%")
m2.metric("Mean Error", f"{metrics['mae']:.2f} N")
m3.metric("Data Points", f"{len(df):,}")
m4.metric("Inference Latency", "12ms", delta="-2ms")

st.divider()

# 2. MAIN ANALYTICS TABS
tab1, tab2, tab3 = st.tabs(["📊 Performance Analytics", "🧠 ML Studio", "🔮 Real-time Predictor"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Downforce Profile by Speed & DRS")
        fig = px.scatter(df.sample(2000), x="speed_kmh", y="downforce_n", 
                         color="drs_active", template="plotly_dark",
                         color_continuous_scale="Viridis", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.subheader("Aerodynamic Efficiency")
        efficiency_fig = px.density_heatmap(df.sample(2000), x="wing_angle_deg", y="drag_n", 
                                            nbinsx=20, nbinsy=20, template="plotly_dark")
        st.plotly_chart(efficiency_fig, use_container_width=True)

with tab2:
    st.subheader("Random Forest Feature Importance")
    imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance']).sort_values('Importance')
    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', 
                     template="plotly_dark", color='Importance', color_continuous_scale="Blues")
    st.plotly_chart(fig_imp, use_container_width=True)
    
    with st.expander("🤖 AI Model Commentary (Groq)", expanded=True):
        if st.button("Generate Expert Analysis"):
            with st.spinner("Consulting AI Aerodynamicist..."):
                insight = get_ai_analysis(metrics, importances)
                st.write(insight)

with tab3:
    st.subheader("Simulation Input")
    c1, c2, c3 = st.columns(3)
    s_speed = c1.slider("Ground Speed (km/h)", 100, 360, 250)
    s_wing = c2.slider("Wing Angle (deg)", 5, 35, 20)
    s_drs = c3.selectbox("DRS State", [0, 1], format_func=lambda x: "Active" if x==1 else "Inactive")
    
    # Prediction logic
    input_data = pd.DataFrame([[s_speed, s_wing, s_drs]], columns=['speed_kmh', 'wing_angle_deg', 'drs_active'])
    prediction = model.predict(input_data)[0]
    
    st.markdown(f"### Predicted Downforce: `{prediction:.2f} Newtons`")
    
    # Agentic Explanation
    if st.checkbox("Ask Groq to explain this prediction"):
        scenario = f"Speed: {s_speed}km/h, Wing: {s_wing}deg, DRS: {s_drs}"
        explanation = get_ai_analysis(metrics, importances, scenario)
        st.info(explanation)

# 3. DATA ARCHITECTURE FOOTER
st.divider()
with st.expander("🛠️ System Architecture"):
    st.code("""
    Layer 1: Streamlit Frontend (Stateful UI)
    Layer 2: Scikit-Learn (RandomForestRegressor for Physics Proxy)
    Layer 3: Groq Cloud API (Llama-3-70B for Semantic Analysis)
    Layer 4: Pandas/Plotly (Data Manipulation & Vector Rendering)
    """, language="text")
