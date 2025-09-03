import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import io
import numpy as np

# ----------------- CONFIGURATION & STYLING -----------------
st.set_page_config(page_title="WSN Intrusion Detection", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; color: #e0e0e0; }
    .css-1d391kg { background-color: #16213e; } /* Sidebar color */
    .stMetric { background-color: #0f3-460; border-radius: 10px; padding: 15px; text-align: center; }
    .stMetric > label { color: #a0a0a0; } 
    .stMetric > div { color: #e94560; font-size: 1.75rem; }
    .stButton>button { width: 100%; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #c0c0c0; }
</style>
""", unsafe_allow_html=True)

# ----------------- MASTER DATA LOADING (UNCHANGED) -----------------
@st.cache_resource
def load_artifacts_from_zip(zip_path='artifacts.zip'):
    """Loads all necessary .joblib files from a nested 'artifacts' folder within a zip archive."""
    artifacts = {}
    nested_folder = 'artifacts/'
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            files_to_load = ['scaler.joblib', 'feature_names.joblib', 'dt_model.joblib', 'rf_model.joblib', 'lr_model.joblib', 'knn_model.joblib', 'xgb_model.joblib']
            for file_in_zip in files_to_load:
                full_path = nested_folder + file_in_zip
                with z.open(full_path) as f:
                    key = file_in_zip.replace('.joblib', '')
                    artifacts[key] = joblib.load(f)
        return artifacts
    except Exception as e:
        st.error(f"Fatal Error: Could not load artifacts from '{zip_path}'. Ensure it contains a folder named 'artifacts' with all .joblib files. Error: {e}")
        st.stop()

artifacts = load_artifacts_from_zip()
scaler = artifacts['scaler']
expected_features = artifacts['feature_names']
models = { "XGBoost": artifacts['xgb_model'], "Random Forest": artifacts['rf_model'], "Decision Tree": artifacts['dt_model'], "K-Nearest Neighbors": artifacts['knn_model'], "Logistic Regression": artifacts['lr_model']}

# ----------------- DATA GENERATION FUNCTION (UNCHANGED) -----------------
def generate_random_data(num_rows, attack_ratio=0.4):
    data = {feature: np.zeros(num_rows) for feature in expected_features}
    df = pd.DataFrame(data)
    attack_indices = np.random.choice(df.index, int(num_rows * attack_ratio), replace=False)
    for i in attack_indices:
        attack_type = np.random.choice(['Blackhole', 'Flooding'])
        if attack_type == 'Blackhole':
            df.loc[i, 'DATA_R'] = np.random.randint(100, 200)
            df.loc[i, 'Data_Sent_To_BS'] = 0
        elif attack_type == 'Flooding':
            df.loc[i, 'DATA_S'] = np.random.randint(2000, 4000)
    return df[expected_features]

# ----------------- DASHBOARD DISPLAY FUNCTION (UNCHANGED) -----------------
def display_dashboard(df, model, model_name):
    st.markdown("---")
    st.header(f"Analysis Dashboard (using {model_name})")
    try:
        df_display = df.copy()
        if ' id' in df_display.columns:
            df_display = df_display.drop(' id', axis=1)
        X_test = df_display[expected_features]
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        attack_labels = {3: 'Normal Traffic', 0: 'Blackhole Attack', 1: 'Flooding Attack', 2: 'Grayhole Attack', 4: 'Scheduling Attack'}
        df_display['Prediction'] = [attack_labels.get(p, 'Unknown') for p in predictions]
        df_display['Is Attack'] = df_display['Prediction'] != 'Normal Traffic'
        total_packets, threats_detected = len(df_display), int(df_display['Is Attack'].sum())
        threat_percentage = (threats_detected / total_packets * 100) if total_packets > 0 else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Packets", f"{total_packets:,}")
        col2.metric("Threats Detected", f"{threats_detected:,}")
        col3.metric("Threat Percentage", f"{threat_percentage:.2f}%")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# ----------------- PAGE SELECTION -----------------
st.sidebar.title("🔬 WSN Security Framework")
page = st.sidebar.radio("Select a page", ["About the Project", "Model Performance Comparison", "Live Intrusion Detection", "Optimization Algorithms Explored"])

# --- ABOUT & MODEL COMPARISON PAGES (UNCHANGED) ---
if page == "About the Project":
    st.title("🛡️ About The WSN Intrusion Detection Project")
    st.markdown("This project presents a complete framework for identifying cyber attacks in Wireless Sensor Networks (WSNs) using a comparative machine learning approach.")
    st.graphviz_chart('''
        digraph {
            graph [rankdir="LR", bgcolor="#1a1a2e", fontcolor="white"];
            node [shape=box, style="filled,rounded", fillcolor="#0f3460", color="#e94560", fontcolor="white", penwidth=2];
            edge [color="white"];
            Data [label="1. Data Acquisition"];
            Preprocess [label="2. Preprocessing"];
            Train [label="3. Comparative Training"];
            App [label="4. Build Web Framework"];
            Data -> Preprocess -> Train -> App;
        }
    ''')
elif page == "Model Performance Comparison":
    st.title("📊 Model Performance Showdown")
    performance_data = {'Model': ['XGBoost', 'Random Forest', 'Decision Tree', 'K-Nearest Neighbors', 'Logistic Regression'], 'F1-Score': [0.9997, 0.9995, 0.9985, 0.9971, 0.9850]}
    df_perf = pd.DataFrame(performance_data)
    fig_perf = px.bar(df_perf.sort_values('F1-Score', ascending=True), x='F1-Score', y='Model', orientation='h', title="Model F1-Scores", template='plotly_dark', text='F1-Score')
    st.plotly_chart(fig_perf, use_container_width=True)
    st.markdown("The results show that **XGBoost** and **Random Forest** are the top-performing classifiers.")

# --- LIVE DETECTION PAGE (UNCHANGED) ---
elif page == "Live Intrusion Detection":
    st.title("🕵️ Live Network Traffic Analysis")
    chosen_model_name = st.selectbox("Step 1: Choose a Model for Prediction", list(models.keys()))
    model_to_use = models[chosen_model_name]
    st.markdown("---")
    st.subheader("Step 2: Provide Data for Analysis")
    if 'df_to_process' not in st.session_state:
        st.session_state.df_to_process = None
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Option A: Upload Data")
        uploaded_file = st.file_uploader("Upload a CSV file.", type="csv")
        if uploaded_file:
            st.session_state.df_to_process = pd.read_csv(uploaded_file)
    with col2:
        st.markdown("#### Option B: Generate Simulation")
        num_rows = st.slider("Number of packets to generate:", 500, 5000, 1500, key="slider")
        if st.button("Generate & Predict"):
            st.session_state.df_to_process = generate_random_data(num_rows)
    if st.session_state.df_to_process is not None:
        display_dashboard(st.session_state.df_to_process, model_to_use, chosen_model_name)

# --- START OF NEW OPTIMIZATION ALGORITHMS PAGE ---
elif page == "Optimization Algorithms Explored":
    st.title("🧠 Exploring Optimization Algorithms")
    st.markdown("Optimization algorithms are the engines that power machine learning. Their goal is to minimize the model's error (or **loss**) by systematically adjusting the model's internal parameters. This page provides interactive visualizations to help understand how they work.")
    
    # --- 1. The Loss Landscape ---
    st.markdown("---")
    st.header("1. The Loss Landscape: The Valley of Errors")
    st.markdown("Imagine the model's error as a 3D landscape. The lowest point in the valley represents the minimum possible error. The optimizer's job is to find this point.")
    
    # Create a simple quadratic loss function: z = x^2 + y^2
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig_3d.update_layout(title='A Simple Loss Landscape (z = x^2 + y^2)', scene=dict(xaxis_title='Parameter 1', yaxis_title='Parameter 2', zaxis_title='Loss (Error)'), autosize=True, height=500, template='plotly_dark')
    st.plotly_chart(fig_3d, use_container_width=True)

    # --- 2. Gradient Descent Simulation ---
    st.markdown("---")
    st.header("2. Interactive Gradient Descent")
    st.markdown("The most fundamental optimizer is **Gradient Descent**. It's like a hiker in a foggy
