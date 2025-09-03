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
    .css-1d3g3k7 { background-color: #16213e; } /* Sidebar color */
    .stMetric { background-color: #0f3460; border-radius: 10px; padding: 15px; text-align: center; }
    .stMetric > label { color: #a0a0a0; } 
    .stMetric > div { color: #e94560; font-size: 1.75rem; }
    .stButton>button { width: 100%; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #c0c0c0; }
</style>
""", unsafe_allow_html=True)

# ----------------- MASTER DATA LOADING & CACHING -----------------
@st.cache_resource
def load_artifacts_from_zip(zip_path='artifacts.zip'):
    """Loads all necessary .joblib files from a single zip archive."""
    artifacts = {}
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Assumes a nested folder named 'artifacts' inside the zip
            nested_folder = 'artifacts/'
            files_to_load = ['scaler.joblib', 'feature_names.joblib', 'dt_model.joblib', 'rf_model.joblib', 'lr_model.joblib', 'knn_model.joblib', 'xgb_model.joblib']
            for file_in_zip in files_to_load:
                full_path = nested_folder + file_in_zip
                with z.open(full_path) as f:
                    key = file_in_zip.replace('.joblib', '')
                    artifacts[key] = joblib.load(f)
        return artifacts
    except Exception as e:
        st.error(f"Fatal Error loading artifacts.zip. Ensure it contains a folder named 'artifacts' with all .joblib files. Error: {e}")
        st.stop()

artifacts = load_artifacts_from_zip()
scaler = artifacts['scaler']
expected_features = artifacts['feature_names']
models = { "XGBoost": artifacts['xgb_model'], "Random Forest": artifacts['rf_model'], "Decision Tree": artifacts['dt_model'], "K-Nearest Neighbors": artifacts['knn_model'], "Logistic Regression": artifacts['lr_model']}

# ----------------- DATA GENERATION FUNCTION -----------------
def generate_random_data(num_rows, attack_ratio=0.3):
    # (This function remains the same as before)
    data = {feature: np.zeros(num_rows) for feature in expected_features}
    df = pd.DataFrame(data)
    # Simple simulation logic
    attack_indices = np.random.choice(df.index, int(num_rows * attack_ratio), replace=False)
    for i in attack_indices:
        attack_type = np.random.choice(['Blackhole', 'Flooding'])
        if attack_type == 'Blackhole':
            df.loc[i, 'DATA_R'] = np.random.randint(1000, 2000)
            df.loc[i, 'Data_Sent_To_BS'] = 0
        elif attack_type == 'Flooding':
            df.loc[i, 'DATA_S'] = np.random.randint(2000, 4000)
    return df

# ----------------- UNIFIED DASHBOARD DISPLAY FUNCTION -----------------
def display_dashboard(df, model, model_name):
    # (This function remains the same as before)
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
        
        # (Visualizations and other dashboard elements)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# ----------------- PAGE SELECTION -----------------
st.sidebar.title("🔬 Comparative Framework")
page = st.sidebar.radio("Select a page", ["About the Project", "Model Performance Comparison", "Live Intrusion Detection"])

# --- START OF UPGRADED "ABOUT" PAGE ---
if page == "About the Project":
    st.title("🛡️ About The WSN Intrusion Detection Project")
    st.markdown("This project presents a complete framework for identifying cyber attacks in Wireless Sensor Networks (WSNs) using a comparative machine learning approach. It addresses the critical need for intelligent security in resource-constrained IoT environments.")

    st.markdown("---")

    st.header("Project Workflow")
    st.markdown("The entire project follows a structured data science lifecycle, from understanding the problem to deploying a functional web application.")
    
    # Workflow visualization
    st.graphviz_chart('''
        digraph {
            graph [rankdir="LR", bgcolor="#1a1a2e", fontcolor="white", label=""];
            node [shape=box, style="filled", fillcolor="#0f3460", color="#e94560", fontcolor="white", penwidth=2];
            edge [color="white"];
            
            subgraph cluster_0 {
                label = "Phase 1: Data & Modeling";
                bgcolor = "#16213e";
                fontcolor="white";
                Data [label="1. Data Acquisition (WSN-DS Dataset)"];
                Preprocess [label="2. Data Preprocessing (Scaling & Encoding)"];
                Train [label="3. Comparative Model Training (5 Classifiers)"];
                Data -> Preprocess -> Train;
            }

            subgraph cluster_1 {
                label = "Phase 2: Deployment";
                bgcolor = "#16213e";
                fontcolor="white";
                Save [label="4. Save Artifacts (Models & Scaler)"];
                App [label="5. Build Interactive Web App (Streamlit)"];
                Deploy [label="6. Deploy on Cloud"];
                Save -> App -> Deploy;
            }
            Train -> Save [lhead=cluster_1, ltail=cluster_0];
        }
    ''')

    st.markdown("---")

    st.header("Key Predictive Features")
    st.markdown("A crucial insight from the analysis is understanding *which* network features are the most powerful indicators of an attack. The Random Forest and XGBoost models identified the following features as most important:")

    # Feature Importance Data from the report
    feature_importance_data = {
        'Feature': [
            'Data_Sent_To_BS', 'DATA_S', 'DATA_R', 'Expaned Energy', 
            'Dist_To_CH', 'Is_CH', 'ADV_S', 'dist_CH_To_BS'
        ],
        'Importance': [0.35, 0.18, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]
    }
    df_feat = pd.DataFrame(feature_importance_data)

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        # Feature Importance Chart
        fig_feat = px.bar(df_feat.sort_values('Importance', ascending=True),
                          x='Importance', y='Feature', orientation='h',
                          title="Top Predictive Features for Attack Detection",
                          template='plotly_dark', text='Importance')
        fig_feat.update_traces(marker_color='#e94560', texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_feat, use_container_width=True)
    with col2:
        st.subheader("Why These Features Matter")
        st.markdown("""
        - **Data_Sent_To_BS:** This is the most critical feature. A massive drop in data reaching the Base Station is a classic sign of a **Blackhole** or **Grayhole** attack, where a malicious node is intercepting and discarding packets.
        - **DATA_S & DATA_R:** An unusually high number of sent or received data packets is a strong indicator of a **Flooding** attack, which aims to overwhelm the network.
        - **Expaned Energy:** Rapid energy depletion is a direct symptom of resource-exhaustion attacks like **Denial-of-Sleep** or Flooding.
        """)
# --- END OF UPGRADED "ABOUT" PAGE ---

elif page == "Model Performance Comparison":
    st.title("📊 Model Performance Showdown")
    # (Content is the same as before)
    performance_data = {'Model': ['XGBoost', 'Random Forest', 'Decision Tree', 'K-Nearest Neighbors', 'Logistic Regression'], 'Accuracy': [0.9997, 0.9995, 0.9985, 0.9971, 0.9850], 'F1-Score': [0.9997, 0.9995, 0.9985, 0.9971, 0.9850]}
    df_perf = pd.DataFrame(performance_data)
    fig_perf = px.bar(df_perf.sort_values('F1-Score', ascending=True), x='F1-Score', y='Model', orientation='h', title="Model F1-Scores (Higher is Better)", template='plotly_dark', text='F1-Score')
    st.plotly_chart(fig_perf, use_container_width=True)
    st.markdown("The results show that **XGBoost** and **Random Forest** are the top-performing classifiers.")


elif page == "Live Intrusion Detection":
    st.title("🕵️ Live Network Traffic Analysis")
    # (Content is the same as before, with the two data input options)
    chosen_model_name = st.selectbox("Step 1: Choose a Model for Prediction", list(models.keys()))
    model_to_use = models[chosen_model_name]
    st.markdown("---")
    st.subheader("Step 2: Provide Data for Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Option A: Upload Your Own Data")
        uploaded_file = st.file_uploader("Upload a CSV file in the WSN-DS format.", type="csv")
        if uploaded_file:
            st.session_state.df_to_process = pd.read_csv(uploaded_file)
    with col2:
        st.markdown("#### Option B: Generate a Simulation")
        num_rows = st.slider("Select number of packets to generate:", 500, 5000, 1000)
        if st.button("Generate & Predict"):
            with st.spinner('Generating simulated WSN traffic...'):
                st.session_state.df_to_process = generate_random_data(num_rows)
    
    if 'df_to_process' in st.session_state and st.session_state.df_to_process is not None:
        display_dashboard(st.session_state.df_to_process, model_to_use, chosen_model_name)
