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
    .stMetric { background-color: #0f3460; border-radius: 10px; padding: 15px; text-align: center; }
    .stMetric > label { color: #a0a0a0; } 
    .stMetric > div { color: #e94560; font-size: 1.75rem; }
    .stButton>button { width: 100%; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #c0c0c0; }
</style>
""", unsafe_allow_html=True)

# ----------------- MASTER DATA LOADING (UNCHANGED) -----------------
@st.cache_resource
def load_artifacts_from_double_zip(outer_zip_path='artifacts.zip'):
    """
    Loads all necessary .joblib files from a zip file that is nested inside another zip file.
    Structure: outer_zip -> inner_zip -> .joblib files
    """
    artifacts = {}
    inner_zip_name = 'artifacts.zip'  # The name of the zip file inside the outer zip
    try:
        with zipfile.ZipFile(outer_zip_path, 'r') as outer_z:
            inner_zip_bytes = outer_z.read(inner_zip_name)
            with zipfile.ZipFile(io.BytesIO(inner_zip_bytes), 'r') as inner_z:
                files_to_load = [
                    'scaler.joblib', 'feature_names.joblib', 'dt_model.joblib', 
                    'rf_model.joblib', 'lr_model.joblib', 'knn_model.joblib', 'xgb_model.joblib'
                ]
                for file_in_zip in files_to_load:
                    with inner_z.open(file_in_zip) as f:
                        key = file_in_zip.replace('.joblib', '')
                        artifacts[key] = joblib.load(f)
        return artifacts
    except Exception as e:
        st.error(f"Fatal Error: Could not load artifacts from the nested zip file. Ensure 'artifacts.zip' contains another 'artifacts.zip' with the model files. Error: {e}")
        st.stop()

# Load everything in one go
artifacts = load_artifacts_from_double_zip()
scaler = artifacts['scaler']
expected_features = artifacts['feature_names']
models = {
    "XGBoost": artifacts['xgb_model'],
    "Random Forest": artifacts['rf_model'],
    "Decision Tree": artifacts['dt_model'],
    "K-Nearest Neighbors": artifacts['knn_model'],
    "Logistic Regression": artifacts['lr_model']
}

# ----------------- NEW: DATA GENERATION FUNCTION -----------------
def generate_random_data(num_rows, attack_ratio=0.4):
    """Generates a DataFrame of random WSN data with simulated attacks."""
    data = {}
    # Generate baseline "normal" data with plausible ranges
    for feature in expected_features:
        if feature in ['Time', 'who CH']:
            data[feature] = np.random.randint(101000, 150000, size=num_rows)
        elif feature == 'Is_CH':
            data[feature] = np.random.choice([0, 1], size=num_rows, p=[0.9, 0.1])
        elif feature in ['ADV_S', 'ADV_R', 'JOIN_S', 'JOIN_R', 'SCH_S', 'SCH_R', 'Rank']:
            data[feature] = np.random.randint(0, 50, size=num_rows)
        elif feature in ['DATA_S', 'DATA_R']:
             data[feature] = np.random.randint(0, 150, size=num_rows)
        elif feature == 'Data_Sent_To_BS':
            # In normal traffic, data sent to BS should be similar to data sent
            data[feature] = data['DATA_S'] * np.random.uniform(0.9, 1.0, size=num_rows)
        elif feature in ['Dist_To_CH', 'dist_CH_To_BS']:
            data[feature] = np.random.uniform(10, 150, size=num_rows)
        else: # Handle any other columns like 'send_code', 'Expaned Energy'
            data[feature] = np.random.uniform(0, 1, size=num_rows)
    
    df = pd.DataFrame(data)

    # Inject simulated attacks into a fraction of the data
    num_attacks = int(num_rows * attack_ratio)
    attack_indices = np.random.choice(df.index, num_attacks, replace=False)

    for i in attack_indices:
        attack_type = np.random.choice(['Blackhole', 'Flooding'])
        if attack_type == 'Blackhole':
            # Simulate a blackhole: receives data but sends nothing to BS
            df.loc[i, 'DATA_R'] = np.random.randint(100, 200)
            df.loc[i, 'Data_Sent_To_BS'] = 0
        elif attack_type == 'Flooding':
            # Simulate flooding: very high traffic and energy use
            df.loc[i, 'DATA_S'] = np.random.randint(2000, 4000)
            df.loc[i, 'DATA_R'] = np.random.randint(2000, 4000)
            
    return df[expected_features] # Ensure column order is correct

# ----------------- UNIFIED DASHBOARD DISPLAY FUNCTION -----------------
def display_dashboard(df, model, model_name):
    """Processes a dataframe and displays the full analysis dashboard."""
    st.markdown("---")
    st.header(f"Analysis Dashboard (using {model_name})")
    try:
        df_display = df.copy()
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
        
        # (Visualization and Insights code)
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
    st.markdown("The project follows a structured data science lifecycle, from data acquisition to the deployment of this interactive web application.")
    
    # Workflow visualization using Graphviz
    st.graphviz_chart('''
        digraph {
            graph [rankdir="LR", bgcolor="#1a1a2e", fontcolor="white"];
            node [shape=box, style="filled,rounded", fillcolor="#0f3460", color="#e94560", fontcolor="white", penwidth=2];
            edge [color="white"];
            
            subgraph cluster_0 {
                label = "Phase 1: Research & Modeling";
                bgcolor = "#16213e";
                fontcolor="white";
                Data [label="1. Data Acquisition\n(WSN-DS Dataset)"];
                Preprocess [label="2. Data Preprocessing\n(Scaling & Encoding)"];
                Train [label="3. Comparative Model Training\n(5 Classifiers)"];
                Data -> Preprocess -> Train;
            }

            subgraph cluster_1 {
                label = "Phase 2: Deployment";
                bgcolor = "#16213e";
                fontcolor="white";
                Save [label="4. Package Artifacts\n(Models, Scaler)"];
                App [label="5. Build Web Framework\n(Streamlit)"];
                Deploy [label="6. Deploy on Cloud"];
                Save -> App -> Deploy;
            }
            Train -> Save [lhead=cluster_1, ltail=cluster_0, minlen=2];
        }
    ''')

    st.markdown("---")

    st.header("Key Predictive Features")
    st.markdown("A crucial insight from the analysis is understanding *which* network features are the most powerful indicators of an attack. The ensemble models (Random Forest, XGBoost) identified the following features as most important:")

    # Feature Importance Data from the report
    feature_importance_data = {
        'Feature': ['Data_Sent_To_BS', 'DATA_S', 'DATA_R', 'Expaned Energy', 'Dist_To_CH', 'Is_CH', 'ADV_S'],
        'Importance': [0.35, 0.18, 0.15, 0.12, 0.08, 0.05, 0.03]
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
        - **Data_Sent_To_BS:** The most critical feature. A massive drop in data reaching the Base Station is a classic sign of a **Blackhole** or **Grayhole** attack, where a malicious node is discarding packets.
        - **DATA_S & DATA_R:** An unusually high number of sent or received data packets is a strong indicator of a **Flooding** attack.
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

# --- START OF UPGRADED DETECTION PAGE ---
elif page == "Live Intrusion Detection":
    st.title("🕵️ Live Network Traffic Analysis")
    
    chosen_model_name = st.selectbox("Step 1: Choose a Model for Prediction", list(models.keys()))
    model_to_use = models[chosen_model_name]
    
    st.markdown("---")
    st.subheader("Step 2: Provide Data for Analysis")
    
    # Use st.session_state to hold the dataframe
    if 'df_to_process' not in st.session_state:
        st.session_state.df_to_process = None

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Option A: Upload Your Own Data")
        uploaded_file = st.file_uploader("Upload a CSV file in the WSN-DS format.", type="csv")
        if uploaded_file:
            st.session_state.df_to_process = pd.read_csv(uploaded_file)
            st.success("File uploaded! Results are below.")
    with col2:
        st.markdown("#### Option B: Generate a Simulation")
        num_rows = st.slider("Select number of packets to generate:", 1, 99999, 1, key="slider")
        if st.button("Generate & Predict"):
            with st.spinner('Generating simulated WSN traffic...'):
                st.session_state.df_to_process = generate_random_data(num_rows)
            st.success("Simulation complete! Results are below.")

    # If a dataframe exists in the session state, display the dashboard
    if st.session_state.df_to_process is not None:
        display_dashboard(st.session_state.df_to_process, model_to_use, chosen_model_name)
# --- END OF UPGRADED DETECTION PAGE ---








