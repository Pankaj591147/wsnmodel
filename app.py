import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import zipfile
import io

# ----------------- CONFIGURATION & STYLING -----------------
st.set_page_config(page_title="WSN Intrusion Detection", page_icon="🤖", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; color: #e0e0e0; }
    .css-1d391kg { background-color: #16213e; }
    .stMetric { background-color: #0f3460; border-radius: 10px; padding: 15px; text-align: center; }
    .stMetric > label { color: #a0a0a0; } 
    .stMetric > div { color: #e94560; font-size: 1.75rem; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------- MASTER DATA LOADING FUNCTION -----------------
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

# ----------------- UI & APP LOGIC -----------------
st.sidebar.title("🔬 Comparative Framework")
page = st.sidebar.radio("Select a page", ["Model Performance Comparison", "Live Intrusion Detection", "About"])

if page == "About":
    st.title("🛡️ WSN Intrusion Detection System")
    st.markdown("### A Comparative Framework for Machine Learning Classifiers")
    st.markdown("This project systematically evaluates five ML classifiers to find the most effective model for securing Wireless Sensor Networks.")
    st.image("https://placehold.co/800x300/1a1a2e/e94560?text=Comparative+Analysis", caption="Comparing models to find the optimal solution.")

elif page == "Model Performance Comparison":
    st.title("📊 Model Performance Showdown")
    st.markdown("Comparison of models based on performance on the WSN-DS test set.")
    performance_data = {'Model': ['XGBoost', 'Random Forest', 'Decision Tree', 'K-Nearest Neighbors', 'Logistic Regression'],
                        'Accuracy': [0.9997, 0.9995, 0.9985, 0.9971, 0.9850],
                        'F1-Score': [0.9997, 0.9995, 0.9985, 0.9971, 0.9850]}
    df_perf = pd.DataFrame(performance_data)
    
    st.markdown("---")
    st.header("Overall Performance Metrics")
    fig = px.bar(df_perf.sort_values('F1-Score', ascending=True), 
                 x='F1-Score', y='Model', orientation='h',
                 title="Model F1-Scores (Higher is Better)",
                 template='plotly_dark', text='F1-Score')
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Insights")
    st.markdown("""
    - **Top Performers:** The **XGBoost** and **Random Forest** models are the clear winners, achieving near-perfect F1-Scores. This is expected as both are powerful ensemble methods well-suited for complex, tabular data.
    - **Best Overall Model:** **XGBoost** is recommended as the optimal model. Its gradient boosting algorithm, which learns from the errors of previous trees, gives it a slight edge in performance and makes it a state-of-the-art choice for this type of security application.
    """)


elif page == "
