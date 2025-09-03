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
    .stMetric { background-color: #0f3460; border-radius: 10px; padding: 15px; }
    .stMetric > label { color: #a0a0a0; } .stMetric > div { color: #e94560; }
</style>
""", unsafe_allow_html=True)

# ----------------- MASTER DATA LOADING FUNCTION (UPGRADED FOR DOUBLE ZIP) -----------------
@st.cache_resource
def load_artifacts_from_double_zip(outer_zip_path='artifacts.zip'):
    """
    Loads all necessary .joblib files from a zip file that is nested inside another zip file.
    Structure: outer_zip -> inner_zip -> .joblib files
    """
    artifacts = {}
    inner_zip_name = 'artifacts.zip'  # The name of the zip file inside the outer zip

    try:
        # 1. Open the outer zip file from the repository
        with zipfile.ZipFile(outer_zip_path, 'r') as outer_z:
            
            # 2. Read the inner zip file into memory as bytes
            inner_zip_bytes = outer_z.read(inner_zip_name)
            
            # 3. Use io.BytesIO to treat the in-memory bytes as a file, and open it as a zip archive
            with zipfile.ZipFile(io.BytesIO(inner_zip_bytes), 'r') as inner_z:
                
                # 4. Now, load all the .joblib files from the inner zip archive
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
        st.error(f"Fatal Error: Could not load artifacts from the nested zip file structure. Please ensure 'artifacts.zip' contains another 'artifacts.zip' with the model files. Error: {e}")
        st.stop()

# Load everything in one go using the new function
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

# ----------------- UI & APP LOGIC (Remains the same) -----------------
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
    st.dataframe(df_perf.set_index('Model'))
    st.markdown("The results show that **XGBoost** and **Random Forest** are the top-performing classifiers.")

elif page == "Live Intrusion Detection":
    st.title("🕵️ Live Network Traffic Analysis")
    chosen_model_name = st.selectbox("Choose a Model for Prediction", list(models.keys()))
    model = models[chosen_model_name]
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df_test = pd.read_csv(uploaded_file)
        try:
            df_display = df_test.copy()
            
            if ' id' in df_display.columns:
                df_display = df_display.drop(' id', axis=1)

            X_test = df_display[expected_features]
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
            attack_labels = {3: 'Normal Traffic', 0: 'Blackhole Attack', 1: 'Flooding Attack', 2: 'Grayhole Attack', 4: 'Scheduling Attack'}
            
            df_with_predictions = df_test.copy()
            df_with_predictions['Prediction'] = [attack_labels.get(p, 'Unknown') for p in predictions]
            
            st.header(f"Analysis Dashboard (using {chosen_model_name})
