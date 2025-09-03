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

# ----------------- MASTER DATA LOADING FUNCTION (CORRECTED) -----------------
@st.cache_resource
def load_artifacts_from_zip(zip_path='artifacts.zip/artifacts.zip'):
    """Loads all necessary .joblib files from a nested folder within a single zip archive."""
    artifacts = {}
    # --- START OF CORRECTION ---
    # Define the path to the files INSIDE the zip archive, including the nested folder.
    # This must match the folder name inside your main zip file.
    nested_path = 'artifacts.zip/'
    # --- END OF CORRECTION ---
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # List of the base filenames to load
            files_to_load = [
                'scaler.joblib', 'feature_names.joblib', 'dt_model.joblib', 
                'rf_model.joblib', 'lr_model.joblib', 'knn_model.joblib', 'xgb_model.joblib'
            ]
            for base_filename in files_to_load:
                # --- START OF CORRECTION ---
                # Construct the full path to the file within the zip archive
                full_path_in_zip = nested_path + base_filename
                # --- END OF CORRECTION ---
                
                with z.open(full_path_in_zip) as f:
                    # The key should be the base filename without the extension
                    key = base_filename.replace('.joblib', '')
                    artifacts[key] = joblib.load(f)
        return artifacts
    except Exception as e:
        st.error(f"Fatal Error: Could not load artifacts from '{zip_path}'. Ensure the file and its nested structure ('{nested_path}') are correct. Error: {e}")
        st.stop()

# Load everything in one go
artifacts = load_artifacts_from_zip()
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
            
            # Drop the ' id' column (with space) from the uploaded data, if it exists
            if ' id' in df_display.columns:
                df_display = df_display.drop(' id', axis=1)

            X_test = df_display[expected_features]
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
            attack_labels = {3: 'Normal Traffic', 0: 'Blackhole Attack', 1: 'Flooding Attack', 2: 'Grayhole Attack', 4: 'Scheduling Attack'}
            
            # Add prediction column back for display
            df_with_predictions = df_test.copy()
            df_with_predictions['Prediction'] = [attack_labels.get(p, 'Unknown') for p in predictions]
            
            st.header(f"Analysis Dashboard (using {chosen_model_name})")
            st.dataframe(df_with_predictions)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

