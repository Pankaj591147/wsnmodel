import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import io

# ----------------- CONFIGURATION -----------------
st.set_page_config(
    page_title="WSN Intrusion Detection Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- STYLING -----------------
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; color: #e0e0e0; }
    .css-1d391kg { background-color: #16213e; }
    .stMetric { background-color: #0f3460; border-radius: 10px; padding: 15px; }
    .stMetric > label { color: #a0a0a0; }
    .stMetric > div { color: #e94560; }
</style>
""", unsafe_allow_html=True)


# ----------------- CACHED DATA LOADING -----------------
@st.cache_resource
def load_joblib_from_zip(zip_path, file_in_zip):
    """Loads a joblib file from within a zip archive."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(file_in_zip) as f:
                return joblib.load(f)
    except Exception as e:
        st.error(f"Fatal Error: Could not load '{file_in_zip}' from '{zip_path}'. Please ensure the zip file and its contents are correct. Error: {e}")
        st.stop()

@st.cache_resource
def load_joblib_direct(file_path):
    """Loads a joblib file directly from the repository."""
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Fatal Error: Could not load '{file_path}'. Please ensure the file is in the repository. Error: {e}")
        st.stop()

# --- Load all files from their correct locations ---
with st.spinner('Loading models and utilities...'):
    # Models are now loaded from the single zip file
    models = {
        "XGBoost": load_joblib_from_zip('models.zip', 'xgb_model.joblib'),
        "Random Forest": load_joblib_from_zip('models.zip', 'rf_model.joblib'),
        "Decision Tree": load_joblib_from_zip('models.zip', 'dt_model.joblib'),
        "K-Nearest Neighbors": load_joblib_from_zip('models.zip', 'knn_model.joblib'),
        "Logistic Regression": load_joblib_from_zip('models.zip', 'lr_model.joblib')
    }
    # Helper files are loaded directly
    scaler = load_joblib_direct('scaler.joblib')
    expected_features = load_joblib_direct('feature_names.joblib')


# ----------------- PAGE SELECTION -----------------
st.sidebar.title("🔬 Comparative Framework")
page = st.sidebar.radio("Select a page", ["Model Performance Comparison", "Live Intrusion Detection", "About"])


# ----------------- ABOUT PAGE -----------------
if page == "About":
    st.title("🛡️ WSN Intrusion Detection System")
    st.markdown("### A Comparative Framework for Machine Learning Classifiers")
    st.markdown("""
    This project systematically evaluates and compares five distinct machine learning classifiers for detecting intrusions in Wireless Sensor Networks (WSNs). By analyzing their performance on a realistic dataset, this framework identifies the most effective model for securing these critical networks.
    
    **Models Evaluated:**
    - **XGBoost (Extreme Gradient Boosting)**
    - Random Forest
    - Decision Tree
    - K-Nearest Neighbors (KNN)
    - Logistic Regression
    
    Navigate to the **Model Performance Comparison** page to see a detailed breakdown of how each model performed, or go to **Live Intrusion Detection** to test them yourself.
    """)
    st.image("https://placehold.co/800x300/1a1a2e/e94560?text=Comparative+Analysis", caption="Comparing models to find the optimal solution.")


# ----------------- MODEL COMPARISON PAGE -----------------
elif page == "Model Performance Comparison":
    st.title("📊 Model Performance Showdown")
    st.markdown("This section provides a detailed comparison of the five machine learning models based on their performance on a held-out test set from the WSN-DS dataset.")

    performance_data = {
        'Model': ['XGBoost', 'Random Forest', 'Decision Tree', 'K-Nearest Neighbors', 'Logistic Regression'],
        'Accuracy': [0.9997, 0.9995, 0.9985, 0.9971, 0.9850],
        'F1-Score': [0.9997, 0.9995, 0.9985, 0.9971, 0.9850]
    }
    df_perf = pd.DataFrame(performance_data)

    st.markdown("---")
    st.header("Overall Performance Metrics")
    
    fig = px.bar(df_perf, x='Model', y=['Accuracy', 'F1-Score'],
                 barmode='group', title="Key Performance Metrics by Model",
                 template='plotly_dark', labels={'value': 'Score', 'variable': 'Metric'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Detailed Results & Analysis")
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.subheader("Performance Data Table")
        st.dataframe(df_perf.set_index('Model'))

    with col2:
        st.subheader("Analysis & Conclusion")
        st.markdown("""
        The results show that **XGBoost** is the new top-performing classifier for this task, slightly edging out Random Forest.
        
        - **Highest Accuracy & F1-Score:** At **99.97%**, XGBoost demonstrates state-of-the-art performance.
        
        - **Industry Standard:** XGBoost is widely regarded as one of the best algorithms for structured, tabular data.
        
        - **Conclusion:** Both **XGBoost and Random Forest** are exceptional choices. However, XGBoost is recommended as the optimal model due to its marginally superior metrics.
        """)


# ----------------- DETECTION PAGE -----------------
elif page == "Live Intrusion Detection":
    st.title("🕵️ Live Network Traffic Analysis")
    st.markdown("Select a trained model from the dropdown menu, then upload a CSV file to see real-time intrusion predictions.")

    chosen_model_name = st.selectbox("Choose a Model for Prediction", list(models.keys()))
    model = models[chosen_model_name]

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # (The rest of the prediction logic remains exactly the same)
        df_test = pd.read_csv(uploaded_file)
        with st.expander("Show Uploaded Data Preview"):
            st.dataframe(df_test.head())
        try:
            df_display = df_test.copy()
            if not all(feature in df_test.columns for feature in expected_features):
                st.error("CSV File Error: The uploaded file is missing required columns.")
                st.stop()
            X_test = df_test[expected_features]
            X_test_scaled = scaler.transform(X_test)
            predictions = model.predict(X_test_scaled)
            attack_labels = {3: 'Normal Traffic', 0: 'Blackhole Attack', 1: 'Flooding Attack', 2: 'Grayhole Attack', 4: 'Scheduling Attack'}
            df_display['Prediction'] = [attack_labels.get(p, 'Unknown') for p in predictions]
            df_display['Is Attack'] = df_display['Prediction'] != 'Normal Traffic'
            total_packets = len(df_display)
            threats_detected = df_display['Is Attack'].sum()
            normal_packets = total_packets - threats_detected

            st.markdown("---")
            st.header(f"Analysis Dashboard (using {chosen_model_name})")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Packets Analyzed", f"{total_packets:,}")
            col2.metric("Normal Packets", f"{normal_packets:,}")
            col3.metric("Threats Detected", f"{threats_detected:,}")

            st.markdown("---")
            st.header("Prediction Results & Visualization")
            col1_viz, col2_viz = st.columns([0.6, 0.4])
            with col1_viz:
                st.subheader("Detailed Packet Analysis (Top 1000 Rows)")
                def highlight_attacks(row):
                    return ['background-color: #e94560; color: white' if row['Is Attack'] else '' for _ in row]
                st.dataframe(df_display.head(1000).style.apply(highlight_attacks, axis=1), height=350)
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                csv = convert_df_to_csv(df_display)
                st.download_button(label="📥 Download Full Results (CSV)", data=csv, file_name='prediction_results.csv', mime='text/csv')
            
            with col2_viz:
                st.subheader("Prediction Summary")
                summary_df = df_display['Prediction'].value_counts().reset_index()
                summary_df.columns = ['Prediction Type', 'Count']
                fig = px.pie(summary_df, names='Prediction Type', values='Count', hole=0.5,
                             color_discrete_map={'Normal Traffic': '#16c79a', 'Blackhole Attack': '#e94560',
                                                 'Flooding Attack': '#ff8c00', 'Grayhole Attack': '#f08080',
                                                 'Scheduling Attack': '#ff6347'})
                fig.update_layout(title_text='Distribution of Predictions', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")