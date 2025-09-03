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
    st.markdown("The most fundamental optimizer is **Gradient Descent**. It's like a hiker in a foggy valley who can only see the ground at their feet. They find the steepest downward slope (the **gradient**) and take a step. The size of that step is the **learning rate**.")
    
    col1, col2 = st.columns([0.7, 0.3])
    with col2:
        st.subheader("Controls")
        start_x = st.slider("Start Position (X)", -4.5, 4.5, 4.0, 0.5)
        learning_rate = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01)
        num_steps = st.slider("Number of Steps", 5, 50, 10, 1)

    # Gradient descent calculation
    path_x, path_y, path_z = [start_x], [0.0], [start_x**2]
    current_x = start_x
    for _ in range(num_steps):
        gradient = 2 * current_x  # Derivative of x^2 is 2x
        current_x = current_x - learning_rate * gradient
        path_x.append(current_x)
        path_y.append(0.0) # Keep it on one axis for simplicity
        path_z.append(current_x**2)
    
    with col1:
        fig_gd = go.Figure()
        # Loss curve
        fig_gd.add_trace(go.Scatter(x=x, y=x**2, mode='lines', name='Loss Function (y = x^2)'))
        # Steps
        fig_gd.add_trace(go.Scatter(x=path_x, y=path_z, mode='markers+lines', name='Optimizer Path', marker=dict(size=10, color='red')))
        fig_gd.update_layout(title=f"Finding the Minimum with Learning Rate: {learning_rate}", xaxis_title="Parameter Value", yaxis_title="Loss (Error)", template='plotly_dark')
        st.plotly_chart(fig_gd, use_container_width=True)

    st.markdown("""
    **Experiment!**
    - **High Learning Rate (e.g., > 0.9):** Notice how the optimizer overshoots the minimum and bounces around wildly.
    - **Low Learning Rate (e.g., < 0.1):** The optimizer takes tiny, slow steps. It's reliable but might take too long to converge.
    - **Good Learning Rate (e.g., ~0.3):** The optimizer finds the minimum efficiently.
    """)

    # --- 3. Advanced Optimizers ---
    st.markdown("---")
    st.header("3. Advanced Optimizers: The Smart Hikers")
    st.markdown("Modern optimizers improve upon Gradient Descent. **Adam** is the most popular because it's like a smart hiker with momentum and adaptive step sizes, allowing it to navigate complex landscapes much faster.")
    
    # Create a complex landscape (Beale function)
    def beale_function(x, y):
        return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

    x_b = np.linspace(-4.5, 4.5, 250)
    y_b = np.linspace(-4.5, 4.5, 250)
    X_b, Y_b = np.meshgrid(x_b, y_b)
    Z_b = beale_function(X_b, Y_b)

    # Simplified paths for visualization
    paths = {
        'SGD': [(3.5, 3.5), (3.0, 3.2), (2.8, 2.5), (2.2, 2.3), (1.8, 2.0), (1.5, 1.8), (1.0, 1.5), (0.7, 1.2), (0.4, 0.9), (0.2, 0.7), (0.1, 0.6), (0.0, 0.5)],
        'Momentum': [(3.5, 3.5), (3.1, 3.1), (2.5, 2.5), (1.8, 1.8), (0.9, 0.9), (0.2, 0.6), (-0.1, 0.5), (0.0, 0.5)],
        'Adam': [(3.5, 3.5), (3.2, 3.0), (2.8, 2.2), (2.0, 1.5), (1.0, 0.8), (0.3, 0.6), (0.0, 0.5)]
    }

    fig_adv = go.Figure()
    fig_adv.add_trace(go.Contour(z=Z_b, x=x_b, y=y_b, colorscale='gray', showscale=False))
    
    for name, path in paths.items():
        fig_adv.add_trace(go.Scatter(x=[p[0] for p in path], y=[p[1] for p in path], mode='lines+markers', name=name))
    
    fig_adv.update_layout(title="Visualizing Optimizer Paths on a Complex Surface", xaxis_title="Parameter 1", yaxis_title="Parameter 2", template='plotly_dark', height=600)
    st.plotly_chart(fig_adv, use_container_width=True)
    st.markdown("Notice how **Adam** takes the most direct and efficient route to the minimum (located at `(0.0, 0.5)`), while **SGD** struggles and takes a noisy path. **Momentum** is better than SGD but can overshoot. This is why Adam is the default choice for most deep learning tasks.")

# --- END OF NEW OPTIMIZATION ALGORITHMS PAGE ---
