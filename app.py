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
st.set_page_config(page_title="WSN Intrusion Detection Framework", page_icon="🤖", layout="wide")
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
        st.error(f"Fatal Error: Could not load artifacts from '{zip_path}'. Error: {e}")
        st.stop()

artifacts = load_artifacts_from_zip()
scaler = artifacts['scaler']
expected_features = artifacts['feature_names']
models = { "XGBoost": artifacts['xgb_model'], "Random Forest": artifacts['rf_model'], "Decision Tree": artifacts['dt_model'], "K-Nearest Neighbors": artifacts['knn_model'], "Logistic Regression": artifacts['lr_model']}

# ----------------- DATA GENERATION & DASHBOARD FUNCTIONS (UNCHANGED) -----------------
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

def display_dashboard(df, model, model_name):
    st.markdown("---")
    st.header(f"Analysis Dashboard (using {model_name})")
    # (Dashboard display logic remains the same)

# ----------------- PAGE SELECTION -----------------
st.sidebar.title("🔬 Project Framework")
page = st.sidebar.radio("Select a page", ["About the Project", "Understanding Optimization", "Model Performance Comparison", "Live Intrusion Detection"])

# ----------------- ABOUT & COMPARISON PAGES (UNCHANGED) -----------------
if page == "About the Project":
    st.title("🛡️ About The WSN Intrusion Detection Project")
    # (Content is the same)
elif page == "Model Performance Comparison":
    st.title("📊 Model Performance Showdown")
    # (Content is the same)

# --- START OF NEW "UNDERSTANDING OPTIMIZATION" PAGE ---
elif page == "Understanding Optimization":
    st.title("🧠 How Do Models Learn? An Intro to Optimization")
    st.markdown("""
    At its core, "training" a machine learning model is a process of optimization. The goal is to find the best possible set of internal parameters (weights) for the model that minimizes its prediction errors. **Optimization Algorithms** are the engines that drive this process.
    """)

    st.header("The Hiker Analogy: Gradient Descent")
    st.markdown("""
    Imagine you are a hiker on a foggy mountain, and your goal is to reach the lowest point in the valley. This is exactly what an optimization algorithm does.
    - **The Mountain:** Represents the "loss landscape." Higher points mean higher prediction error.
    - **Your Position:** The current state of the model's parameters.
    - **The Valley Floor:** The point of minimum error—the best possible model.
    
    Because of the fog, you can't see the whole mountain. So, you look at the ground beneath your feet to find the steepest downward slope (the **gradient**) and take a step. You repeat this process until you can no longer go downhill. The size of your step is called the **learning rate**.
    """)

    st.markdown("---")
    st.header("Interactive Visualization: The Race to the Bottom")
    st.markdown("Select different optimization algorithms below to see how they navigate the loss landscape to find the minimum. Observe the differences in their paths.")

    # Create the 3D loss surface
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2 # A simple convex loss function
    
    surface = go.Surface(x=X, y=Y, z=Z, opacity=0.6, colorscale='viridis', showscale=False)
    
    # --- Simulate paths for different optimizers ---
    def simulate_path(optimizer_type):
        path_x, path_y = [-9], [9] # Starting point
        velocity_x, velocity_y = 0, 0
        m_x, m_y = 0, 0
        v_x, v_y = 0, 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for t in range(1, 20):
            grad_x, grad_y = 2 * path_x[-1], 2 * path_y[-1]
            
            if optimizer_type == 'SGD':
                learning_rate = 0.1
                # Add noise to simulate stochastic nature
                grad_x += np.random.randn() * 4
                grad_y += np.random.randn() * 4
                path_x.append(path_x[-1] - learning_rate * grad_x)
                path_y.append(path_y[-1] - learning_rate * grad_y)
            
            elif optimizer_type == 'Momentum':
                learning_rate, gamma = 0.1, 0.8
                velocity_x = gamma * velocity_x + learning_rate * grad_x
                velocity_y = gamma * velocity_y + learning_rate * grad_y
                path_x.append(path_x[-1] - velocity_x)
                path_y.append(path_y[-1] - velocity_y)
                
            elif optimizer_type == 'Adam':
                learning_rate = 0.6
                m_x = beta1 * m_x + (1 - beta1) * grad_x
                m_y = beta1 * m_y + (1 - beta1) * grad_y
                v_x = beta2 * v_x + (1 - beta2) * (grad_x**2)
                v_y = beta2 * v_y + (1 - beta2) * (grad_y**2)
                m_hat_x, m_hat_y = m_x / (1 - beta1**t), m_y / (1 - beta1**t)
                v_hat_x, v_hat_y = v_x / (1 - beta2**t), v_y / (1 - beta2**t)
                path_x.append(path_x[-1] - learning_rate * m_hat_x / (np.sqrt(v_hat_x) + eps))
                path_y.append(path_y[-1] - learning_rate * m_hat_y / (np.sqrt(v_hat_y) + eps))
                
        path_z = np.array(path_x)**2 + np.array(path_y)**2
        return path_x, path_y, path_z

    # UI for selecting optimizers
    optimizer_options = st.multiselect(
        "Select algorithms to compare:",
        ['SGD', 'Momentum', 'Adam'],
        default=['SGD', 'Adam']
    )

    paths_to_plot = []
    colors = {'SGD': '#e94560', 'Momentum': '#ff8c00', 'Adam': '#16c79a'}

    for opt in optimizer_options:
        px, py, pz = simulate_path(opt)
        paths_to_plot.append(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers', 
                                          name=opt, line=dict(color=colors[opt], width=8), 
                                          marker=dict(size=5)))

    fig = go.Figure(data=[surface] + paths_to_plot)
    fig.update_layout(title='Optimization Paths on a Loss Surface', template='plotly_dark',
                      scene=dict(xaxis_title='Weight 1', yaxis_title='Weight 2', zaxis_title='Loss (Error)'),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Interpreting the Paths")
    if 'SGD' in optimizer_options:
        st.markdown("- **Stochastic Gradient Descent (SGD):** The path is noisy and erratic. Because it looks at only one data point at a time, its direction is jumpy, but it still makes progress towards the minimum.")
    if 'Momentum' in optimizer_options:
        st.markdown("- **Momentum:** This path is much smoother. It builds up speed in the correct direction, helping it to overcome small bumps and converge faster than standard SGD.")
    if 'Adam' in optimizer_options:
        st.markdown("- **Adam (Adaptive Moment Estimation):** This is often the fastest and most direct path. It combines the idea of momentum with adaptive learning rates, allowing it to take confident, intelligent steps towards the goal. It is the default choice for most deep learning problems today.")

# --- END OF NEW PAGE ---

elif page == "Live Intrusion Detection":
    st.title("🕵️ Live Network Traffic Analysis")
    # (Content is the same as before)
