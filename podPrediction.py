import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest
import openai
from pathlib import Path
import streamlit as st

# Retrieve API Key from Streamlit Secrets
openai_key = st.secrets["OPENAI_API_KEY"]

# Ensure API key is set
if not openai_key:
    raise Exception("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI Client
ai_client = openai.OpenAI(api_key=openai_key)

# Streamlit UI Header
st.title("Smart Performance Forecasting for OpenShift Pods")

# App Description
st.write(
    "This application predicts the optimal number of pods required in OpenShift based on LoadRunner performance data. "
    "It considers factors like TPS (transactions per second), CPU and memory usage per pod, and threshold limits "
    "to recommend the best scaling strategy."
)

# Add some spacing before showing the name
st.markdown("<br>", unsafe_allow_html=True)

# Display developer name in small font at the bottom
st.markdown(
    "<p style='font-size:18px; text-align:center; color:gray;'>Developed by Devesh Kumar</p>",
    unsafe_allow_html=True
)

# Upload CSV File (LoadRunner Report)
uploaded_csv = st.file_uploader("Upload LoadRunner Performance Report", type=["csv"])

if uploaded_csv:
    # Load Data
    perf_data = pd.read_csv(uploaded_csv)

    # Verify Columns
    required_cols = ["TPS", "CPU_Cores", "Memory_GB", "ResponseTime_ms", "CPU_Load", "Memory_Load"]
    if not all(col in perf_data.columns for col in required_cols):
        st.error("Missing required columns in uploaded CSV.")
        st.stop()

    st.write("Sample Data Preview:", perf_data.head())

    # Define Model Features & Target Variables
    model_inputs = ["TPS", "CPU_Cores", "Memory_GB", "ResponseTime_ms"]
    cpu_target = "CPU_Load"
    memory_target = "Memory_Load"

    # Extract Features & Targets
    X_trainset = perf_data[model_inputs]
    y_cpu = perf_data[cpu_target]
    y_memory = perf_data[memory_target]

    # Data Splitting for Model Training
    X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(X_trainset, y_cpu, test_size=0.2, random_state=42)
    X_train_mem, X_test_mem, y_memory_train, y_memory_test = train_test_split(X_trainset, y_memory, test_size=0.2, random_state=42)

    # Train Performance Estimation Models
    cpu_forecast_model = LinearRegression().fit(X_train, y_cpu_train)
    memory_forecast_model = LinearRegression().fit(X_train, y_memory_train)

    # Model Accuracy Evaluation
    cpu_accuracy = r2_score(y_cpu_test, cpu_forecast_model.predict(X_test))
    memory_accuracy = r2_score(y_memory_test, memory_forecast_model.predict(X_test))

    st.write(f"Model Accuracy: CPU Forecast R² Score: {cpu_accuracy:.2f}, Memory Forecast R² Score: {memory_accuracy:.2f}")

    # Function to Estimate Optimal Pod Count
    def compute_pod_estimate(expected_tps, cpu_allocation, memory_allocation, response_target, max_cpu=75, max_memory=75):
        for pod_count in range(1, 50):
            avg_tps_per_pod = expected_tps / pod_count
            test_input = pd.DataFrame([[avg_tps_per_pod, cpu_allocation, memory_allocation, response_target]], columns=model_inputs)

            projected_cpu = cpu_forecast_model.predict(test_input)[0]
            projected_memory = memory_forecast_model.predict(test_input)[0]

            if projected_cpu <= max_cpu and projected_memory <= max_memory:
                return pod_count, projected_cpu, projected_memory
        return None, None, None

    # User Input for Prediction
    user_tps = st.slider("Expected TPS:", min_value=10, max_value=200, step=10, value=40)
    user_cpu = st.slider("CPU per Pod:", min_value=1, max_value=4, step=1, value=1)
    user_memory = st.slider("Memory per Pod (GB):", min_value=2, max_value=8, step=1, value=2)
    user_response_time = st.slider("Expected Response Time (ms):", min_value=100, max_value=500, step=10, value=200)

    estimated_pods, estimated_cpu, estimated_memory = compute_pod_estimate(user_tps, user_cpu, user_memory, user_response_time)

    if estimated_pods:
        st.write(f"### Projected Pods Needed: {estimated_pods}")
        st.write(f"Projected CPU Utilization per Pod: {estimated_cpu:.2f}%")
        st.write(f"Projected Memory Utilization per Pod: {estimated_memory:.2f}%")
    else:
        st.write("No valid pod configuration found within limits.")

    # AI Performance Optimization Insights
    if estimated_pods is not None:
        performance_query = (
            f"Our OpenShift system processes {user_tps} TPS with an expected response time of {user_response_time} ms. "
            f"We estimated {estimated_pods} pods will be required with {estimated_cpu:.2f}% CPU "
            f"and {estimated_memory:.2f}% memory utilization per pod. "
            f"LoadRunner results indicate potential latency risks. Suggest performance improvements."
        )

        ai_response = ai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": performance_query}],
            temperature=0.2,
            max_tokens=500
        )

        st.write("### AI Performance Recommendations:")
        st.write(ai_response.choices[0].message.content)

    # Anomaly Detection on Performance Data
    anomaly_detector = IsolationForest(contamination=0.1)
    perf_data["anomaly_label"] = anomaly_detector.fit_predict(perf_data[["TPS", "CPU_Load", "Memory_Load"]])

    outlier_records = perf_data[perf_data["anomaly_label"] == -1]
    if not outlier_records.empty:
        st.write("### Detected Anomalies:", outlier_records)
    else:
        st.write("### No Anomalies Detected.")

    # Visualization of CPU vs TPS
    fig, ax = plt.subplots()
    sns.lineplot(x=perf_data["TPS"], y=perf_data["CPU_Load"], marker="o", ax=ax)
    ax.axhline(y=75, color="red", linestyle="--", label="75% CPU Threshold")
    ax.set_title("CPU Load vs TPS")
    ax.set_xlabel("Transactions Per Second (TPS)")
    ax.set_ylabel("CPU Load (%)")
    ax.legend()
    st.pyplot(fig)
