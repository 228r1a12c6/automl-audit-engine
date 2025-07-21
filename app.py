import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from ml_engine import (
    load_model_and_metadata,
    predict_and_check_drift,
    save_history,
    load_history
)

# --- CONFIG ---
MODEL_PATH = "models/model.joblib"
METADATA_PATH = "models/metadata.json"
HISTORY_PATH = "history/history.csv"

st.set_page_config(page_title="AutoML Drift Detector", layout="wide")

# --- SIDEBAR ---
st.sidebar.image("https://avatars.githubusercontent.com/u/9919?s=200&v=4", width=80)
st.sidebar.markdown("### 👤 Yash Tandle")
st.sidebar.markdown("##### 🔍 AutoML Drift Monitoring")
st.sidebar.markdown("---")

history_df = load_history(HISTORY_PATH)
total_drifts = history_df["drift"].sum() if not history_df.empty else 0
last_retrain = history_df["timestamp"].iloc[-1] if not history_df.empty else "N/A"

st.sidebar.metric("⚠️ Total Drifts Detected", int(total_drifts))
st.sidebar.metric("🕒 Last Retrain", last_retrain)

# --- HEADER ---
st.title("🧠 AutoML Audit Engine")
st.caption("Monitor, detect, and act upon model drift.")

# --- MODEL CHECK AND LOAD (CRUCIAL PLACEMENT) ---
# This part ensures the model and metadata are loaded BEFORE they are used.
if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    st.error("❌ Base model not found. Please train it using `baseline_train.py`.")
    st.stop()

model, metadata = load_model_and_metadata(MODEL_PATH, METADATA_PATH)
# Now 'metadata' is loaded and ready to be used!

# --- Baseline Model Performance Section (CORRECTED PLACEMENT) ---
st.header("📊 Baseline Model Performance")
if metadata: 
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Accuracy", value=f"{metadata.get('accuracy', 0):.2f}")
    with col2:
        st.metric(label="Macro Precision", value=f"{metadata.get('precision_macro', 0):.2f}")
    with col3:
        st.metric(label="Macro Recall", value=f"{metadata.get('recall_macro', 0):.2f}")
    with col4:
        st.metric(label="Macro F1-Score", value=f"{metadata.get('f1_macro', 0):.2f}")
    
    st.markdown("---")
    st.subheader("Detailed Classification Report (Baseline)")
    st.json(metadata.get("classification_report", {}))
    
else:
    st.warning("Baseline model performance metrics not available. Please ensure baseline_train.py runs successfully.")

st.markdown("---")

# --- UPLOAD TEST DATA ---
test_file = st.file_uploader("📁 Upload Test Data CSV")

if test_file:
    df = pd.read_csv(test_file)
    if 'target' not in df.columns:
        st.error("❌ 'target' column missing in uploaded data.")
        st.stop()

    X_test = df.drop("target", axis=1)
    y_test = df["target"]

    # Ensure baseline_accuracy is retrieved from the loaded metadata for consistency
    baseline_accuracy_for_drift = metadata.get('accuracy', 0) # Use 'accuracy' key now, not 'baseline_accuracy'

    y_pred, current_accuracy, drift_detected = predict_and_check_drift(
        model, X_test, y_test, baseline_accuracy_for_drift # Use the correct key here
    )

    # Make sure save_history also uses the correct baseline accuracy key
    save_history(HISTORY_PATH, baseline_accuracy_for_drift, current_accuracy, drift_detected)

    # --- STATUS BANNER ---
    if drift_detected:
        st.warning(f"⚠️ Drift Detected! Accuracy dropped to {current_accuracy:.2f}. Retraining recommended.")
    else:
        st.success(f"✅ Model Healthy. Current Accuracy: {current_accuracy:.2f}.")

    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    # Ensure this also uses the correct key 'accuracy'
    col1.metric("📊 Baseline Accuracy", f"{baseline_accuracy_for_drift:.2f}") 
    col2.metric("📉 Current Accuracy", f"{current_accuracy:.2f}")
    col3.metric("⚠️ Drift Detected", "Yes" if drift_detected else "No")

    # --- RESULTS ---
    st.subheader("🧾 Prediction Results")
    df["Prediction"] = y_pred
    st.dataframe(df)
    st.download_button("📥 Download Results", df.to_csv(index=False).encode(), "predictions.csv")

    # --- DRIFT HISTORY CHART ---
    st.subheader("📈 Drift History")
    # Make sure history_df loading is updated to use 'accuracy' if your save_history uses it
    # For now, assuming history_df will eventually be updated too, or continue using baseline_accuracy if it's fine
    st.line_chart(history_df[["baseline_accuracy", "current_accuracy"]])

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2025 | Built by Yash Tandle 🚀")