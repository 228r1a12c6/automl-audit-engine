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

# --- MAIN HEADER ---
st.title("🧠 AutoML Audit Engine")
st.caption("Monitor, detect, and act upon model drift.")

# --- MODEL CHECK AND LOAD (This must remain at the top of the main content) ---
# It loads the model and metadata, which are needed for both baseline display AND prediction.
if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    st.error("❌ Base model not found. Please train it using `baseline_train.py`.")
    st.stop() # Stop the app if model/metadata are not found

model, metadata = load_model_and_metadata(MODEL_PATH, METADATA_PATH)
# Now 'model' and 'metadata' are loaded and ready to be used by subsequent sections.

# --- UPLOAD TEST DATA SECTION (Moved Higher Up) ---
st.header("⬆️ Upload New Data for Audit")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# --- Baseline Model Performance Section (Moved Below Upload) ---
# It's good to show this always, as static reference
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


# --- DYNAMIC CONTENT BASED ON UPLOADED FILE (Remains conditional) ---
if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(new_data.head())

        if 'target' not in new_data.columns:
            st.error("❌ 'target' column missing in uploaded data. Please ensure your CSV has a 'target' column.")
            st.stop() # Stop if target is missing after upload

        X_test = new_data.drop("target", axis=1)
        y_test = new_data["target"]

        baseline_accuracy_for_drift = metadata.get('accuracy', 0)

        y_pred, current_accuracy, drift_detected = predict_and_check_drift(
            model, X_test, y_test, baseline_accuracy_for_drift
        )

        save_history(HISTORY_PATH, baseline_accuracy_for_drift, current_accuracy, drift_detected)

        # --- STATUS BANNER ---
        if drift_detected:
            st.warning(f"⚠️ Drift Detected! Accuracy dropped to {current_accuracy:.2f}. Retraining recommended.")
        else:
            st.success(f"✅ Model Healthy. Current Accuracy: {current_accuracy:.2f}.")

        # --- CURRENT METRICS ---
        st.subheader("Current Model Performance")
        col1_curr, col2_curr = st.columns(2)
        col1_curr.metric("📉 Current Accuracy", f"{current_accuracy:.2f}")
        col2_curr.metric("⚠️ Drift Detected", "Yes" if drift_detected else "No")


        # --- RESULTS ---
        st.subheader("🧾 Prediction Results")
        new_data["Prediction"] = y_pred
        st.dataframe(new_data)
        st.download_button("📥 Download Results", new_data.to_csv(index=False).encode(), "predictions.csv", key="download_pred_btn")

        # --- DRIFT HISTORY CHART ---
        st.subheader("📈 Drift History")
        # Reload history to include the latest entry for the chart
        updated_history_df = load_history(HISTORY_PATH)
        st.line_chart(updated_history_df[["baseline_accuracy", "current_accuracy"]])

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a file with data.")
    except Exception as e:
        st.error(f"An error occurred while processing the uploaded file: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2025 | Built by Yash Tandle 🚀")