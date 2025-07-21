import pandas as pd
import os
import joblib
import json
from sklearn.metrics import accuracy_score
from datetime import datetime

def load_model_and_metadata(model_path, metadata_path):
    model = joblib.load(model_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return model, metadata

def predict_and_check_drift(model, X_test, y_test, baseline_accuracy, drift_threshold=0.1):
    y_pred = model.predict(X_test)
    current_accuracy = accuracy_score(y_test, y_pred)
    drift_detected = current_accuracy < (baseline_accuracy - drift_threshold)
    return y_pred, current_accuracy, drift_detected

def save_history(history_path, baseline_accuracy, current_accuracy, drift_detected):
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_accuracy": baseline_accuracy,
        "current_accuracy": current_accuracy,
        "drift": drift_detected
    }
    df_record = pd.DataFrame([record])

    if os.path.exists(history_path):
        df_record.to_csv(history_path, mode="a", header=False, index=False)
    else:
        df_record.to_csv(history_path, index=False)

def load_history(history_path):
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    else:
        return pd.DataFrame(columns=["timestamp", "baseline_accuracy", "current_accuracy", "drift"])
