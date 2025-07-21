import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
import numpy as np
import requests # <--- NEW: Import for sending notifications
from datetime import datetime # <--- ADD THIS LINE
def load_model_and_metadata(model_path: str, metadata_path: str):
    """
    Loads the machine learning model and its associated metadata.
    """
    try:
        model = joblib.load(model_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return model, metadata
    except FileNotFoundError:
        print(f"Error: Model or metadata file not found at {model_path} or {metadata_path}.")
        return None, None
    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        return None, None

def predict_and_check_drift(model, X_test: pd.DataFrame, y_test: pd.Series, baseline_accuracy: float, drift_threshold: float = 0.05):
    """
    Makes predictions using the model, calculates current accuracy,
    and checks for model drift against a baseline accuracy.
    """
    try:
        y_pred = model.predict(X_test)
        current_accuracy = accuracy_score(y_test, y_pred)
        
        drift_detected = (baseline_accuracy - current_accuracy) > drift_threshold

        return y_pred, current_accuracy, drift_detected
    except Exception as e:
        print(f"Error during prediction or drift check: {e}")
        # Return default/safe values in case of an error
        return pd.Series([0]*len(X_test)), 0.0, False # Return dummy predictions, 0 accuracy, no drift

def save_history(history_path: str, baseline_accuracy: float, current_accuracy: float, drift_detected: bool):
    """
    Saves the current run's accuracy and drift status to a history CSV file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "baseline_accuracy": baseline_accuracy,
        "current_accuracy": current_accuracy,
        "drift": drift_detected
    }])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    if os.path.exists(history_path):
        history_entry.to_csv(history_path, mode='a', header=False, index=False)
    else:
        history_entry.to_csv(history_path, mode='w', header=True, index=False)
    print(f"History saved to {history_path}")

def load_history(history_path: str):
    """
    Loads the prediction history from a CSV file.
    """
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    return pd.DataFrame(columns=["timestamp", "baseline_accuracy", "current_accuracy", "drift"])

# --- NEW FUNCTION: For sending Slack notifications ---
def send_slack_notification(message: str):
    """
    Sends a notification message to a Slack channel using a webhook URL.
    The webhook URL is retrieved from environment variables for security.
    """
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    if not slack_webhook_url:
        print("Error: SLACK_WEBHOOK_URL environment variable not set. Cannot send Slack notification.")
        return

    payload = {
        "text": f"🤖 AutoML Audit Engine Alert!\n\n{message}"
    }

    try:
        response = requests.post(
            slack_webhook_url,
            json=payload,
            timeout=10 # Set a timeout for the request to prevent long waits
        )
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        print("Slack notification sent successfully!")
    except requests.exceptions.Timeout:
        print("Error: Slack notification request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending Slack notification: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while sending Slack notification: {e}")