🔍 AutoML Audit Engine
A production-ready Machine Learning Drift Detection and Retraining System, featuring real-time monitoring and an interactive dashboard. This project addresses the critical challenge of maintaining model performance in dynamic data environments by automatically detecting concept drift and enabling seamless model retraining.

🚀 Features
✅ Model Drift Detection: Automatically identifies performance degradation (e.g., accuracy drop) in your deployed ML models.

📊 Drift History Tracking: Visualizes historical drift events and model performance trends directly in the dashboard.

🔁 One-Click Retraining: Trigger model retraining effortlessly from the Streamlit UI to adapt to new data.

📈 Streamlit Dashboard: Provides a real-time, intuitive interface for monitoring model health and interacting with the system.

🛠️ Modular Backend: Designed with clear separation of concerns, making it ready for future integration with RESTful APIs (e.g., FastAPI/Flask).

🐳 Dockerized for Deployment: Packaged as a Docker image for consistent and easy deployment across various environments.

📂 Project Structure
automl_audit_engine/
├── app.py                  # Streamlit Dashboard UI
├── ml_engine.py            # Core ML Backend Functions (predict, retrain, drift detect)
├── baseline_train.py       # Script to train the initial baseline model
├── retrain.py              # Automated model retraining script
├── monitor.py              # Logic for continuous model drift monitoring
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image configuration for the application
├── README.md               # Project documentation (this file)
├── models/                 # Directory for saving trained models and metadata
├── data/                   # Sample and test data files
├── history/                # Logs and data related to drift detection history
├── .streamlit/             # Streamlit UI theme and configuration files
└── screenshots/            # UI screenshot images for documentation
📷 Project Screenshots
Dashboard Overview	Drift History Chart

Export to Sheets
💡 How It Works
Data Upload: Users upload a new CSV dataset via the interactive dashboard.

Drift Detection: The system automatically checks the current model's accuracy against the new data and compares it with the established baseline performance.

Drift Flagging: If a significant drop in accuracy is detected (indicating concept drift), the system flags the drift event.

Visualization: Drift history, accuracy trends, and other relevant metrics are visualized on the dashboard for immediate insights.

Retraining Trigger: Users can initiate model retraining directly from the dashboard, leveraging the new data to update the model.

⚙️ Tech Stack
Python: Primary programming language.

scikit-learn: For machine learning model development.

pandas: For data manipulation and analysis.

Streamlit: For building the interactive web dashboard.

Docker: For application containerization and simplified deployment.

🚀 Quick Start
Follow these steps to get the AutoML Audit Engine up and running on your local machine.

⏳ Run Locally
Clone the repository:

Bash

git clone https://github.com/228r1a12c6/automl_audit_engine.git
cd automl_audit_engine
Install dependencies:

Bash

pip install -r requirements.txt
Train the initial baseline model (run this only once):

Bash

python baseline_train.py
Launch the Streamlit dashboard:

Bash

streamlit run app.py
Your application should now be accessible in your web browser, typically at http://localhost:8501.

🐳 Run via Docker
Ensure Docker is installed and running on your system.

Build the Docker image:

Bash

docker build -t automl_audit_engine .
Run the Docker container:

Bash

docker run -p 7860:8501 automl_audit_engine
(Note: Streamlit defaults to port 8501, so we map external 7860 to internal 8501.)

Then, open your web browser and navigate to: http://localhost:7860

🛠️ Architecture Flow
Code snippet

graph TD
    A[Data Upload (via Streamlit UI)] --> B(Monitor.py - Drift Detection);
    B --> C{Accuracy Drop?};
    C -- Yes --> D(Retrain.py - Automated Retraining);
    C -- No --> E(Dashboard Display - Current Status);
    D --> F(Model Update - Save New Model);
    F --> E;
    E --> G(ML_Engine.py - Prediction/Drift Check);
    G --> B;
📣 Author
Built by Yash Tandle 🚀
[GitHub Profile](https://github.com/228r1a12c6)