# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train the baseline model (run this once during image build)
# This assumes baseline_train.py creates/saves the initial model
RUN python baseline_train.py

# Expose the port your Streamlit app will run on
EXPOSE 7860

# Command to run your Streamlit app, listening on all interfaces
# Added --server.enableCORS=false and --server.enableXsrfProtection=false for cloud deployment stability
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]