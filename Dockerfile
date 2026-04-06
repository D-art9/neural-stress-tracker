# Use official Python lightweight image
FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install Python dependencies
# Using --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Configure Streamlit server settings for cloud
# Enable CORS and disable file watcher for performance
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_CORS=true
ENV STREAMLIT_SERVER_HEADLESS=true

# Launch the application
ENTRYPOINT ["python", "-m", "streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
