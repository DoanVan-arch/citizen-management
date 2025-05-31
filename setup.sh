#!/bin/bash

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libopus-dev \
    libvpx-dev \
    libzbar0 \
    libzbar-dev \
    python3-zbar \
    zbar-tools

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Make the script executable
chmod +x setup.sh

echo "Setup complete! You can now run the app with: streamlit run streamlit_app.py"