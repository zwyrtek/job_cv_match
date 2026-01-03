#!/bin/bash

echo "========================================"
echo "Job-CV Matching Application"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Check if model exists
if [ ! -d "models" ] || [ ! -f "models/model.keras" ]; then
    echo ""
    echo "⚠ Model files not found!"
    echo "Please run 'python3 train_model.py' first to train the model."
    echo "Or copy the pre-trained models/ directory to this location."
    exit 1
fi

echo ""
echo "========================================"
echo "Starting Flask API server..."
echo "========================================"
echo ""
echo "The application will be available at:"
echo "  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Flask app
python3 app.py
