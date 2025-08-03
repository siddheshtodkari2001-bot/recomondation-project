#!/bin/bash

echo "🎬 Movie Recommendation System"
echo "================================"
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting Streamlit application..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""
streamlit run app.py 