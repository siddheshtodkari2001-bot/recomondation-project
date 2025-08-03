# 🚀 Quick Start Guide

## Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Open Your Browser
The app will automatically open at `http://localhost:8501`

---

## Alternative Quick Start Options

### Windows Users
Double-click `run_app.bat` or run:
```cmd
run_app.bat
```

### Mac/Linux Users
```bash
chmod +x run_app.sh
./run_app.sh
```

### Test the System First
```bash
python run_demo.py
```

---

## What You'll See

1. **Sidebar Controls:**
   - Select any user ID from the dropdown
   - Adjust number of recommendations (5-20)
   - Toggle implicit feedback mode
   - Retrain the model

2. **Main Dashboard:**
   - Dataset statistics and visualizations
   - Personalized movie recommendations
   - Model performance metrics
   - Detailed analysis sections

3. **Key Features:**
   - Automatic MovieLens dataset download
   - Real-time recommendation generation
   - Interactive visualizations
   - Fallback to popular items for new users

---

## Troubleshooting

- **First run is slow:** The app downloads the MovieLens dataset (~1MB)
- **Port already in use:** Change the port with `streamlit run app.py --server.port 8502`
- **Memory issues:** Close other applications or reduce model parameters in `config.py`

---

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the code in `app.py` and `utils.py`
- Customize parameters in `config.py`
- Add your own datasets or algorithms

**Enjoy exploring the recommendation system! 🎬** 