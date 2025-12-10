# 🚀 Hotel Cancellation Predictor - Deployment Guide

## 📋 Prerequisites

Before deploying, make sure you have:
- ✅ Python 3.8 or higher installed
- ✅ All model files ready (`.pkl` files)
- ✅ Git installed (for GitHub deployment)

## 📦 Required Files

Your project directory should contain:

```
hotel-cancellation-predictor/
│
├── app.py                              # Main Streamlit app
├── requirements.txt                     # Python dependencies
├── random_forest_best_model.pkl        # Trained model (or your best model)
├── scaler.pkl                          # Feature scaler
├── label_encoders.pkl                  # Label encoders
├── model_features.csv                  # Feature list
├── feature_importance.csv              # Feature importance (optional)
├── README.md                           # Project documentation
└── .gitignore                          # Git ignore file
```

## 🔧 Local Testing

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (FREE & RECOMMENDED) ⭐

**Advantages:**
- 100% FREE
- Easy to deploy
- Automatic updates from GitHub
- Good for portfolios

**Steps:**

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/hotel-cancellation-predictor.git
   git push -u origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"

3. **Configure Deployment**
   - Repository: `YOUR_USERNAME/hotel-cancellation-predictor`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Wait for Deployment** (2-5 minutes)
   - Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

**⚠️ IMPORTANT for Streamlit Cloud:**
- Model files (`.pkl`) must be < 100MB each
- If files are too large, use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.pkl"
  git add .gitattributes
  git commit -m "Add Git LFS"
  ```

---

### Option 2: Hugging Face Spaces (FREE)

**Advantages:**
- FREE
- Good for ML projects
- Integration with Hugging Face ecosystem

**Steps:**

1. **Create Hugging Face Account**
   - Visit: https://huggingface.co/join

2. **Create New Space**
   - Go to: https://huggingface.co/new-space
   - Select "Streamlit" as SDK
   - Choose public or private

3. **Upload Files**
   - Upload all files from your project
   - Make sure `app.py` is in the root directory

4. **App is Live!**
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/hotel-predictor`

---

### Option 3: Render (FREE Tier Available)

**Steps:**

1. **Create Render Account**
   - Visit: https://render.com

2. **Create New Web Service**
   - Connect GitHub repository
   - Select "Python"

3. **Configure**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment

---

### Option 4: Heroku (PAID - Not Recommended for Free)

Heroku no longer has a free tier, so we recommend other options above.

---

## 🔒 .gitignore File

Create a `.gitignore` file to exclude unnecessary files:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Data files (if too large)
*.csv
*.xlsx
*.data

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
```

---

## 📝 README.md Template

Create a `README.md` for your repository:

```markdown
# 🏨 Hotel Booking Cancellation Predictor

AI-powered system to predict hotel booking cancellations and provide actionable recommendations.

## 🎯 Features
- Single booking prediction
- Batch prediction (CSV upload)
- Risk categorization (High/Medium/Low)
- Actionable recommendations
- Interactive visualizations

## 🚀 Live Demo
👉 [Try it here](YOUR_DEPLOYMENT_URL)

## 💻 Local Installation

1. Clone repository:
   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/hotel-cancellation-predictor.git
   cd hotel-cancellation-predictor
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Run app:
   \`\`\`bash
   streamlit run app.py
   \`\`\`

## 📊 Model Performance
- Recall: 85.2%
- Precision: 78.3%
- F2-Score: 83.1%

## 👨‍💻 Author
[Your Name] - Final Project 2024

## 📄 License
MIT License
```

---

## 🎨 Customization Tips

### 1. Change Theme
Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor="#1f77b4"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"
```

### 2. Add Custom Logo
- Add your logo image to the project
- Update `st.set_page_config()` in `app.py`:
  ```python
  st.set_page_config(
      page_title="Hotel Predictor",
      page_icon="🏨",  # or path to image
      ...
  )
  ```

### 3. Add Analytics (Optional)
Add Google Analytics to track usage.

---

## 🐛 Troubleshooting

### Issue: Model file too large for GitHub

**Solution 1: Use Git LFS**
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Track model files with LFS"
```

**Solution 2: Use external storage**
- Upload model to Google Drive
- Load model from URL in app

### Issue: ModuleNotFoundError

**Solution:**
- Make sure `requirements.txt` is complete
- Check Python version compatibility

### Issue: App crashes on Streamlit Cloud

**Solution:**
- Check logs in Streamlit Cloud dashboard
- Ensure all file paths are correct
- Verify model file integrity

---

## 📧 Support

If you encounter issues:
1. Check the logs in your deployment platform
2. Verify all files are uploaded correctly
3. Test locally first before deploying

---

## 🎉 Deployment Checklist

Before deploying, make sure:
- [ ] App runs locally without errors
- [ ] All model files are present
- [ ] requirements.txt is complete
- [ ] README.md is informative
- [ ] .gitignore is configured
- [ ] GitHub repository is public (for free deployment)
- [ ] Large files are handled with Git LFS (if needed)
- [ ] Test all features (single prediction, batch prediction)

---

**🚀 Ready to Deploy!**

Choose your platform and follow the steps above. Good luck!