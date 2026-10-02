# 🚀 Deployment Guide: Render (Backend) & Streamlit Cloud (Frontend)

This guide walks you through deploying the **Intelligent Vision Platform** across Render (FastAPI Backend) and Streamlit Community Cloud (Frontend UI).

---

## 1. ⚙️ Deploy Backend to Render

### Option A: Using Render Web Dashboard (Recommended)
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository: `https://github.com/BesthaMahesh/intelligent_vision_agent`.
4. Configure the Web Service settings:
   - **Name:** `intelligent-vision-backend`
   - **Region:** Any (e.g., Oregon or Frankfurt)
   - **Branch:** `main`
   - **Root Directory:** *(leave empty or set to root)*
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
5. **Environment Variables**:
   Under the **Environment Variables** section, add:
   - `GROQ_API_KEY`: `your_groq_api_key`
   - `OPENAI_API_KEY`: *(optional)*
   - `ANTHROPIC_API_KEY`: *(optional)*
   - `PYTHON_VERSION`: `3.11.9`
6. Click **Create Web Service**.
7. Once deployed, copy your Render URL (e.g., `https://intelligent-vision-backend.onrender.com`).

---

## 2. 🎨 Deploy Frontend to Streamlit Community Cloud

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **Create app**.
3. Select your GitHub repository:
   - **Repository:** `BesthaMahesh/intelligent_vision_agent`
   - **Branch:** `main`
   - **Main file path:** `frontend/app.py`
4. Expand **Advanced settings** (or Secrets):
   Add the following secret:
   ```toml
   BACKEND_API_URL = "https://intelligent-vision-backend.onrender.com"
   ```
5. Click **Deploy!**.

---

## 3. 🧪 Testing the Deployment

1. Open your Streamlit URL.
2. Sign in with your registered account or register a new one.
3. Upload an image in the **Image Analysis & Detection** view.
4. Interact with the **Visual Assistant (QA)** to ask questions about detected objects.
