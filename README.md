# 🎬 Movie Recommender

![Vercel Compatible](https://img.shields.io/badge/Vercel-Serverless%20Functions-black)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)

A modern, fast, and lightweight movie recommendation engine utilizing TF-IDF cosine similarity, integrated seamlessly with the TMDB API. Built from the ground up to bypass Vercel's strict 500MB Lambda Size Constraints by using pure Python mathematics and pre-compressed JSON matrices instead of heavy Pandas/SciPy binaries.

### Features

- **Lightning-Fast Machine Learning:** Completely removed `scipy`, `scikit-learn`, and `pandas` dependencies. Similarity mappings are generated ahead of time using a local extract script and dumped into a highly compressed JSON dictionary for the Vercel cloud environment.
- **Premium Glassmorphism UI:** Complete CSS overhaul providing smooth hover animations, backdrop blurs, and premium grid card designs alongside dynamic Light/Dark side-bar theme toggling.
- **Dual Architecture:** The recommendation calculation engine (`FastAPI`) runs entirely isolated from the interactive streaming frontend (`Streamlit`), making it highly modular and scalable.

---

## 🚀 Easy Deployment

Due to architectural constraints on cloud serverless functions (like Vercel dropping WebSocket connections and having strict 500MB build sizes), this monorepo is designed to be split across two free-tier services.

### 1. Deploying the Backend (Vercel)

The backend is designed perfectly to fit inside a Vercel serverless function using `vercel.json` and `api/index.py`.

1. Log into [Vercel](https://vercel.com/) and create a new project.
2. Import this GitHub repository: `https://github.com/Tanmay-312/movie-recommender.git`
3. Vercel will auto-detect the configuration. In the Environment Variables section, add:
   - `TMDB_API_KEY`: Your personal TMDB developer key.
4. Click **Deploy**. Vercel will spin up the FastAPI server via Serverless Functions.

### 2. Deploying the Frontend (Streamlit Community Cloud)

Streamlit requires persistent WebSockets to function, which Vercel does not support. You must deploy the frontend separately on Streamlit Community Cloud.

1. Log into [Streamlit Share](https://share.streamlit.io).
2. Connect your GitHub repository.
3. Select `app.py` as your main entry point file.
4. Go to **Advanced Settings -> Secrets** and paste the deployed URL of your Vercel backend so the UI knows where to fetch movie data:
   ```toml
   API_URL="https://your-vercel-deployment-url.vercel.app"
   ```
5. Click **Deploy**.

---

## 💻 Local Development

If you want to run both the FastAPI backend and Streamlit frontend locally at the same time:

**1. Install dependencies**

```bash
# This lightweight list intentionally excludes heavy Data Science tools
pip install -r requirements.txt
```

**2. Configure Environment**  
Create a `.env` file in the root directory and add your TMDB key:

```env
TMDB_API_KEY=your_key_here
```

**3. Run FastAPI Backend**

```bash
uvicorn main:app --reload
# Starts the server on http://127.0.0.1:8000
```

**4. Run Streamlit Frontend (In a new terminal window)**

```bash
streamlit run app.py
# Bootstraps the UI automatically binding to local API instance
```

---

## 🛠️ The ML Data Extraction Script

If your TF-IDF corpus needs to be retrained or expanded in the future, you will recreate the huge DataFrames (`df.pkl`) and SciPy Sparse Matrices (`tfidf_matrix.pkl`) locally with traditional Notebooks.

To convert those heavy pickled objects back down into the web-ready lightweight dictionary format expected by this Vercel deployment:

1. Re-install the heavy analytics tools locally: `pip install pandas scipy scikit-learn`
2. Run the extraction pipeline: `python extract_data.py`.
3. This creates a highly compressed `tfidf_lite.json`.
4. **Delete the huge `.pkl` and `.csv` files** before pushing your changes to GitHub, otherwise Vercel's 500MB Lambda limit will crash your next deployment!
