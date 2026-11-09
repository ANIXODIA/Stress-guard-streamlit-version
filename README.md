# StressGuard AI

Streamlit app that estimates a stress category (Rendah / Sedang / Tinggi) from 8 daily indicators,
using a Random Forest trained on the Student Early Stress Detection dataset. Not a medical diagnosis.

    pip install -r requirements.txt
    streamlit run app.py

Files: `app.py` (whole app), `stress_model.json` (model), `model_meta.json` (metrics), `requirements.txt`.
Deploy: push to GitHub, then Streamlit Community Cloud -> New app -> pick this repo and `app.py`.
