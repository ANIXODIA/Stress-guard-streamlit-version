# 🌸 StressGuard AI (Streamlit version)

An early student-stress screening prototype. The user enters eight everyday indicators (sleep, study load, academic pressure, anxiety, depression, burnout, attendance and missed submissions) and a Random Forest model predicts one of three stress categories: **Rendah** (Low), **Sedang** (Medium) or **Tinggi** (High). The app then shows supportive, non-clinical suggestions.

> **Disclaimer:** This is a prototype, not a medical or diagnostic tool. Results do not replace the judgment of a psychologist or health professional.

**Live app:** _add your Streamlit Community Cloud URL here_

> The app interface is written in Indonesian (Bahasa Indonesia).

---

## Features

- Eight slider inputs with sensible defaults, grouped in a form so the model only runs when **Prediksi** is pressed
- Predicted stress category with a 3-level progress indicator, a short explanation and practical tips
- Campus counseling prompt shown when the result is "Tinggi"
- "About the model" section showing the algorithm, dataset, validation method and test metrics, read from `model_meta.json`
- Pure-Python inference: the trained Random Forest is exported to JSON, so **scikit-learn is not needed** at runtime and there are no version-mismatch problems

## Quick start

```bash
git clone https://github.com/ANIXODIA/Stress-guard-streamlit-version.git
cd Stress-guard-streamlit-version
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501.

## Input features

| Feature | Label in app | Range |
|---|---|---|
| `sleep_hours_per_day` | Average sleep per night (hours) | 4 – 10 |
| `study_hours_per_week` | Study hours per week | 2 – 35 |
| `academic_pressure_score` | Academic pressure | 0 – 10 |
| `anxiety_score` | Anxiety | 0 – 10 |
| `depression_score` | Depression | 0 – 10 |
| `burnout_score` | Burnout | 0 – 10 |
| `attendance_rate` | Attendance (0 low – 1 high) | 0.0 – 1.0 |
| `missed_submissions_count` | Missed assignments | 0 – 20 |

## Output classes

| Class | Meaning |
|---|---|
| `Rendah` | Low stress signals |
| `Sedang` | Medium stress signals |
| `Tinggi` | High stress signals, counseling support is suggested |

## Model

| | |
|---|---|
| Algorithm | Random Forest |
| Dataset | [Student Early Stress Detection](https://www.kaggle.com/datasets/rishavraj1232/student-early-stress-detection) by Rishav Raj (Kaggle) |
| Dataset size | 102,714 rows, 39 columns |
| Validation | 80/20 stratified split + 5-fold cross-validation |
| Test set | 20,543 rows |
| Accuracy (test) | 99.89% |
| Macro-F1 (test) | 99.84% |
| 5-fold CV macro-F1 | Random Forest 99.79%, Logistic Regression 99.78% |

These numbers measure how well the model matches the **dataset's labels**. They are not a measure of real-world diagnostic accuracy.

### How inference works

The trained forest is stored in `stress_model.json`. Each tree is stored as parallel arrays:

| Key | Meaning |
|---|---|
| `f` | feature index used at each node |
| `t` | split threshold |
| `l` / `r` | left / right child index (`-1` marks a leaf) |
| `v` | class probabilities at leaf nodes |

`predict_label()` in `streamlit_app.py` walks every tree from the root to a leaf (going left when `value <= threshold`), sums the leaf probabilities across all trees, and returns the class with the highest total.

## Project structure

```
.
├── streamlit_app.py        # Streamlit UI and JSON-based Random Forest inference
├── requirements.txt        # Python dependencies
├── model/
│   ├── stress_model.json   # Exported Random Forest (features, classes, trees)
│   └── model_meta.json     # Algorithm, dataset, validation and metrics shown in the app
└── README.md
```

`streamlit_app.py` loads the model from `model/stress_model.json` and `model/model_meta.json`. If either file is missing from `model/`, the app shows an error and stops.

## Configuration

Edit these constants near the top of `streamlit_app.py`:

| Constant | Purpose |
|---|---|
| `TEAM` | Team name shown in the footer |
| `CAMPUS` | Campus name shown in the footer |
| `COUNSELING_CONTACT` | Counseling contact shown for "Tinggi" results |
| `FIELDS` | Slider definitions (name, label, min, max, step, default) |
| `RESULTS` | Text, tips and closing message for each class |

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
3. Select this repository, branch `main`, and main file `streamlit_app.py`.
4. Click **Deploy**. Every push to `main` redeploys the app automatically.

## References

- [Streamlit documentation](https://docs.streamlit.io)
- Student Early Stress Detection dataset by Rishav Raj: https://www.kaggle.com/datasets/rishavraj1232/student-early-stress-detection

## License

_Add a license here (for example MIT) or remove this section._
