"""
StressGuard AI - Streamlit version.

Run:   pip install streamlit
       streamlit run streamlit_app.py

The prediction uses model/stress_model.json (the trained Random Forest exported to JSON),
so scikit-learn is NOT needed and there is no version-mismatch problem.
"""
import json
from pathlib import Path

import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "stress_model.json"
META_PATH = BASE / "model" / "model_meta.json"

TEAM = "Belum diisi"
CAMPUS = "Belum diisi"
COUNSELING_CONTACT = "[kontak konseling kampus]"
DISCLAIMER = "Ini bukan diagnosis medis. Hasil tidak menggantikan penilaian psikolog atau tenaga kesehatan."

# name, caption, min, max, step, default   (int vs float matters: Streamlit needs matching types)
FIELDS = [
    ("sleep_hours_per_day",      "Rata-rata tidur per malam (jam)", 4.0, 10.0, 0.5, 6.5),
    ("study_hours_per_week",     "Jam belajar per minggu",          2,   35,   1,   24),
    ("academic_pressure_score",  "Tekanan akademik (0 ringan - 10 sangat berat)", 0, 10, 1, 7),
    ("anxiety_score",            "Kecemasan (0 ringan - 10 sangat berat)",        0, 10, 1, 4),
    ("depression_score",         "Depresi (0 ringan - 10 sangat berat)",          0, 10, 1, 3),
    ("burnout_score",            "Kelelahan / burnout (0 ringan - 10 sangat berat)", 0, 10, 1, 5),
    ("attendance_rate",          "Kehadiran (0 rendah - 1 tinggi)", 0.0, 1.0, 0.01, 0.85),
    ("missed_submissions_count", "Jumlah tugas yang terlewat",      0,   20,   1,   2),
]

RESULTS = {
    "Rendah": {
        "level": 0, "icon": "🌱",
        "text": "Ritme harianmu tampak cukup terjaga. Tetap beri ruang untuk istirahat dan hal yang kamu sukai.",
        "tips": ["Pertahankan waktu tidur yang teratur.", "Sisihkan jeda untuk hal yang menyenangkan."],
        "closing": "Jaga ritmemu, bukan kesempurnaan.",
    },
    "Sedang": {
        "level": 1, "icon": "⛅",
        "text": "Ada beban yang mungkin sedang terasa. Kamu boleh melambat dan membagi tugas menjadi langkah yang lebih kecil.",
        "tips": ["Pilih satu tugas kecil untuk dimulai.", "Ambil jeda dan cerita ke orang tepercaya."],
        "closing": "Tidak semua harus selesai hari ini.",
    },
    "Tinggi": {
        "level": 2, "icon": "🤝",
        "text": "Bebanmu mungkin terasa berat. Kamu tidak perlu menghadapinya sendirian; dukungan layak kamu dapatkan.",
        "tips": ["Jeda sejenak dari aktivitas yang membebani.", "Hubungi layanan konseling kampus."],
        "closing": "Mencari dukungan adalah langkah berani.",
    },
}


@st.cache_resource
def load_model():
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_meta():
    return json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}


def predict_label(model: dict, row: list) -> str:
    """Random Forest from JSON: walk every tree to a leaf, average the class probabilities."""
    total = [0.0] * len(model["classes"])
    for t in model["trees"]:
        i = 0
        while t["l"][i] != -1:
            i = t["l"][i] if row[t["f"][i]] <= t["t"][i] else t["r"][i]   # left when value <= threshold
        for k, p in enumerate(t["v"][i]):
            total[k] += p
    return model["classes"][max(range(len(total)), key=lambda k: total[k])]


# ------------------------------------------------------------------ page --
st.set_page_config(page_title="StressGuard AI", page_icon="🌸", layout="wide")

st.title("🌸 StressGuard AI")
st.caption("PROTOTIPE / 2026 · Deteksi stres awal berdasarkan dataset Student Early Stress Detection")
st.write("Cek sinyal stres dari keseharian. Bukan ujian, tidak ada jawaban baik atau buruk, "
         "dan ini bukan diagnosis medis.")

if not MODEL_PATH.exists():
    st.error("File model/stress_model.json tidak ditemukan. Letakkan di folder model/ di samping streamlit_app.py.")
    st.stop()

model = load_model()
meta = load_meta()

form_col, result_col = st.columns([3, 2], gap="large")

with form_col:
    st.subheader("Catatan keseharianmu")
    st.caption("Nilai terisi adalah contoh. Sesuaikan dengan yang kamu rasakan, bukan yang diharapkan orang lain.")
    with st.form("predict_form"):
        left, right = st.columns(2)
        values = {}
        for n, (name, caption, lo, hi, step, default) in enumerate(FIELDS):
            with (left if n % 2 == 0 else right):
                values[name] = st.slider(caption, min_value=lo, max_value=hi, value=default, step=step, key=name)
        submitted = st.form_submit_button("Prediksi", use_container_width=True, type="primary")
    st.caption("Hasil dihitung oleh model machine learning yang dilatih dari dataset. Bukan diagnosis medis.")

with result_col:
    st.subheader("Sebuah gambaran, bukan label.")
    if not submitted:
        st.info("Atur nilai di sebelah kiri lalu tekan **Prediksi** untuk melihat hasil dari model.")
    else:
        row = [float(values[name]) for name in model["features"]]
        label = predict_label(model, row)
        r = RESULTS[label]

        st.caption("HASIL PREDIKSI MODEL")
        st.markdown(f"### {r['icon']} Kategori stres: {label}")
        st.progress((r["level"] + 1) / 3, text=f"Tingkat {r['level'] + 1} dari 3")
        (st.success if label == "Rendah" else st.warning if label == "Sedang" else st.error)(r["text"])

        st.markdown("**Yang bisa kamu coba:**")
        for tip in r["tips"]:
            st.markdown(f"- {tip}")
        if label == "Tinggi":
            st.markdown(f"**Layanan konseling kampus:** {COUNSELING_CONTACT}")
            st.caption("Kontak belum diisi. Cek kanal resmi kampus untuk informasi layanan.")
        st.markdown(f"*{r['closing']}*")
        st.caption(DISCLAIMER)

st.divider()

# ------------------------------------------------------------ about model --
st.header("Kenali model di balik hasilnya.")
st.write("StressGuard AI memakai prediksi berbasis machine learning yang dilatih dari dataset di bawah.")
a, b, c = st.columns(3)
a.metric("Algoritma", meta.get("algorithm", "Belum diisi"))
ds = meta.get("dataset", {})
b.metric("Dataset", "Student Early Stress Detection")
b.caption(f"{ds.get('author', 'Rishav Raj')} · {ds.get('rows', '-')} rekaman latih · Kaggle")
c.metric("Validasi", meta.get("validation", "Belum diisi"))

m = meta.get("metrics")
if m:
    x, y = st.columns(2)
    x.metric("Akurasi (data uji)", f"{m['accuracy'] * 100:.1f}%")
    y.metric("Macro-F1 (data uji)", f"{m['macro_f1'] * 100:.1f}%")
    st.caption(f"Dihitung pada {meta.get('test_rows', '-')} baris data uji yang tidak dipakai saat pelatihan. "
               "Angka ini menggambarkan kecocokan dengan label dataset, bukan akurasi diagnosis.")
else:
    st.info("Metrik belum tersedia. Jalankan train_model.py untuk melatih model.")

st.divider()
st.caption(f"Tim: {TEAM} · Kampus: {CAMPUS} · Referensi: Student Early Stress Detection oleh Rishav Raj "
           "(https://www.kaggle.com/datasets/rishavraj1232/student-early-stress-detection) · "
           "Bukan alat diagnosis medis.")
