"""StressGuard AI - Streamlit app.  Run:  streamlit run app.py"""
import json
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent
TEAM, CAMPUS = "Kelompok 2", "Universitas Bunda Mulia"
DATASET_URL = "https://www.kaggle.com/datasets/rishavraj1232/student-early-stress-detection"
COUNSELING_CONTACT = ""  # phone / e-mail / URL shown on the "Tinggi" result
DISCLAIMER = "Ini bukan diagnosis medis. Hasil tidak menggantikan penilaian psikolog atau tenaga kesehatan."

# (feature name, label, caption, min, max, step, default, format, group)
FIELDS = [
    ("sleep_hours_per_day", "Jam tidur", "Rata-rata tidur per malam", 4.0, 10.0, 0.5, 6.5, "%g jam", "Kebiasaan harian"),
    ("study_hours_per_week", "Jam belajar", "Total jam belajar per minggu", 2.0, 35.0, 1.0, 24.0, "%g jam", "Kebiasaan harian"),
    ("academic_pressure_score", "Tekanan akademik", "0 = ringan · 10 = sangat berat", 0, 10, 1, 7, "%d / 10", "Beban akademik"),
    ("attendance_rate", "Kehadiran kuliah (%)", "Seberapa sering kamu hadir", 0, 100, 1, 85, "%d", "Beban akademik"),
    ("missed_submissions_count", "Tugas terlewat", "Jumlah tugas yang terlewat", 0, 20, 1, 2, "%d", "Beban akademik"),
    ("anxiety_score", "Rasa cemas", "0 = ringan · 10 = sangat berat", 0, 10, 1, 4, "%d / 10", "Kondisi emosional"),
    ("depression_score", "Perasaan murung", "0 = ringan · 10 = sangat berat", 0, 10, 1, 3, "%d / 10", "Kondisi emosional"),
    ("burnout_score", "Kelelahan (burnout)", "0 = ringan · 10 = sangat berat", 0, 10, 1, 5, "%d / 10", "Kondisi emosional"),
]

RESULTS = {
    "Rendah": ("🌱", "success", "Ritme harianmu tampak cukup terjaga. Tetap beri ruang untuk istirahat dan hal yang kamu sukai.",
               ["Pertahankan waktu tidur yang teratur.", "Sisihkan jeda untuk hal yang menyenangkan."],
               "Jaga ritmemu, bukan kesempurnaan."),
    "Sedang": ("⛅", "warning", "Ada beban yang mungkin sedang terasa. Kamu boleh melambat dan membagi tugas menjadi langkah yang lebih kecil.",
               ["Pilih satu tugas kecil untuk dimulai.", "Ambil jeda dan cerita ke orang tepercaya."],
               "Tidak semua harus selesai hari ini."),
    "Tinggi": ("🫶", "error", "Bebanmu mungkin terasa berat. Kamu tidak perlu menghadapinya sendirian; dukungan layak kamu dapatkan.",
               ["Jeda sejenak dari aktivitas yang membebani.", f"Hubungi layanan konseling {CAMPUS}."],
               "Mencari dukungan adalah langkah berani."),
}


@st.cache_resource
def load_model():
    return json.loads((HERE / "stress_model.json").read_text(encoding="utf-8"))


@st.cache_data
def load_meta():
    return json.loads((HERE / "model_meta.json").read_text(encoding="utf-8"))


def predict(values: dict) -> str:
    """Random Forest exported to JSON: sum the leaf class probabilities of every tree."""
    model = load_model()
    x = [float(values[n]) for n in model["features"]]
    sums = [0.0] * len(model["classes"])
    for t in model["trees"]:
        i = 0
        while t["l"][i] != -1:
            i = t["l"][i] if x[t["f"][i]] <= t["t"][i] else t["r"][i]
        for k, p in enumerate(t["v"][i]):
            sums[k] += p
    return model["classes"][max(range(len(sums)), key=lambda k: (sums[k], -k))]


def show_result(label: str):
    icon, kind, text, tips, closing = RESULTS[label]
    box = getattr(st, kind)
    box(f"### {icon} Kategori stres: {label}\n\n{text}")
    st.markdown("**Yang bisa kamu coba:**")
    for t in tips:
        st.markdown(f"- {t}")
    if label == "Tinggi":
        st.info(COUNSELING_CONTACT or f"Cek kanal resmi {CAMPUS} untuk informasi layanan konseling.")
    st.markdown(f"*{closing}*")
    st.caption(DISCLAIMER)


def reset():
    for f in FIELDS:
        st.session_state[f[0]] = f[6]
    st.session_state.pop("label", None)


def page_check():
    st.header("Bagaimana kabarmu belakangan ini?")
    st.write("Sesuaikan slider dengan yang kamu rasakan, bukan yang diharapkan orang lain. "
             "Nilai awal hanyalah contoh. " + DISCLAIMER)
    form, out = st.columns([3, 2], gap="large")
    with form:
        for group in dict.fromkeys(f[8] for f in FIELDS):
            st.subheader(group)
            cols = st.columns(2)
            for i, f in enumerate([f for f in FIELDS if f[8] == group]):
                with cols[i % 2]:
                    st.slider(f[1], f[3], f[4], f[6], f[5], format=f[7], key=f[0])
                    st.caption(f[2])
        b1, b2 = st.columns([2, 1])
        if b1.button("Prediksi", type="primary", use_container_width=True):
            vals = {f[0]: st.session_state[f[0]] for f in FIELDS}
            vals["attendance_rate"] /= 100  # slider is in percent, model expects 0-1
            st.session_state["label"] = predict(vals)
        b2.button("Setel ulang", on_click=reset, use_container_width=True)
    with out:
        if "label" in st.session_state:
            show_result(st.session_state["label"])
        else:
            st.info("Tekan **Prediksi** untuk melihat hasil dari model.")


def page_home():
    st.title("🌸 Cek sinyal stres dari keseharian.")
    st.write("Isi catatan singkat tentang tidur, belajar, dan perasaanmu belakangan ini. Model machine learning kami "
             "memberi gambaran tingkat stres, lengkap dengan langkah kecil yang bisa dicoba. Ini bukan diagnosis medis.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Indikator keseharian", 8)
    c2.metric("Kategori hasil", 3)
    c3.metric("Akun yang perlu dibuat", 0)
    st.subheader("Cara kerja")
    a, b, c = st.columns(3)
    a.markdown("**1. Isi catatan**\n\nAtur delapan slider: tidur, belajar, beban akademik, dan kondisi emosional.")
    b.markdown("**2. Model membaca pola**\n\nRandom Forest yang dilatih dari dataset Student Early Stress Detection.")
    c.markdown("**3. Lihat gambaran**\n\nKategori rendah, sedang, atau tinggi beserta langkah kecil yang bisa dicoba.")
    st.caption(f"Dibuat oleh {TEAM} · {CAMPUS} · Dataset: [Student Early Stress Detection]({DATASET_URL}) oleh Rishav Raj (Kaggle). {DISCLAIMER}")


def page_about():
    meta = load_meta()
    ds, m = meta.get("dataset", {}), meta.get("metrics", {})
    st.header("Kenali model di balik hasilnya.")
    st.markdown(f"- **Algoritma:** {meta.get('algorithm', '-')}\n"
                f"- **Dataset:** {ds.get('name', '-')} ({ds.get('author', '-')}, Kaggle), {ds.get('rows', 0):,} baris\n"
                f"- **Validasi:** {meta.get('validation', '-')}\n"
                f"- **Data uji:** {meta.get('test_rows', 0):,} baris\n"
                f"- **Keluaran:** Rendah · Sedang · Tinggi")
    if m:
        c1, c2 = st.columns(2)
        c1.metric("Akurasi", f"{m['accuracy'] * 100:.1f}%")
        c2.metric("Macro-F1", f"{m['macro_f1'] * 100:.1f}%")
        if m["accuracy"] >= 0.98:
            st.info("Skor setinggi ini biasanya berarti label dataset sangat mudah ditebak dari indikator yang dipakai. "
                    "Jangan dibaca sebagai jaminan ketepatan pada kondisi nyata.")
    st.subheader("Indikator")
    ranges = meta.get("feature_ranges", {})
    st.table([{"Indikator": f[1].replace(" (%)", ""), "Kolom dataset": f[0],
               "Rentang data latih": " – ".join(str(v) for v in ranges.get(f[0], []))} for f in FIELDS])
    st.subheader("Keterbatasan")
    st.markdown("- **Bukan diagnosis.** Gunakan sebagai bahan refleksi.\n"
                "- **Hanya delapan indikator.** Keluarga, kesehatan, atau keuangan tidak tertangkap.\n"
                "- **Bergantung pada dataset.** Belum tentu cocok untuk semua mahasiswa.")


st.set_page_config(page_title="StressGuard AI", page_icon="🌸", layout="wide")
tab_home, tab_check, tab_about = st.tabs(["Beranda", "Cek stres", "Tentang model"])
with tab_home:
    page_home()
with tab_check:
    page_check()
with tab_about:
    page_about()
