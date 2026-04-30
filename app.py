import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================

# CONFIG

# ============================================================

MODEL_NAME = "fatihadr/augmented-indobert-klasifikasi-depresi"
MAX_LEN = 512

st.set_page_config(
page_title="Deteksi Depresi",
page_icon="🧠",
layout="wide"
)

# ============================================================

# LOAD MODEL

# ============================================================

@st.cache_resource
def load_model():
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
return tokenizer, model

tokenizer, model = load_model()

# ============================================================

# LABEL

# ============================================================

label_map = {
0: "Tidak Depresi",
1: "Depresi"
}

# ============================================================

# PREDICT FUNCTION

# ============================================================

def predict_text(text):
inputs = tokenizer(
str(text),
return_tensors="pt",
truncation=True,
padding=True,
max_length=MAX_LEN
)

with torch.no_grad():
    outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=1)[0]

pred = torch.argmax(probs).item()
confidence = probs[pred].item() * 100

return pred, confidence, probs.tolist()

# ============================================================

# UI

# ============================================================

st.title("🧠 Klasifikasi Depresi")
st.caption("Model bekerja optimal jika diberikan beberapa teks (riwayat), bukan satu kalimat saja.")

menu = st.sidebar.selectbox(
"Pilih Menu",
["Input Multi Teks", "Upload CSV", "Debug Model"]
)

threshold = st.sidebar.slider(
"Threshold Depresi",
0.1, 0.9, 0.45, 0.05
)

# ============================================================

# INPUT MULTI TEKS

# ============================================================

if menu == "Input Multi Teks":

st.subheader("Masukkan beberapa teks (pisahkan dengan enter)")

user_input = st.text_area(
    "Contoh:\naku capek hidup\nmerasa kosong\nkehilangan semangat",
    height=250
)

if st.button("Prediksi"):

    texts = [t.strip() for t in user_input.split("\n") if t.strip() != ""]

    if len(texts) == 0:
        st.warning("Masukkan minimal satu teks.")
    else:
        combined_text = " ".join(texts)

        pred, conf, probs = predict_text(combined_text)

        dep_prob = probs[1]
        non_prob = probs[0]

        if dep_prob > threshold:
            st.error("Terindikasi Depresi")
        else:
            st.success("Tidak Terindikasi Depresi")

        st.write(f"Confidence: {conf:.2f}%")

        st.write("### Probabilitas")
        st.write(f"Depresi: {dep_prob:.4f}")
        st.write(f"Tidak Depresi: {non_prob:.4f}")

        st.bar_chart({
            "Depresi": dep_prob,
            "Tidak Depresi": non_prob
        })

# ============================================================

# UPLOAD CSV

# ============================================================

if menu == "Upload CSV":

st.subheader("Upload CSV")

file = st.file_uploader("Upload file CSV (kolom: text)", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    if "text" not in df.columns:
        st.error("CSV harus memiliki kolom 'text'")
    else:
        hasil_list = []
        conf_list = []

        for txt in df["text"]:
            pred, conf, probs = predict_text(txt)

            if probs[1] > threshold:
                hasil = "Depresi"
            else:
                hasil = "Tidak Depresi"

            hasil_list.append(hasil)
            conf_list.append(conf)

        df["prediksi"] = hasil_list
        df["confidence"] = conf_list

        st.success("Prediksi selesai")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Hasil",
            csv,
            "hasil_prediksi.csv",
            "text/csv"
        )

# ============================================================

# DEBUG MODE

# ============================================================

if menu == "Debug Model":

st.subheader("Debug Model")

text = st.text_area("Masukkan teks untuk debug")

if st.button("Debug"):

    pred, conf, probs = predict_text(text)

    st.write("Probabilities:", probs)
    st.write("Predicted Index:", pred)
    st.write("Label:", label_map[pred])
# ============================================================

# FOOTER

# ============================================================

st.caption("Digunakan untuk penelitian, bukan diagnosis medis.")
