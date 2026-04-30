import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "fatihadr/augmented-indobert-klasifikasi-depresi"

st.set_page_config(page_title="Deteksi Depresi", page_icon="🧠")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

label_map = {
    0: "Depresi",
    1: "Tidak Depresi"
}

def predict_text(text):
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0][pred].item() * 100

    return label_map[pred], conf

st.title("🧠 Klasifikasi Depresi")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Input Teks", "Upload CSV"]
)

# ==================================================
# INPUT MANUAL
# ==================================================
if menu == "Input Teks":

    text = st.text_area("Masukkan teks")

    if st.button("Prediksi"):
        hasil, conf = predict_text(text)

        st.success(f"Hasil: {hasil}")
        st.info(f"Confidence: {conf:.2f}%")

# ==================================================
# CSV
# ==================================================
if menu == "Upload CSV":

    st.write("Upload file CSV yang memiliki kolom bernama: text")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        st.write("Preview Data:")
        st.dataframe(df.head())

        if "text" not in df.columns:
            st.error("CSV harus memiliki kolom text")
        else:
            hasil_list = []
            conf_list = []

            for txt in df["text"]:
                hasil, conf = predict_text(txt)
                hasil_list.append(hasil)
                conf_list.append(conf)

            df["prediksi"] = hasil_list
            df["confidence"] = conf_list

            st.success("Prediksi selesai")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Hasil CSV",
                csv,
                "hasil_prediksi.csv",
                "text/csv"
            )

st.caption("Untuk penelitian, bukan diagnosis medis")
