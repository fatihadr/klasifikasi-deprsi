import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "fatihadr/augmented-indobert-klasifikasi-depresi"
MAX_LEN = 512
MIN_TEXT_LENGTH = 5
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

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
    """Load tokenizer and model from Hugging Face."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        logger.info("Model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Initialize tokenizer and model globally
tokenizer, model = load_model()

# ============================================================
# LABEL MAPPING
# ============================================================

label_map = {
    0: "Tidak Depresi",
    1: "Depresi"
}

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_text(text):
    """
    Predict depression classification for input text.
    
    Args:
        text (str): Input text to classify
        
    Returns:
        tuple: (predicted_label_index, confidence_score, probability_list)
    """
    try:
        # Validate input
        if not text or not text.strip():
            return None, 0.0, [0.5, 0.5]
        
        if len(text.strip()) < MIN_TEXT_LENGTH:
            return None, 0.0, [0.5, 0.5]
        
        # Tokenize
        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate probabilities
        probs = F.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item() * 100
        
        return pred, confidence, probs.tolist()
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, 0.0, [0.5, 0.5]

# ============================================================
# UI COMPONENTS
# ============================================================

def render_prediction_result(pred, confidence, probs, threshold):
    """Render prediction results with visualization."""
    if pred is None:
        st.warning("⚠️ Tidak dapat melakukan prediksi. Teks terlalu pendek.")
        return
    
    dep_prob = probs[1]  # Depression probability
    non_prob = probs[0]  # Non-depression probability
    
    # Result display
    col1, col2 = st.columns(2)
    
    with col1:
        if dep_prob > threshold:
            st.error("🚨 Terindikasi Depresi")
        else:
            st.success("✅ Tidak Terindikasi Depresi")
    
    with col2:
        st.metric("Confidence Score", f"{confidence:.2f}%")
    
    # Probability details
    st.divider()
    st.subheader("📊 Analisis Probabilitas")
    
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric("Depresi", f"{dep_prob:.4f} ({dep_prob*100:.2f}%)")
    with prob_col2:
        st.metric("Tidak Depresi", f"{non_prob:.4f} ({non_prob*100:.2f}%)")
    
    # Visualization
    st.bar_chart({
        "Depresi": dep_prob,
        "Tidak Depresi": non_prob
    })

# ============================================================
# PAGE LAYOUT
# ============================================================

st.title("🧠 Klasifikasi Depresi")
st.caption("Model bekerja optimal jika diberikan beberapa teks (riwayat), bukan satu kalimat saja.")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    menu = st.selectbox(
        "Pilih Menu",
        ["Input Multi Teks", "Upload CSV", "Debug Model"]
    )
    
    st.divider()
    
    threshold = st.slider(
        "🎯 Threshold Depresi",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Nilai probabilitas minimum untuk klasifikasi Depresi"
    )
    
    st.divider()
    st.warning(
        "⚠️ **Disclaimer**: Model ini digunakan untuk penelitian, "
        "bukan untuk diagnosis medis profesional."
    )

# ============================================================
# INPUT MULTI TEKS
# ============================================================

if menu == "Input Multi Teks":
    st.subheader("📝 Input Multi Teks")
    st.info("💡 Masukkan beberapa teks (pisahkan dengan enter) untuk hasil yang lebih akurat")
    
    user_input = st.text_area(
        "Contoh:\naku capek hidup\nmerasa kosong\nkehilangan semangat",
        height=250,
        placeholder="Masukkan teks di sini..."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        predict_btn = st.button("🔍 Prediksi", use_container_width=True)
    
    with col2:
        clear_btn = st.button("🗑️ Hapus", use_container_width=True)
    
    if clear_btn:
        st.session_state.clear()
        st.rerun()
    
    if predict_btn:
        texts = [t.strip() for t in user_input.split("\n") if t.strip()]
        
        if len(texts) == 0:
            st.warning("⚠️ Masukkan minimal satu teks.")
        else:
            with st.spinner("⏳ Memproses..."):
                combined_text = " ".join(texts)
                pred, conf, probs = predict_text(combined_text)
                render_prediction_result(pred, conf, probs, threshold)
                
                # Show input summary
                with st.expander("📋 Ringkasan Input"):
                    st.write(f"**Jumlah baris**: {len(texts)}")
                    st.write(f"**Total karakter**: {len(combined_text)}")
                    st.text(combined_text)

# ============================================================
# UPLOAD CSV
# ============================================================

elif menu == "Upload CSV":
    st.subheader("📤 Upload CSV")
    st.info("💡 File CSV harus memiliki kolom bernama 'text'")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        help="Maksimal ukuran file: 10MB"
    )
    
    if uploaded_file is not None:
        # Validate file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ File terlalu besar. Maksimal ukuran: {MAX_FILE_SIZE / (1024*1024):.0f}MB")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                if "text" not in df.columns:
                    st.error("❌ CSV harus memiliki kolom 'text'")
                    st.write("Kolom yang tersedia:", df.columns.tolist())
                else:
                    # Show preview
                    st.subheader("📊 Preview Data")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Validate data
                    null_count = df["text"].isnull().sum()
                    if null_count > 0:
                        st.warning(f"⚠️ Ditemukan {null_count} baris kosong")
                    
                    # Process predictions
                    if st.button("🚀 Jalankan Prediksi"):
                        with st.spinner("⏳ Memproses..."):
                            progress_bar = st.progress(0)
                            hasil_list = []
                            conf_list = []
                            
                            total = len(df)
                            for idx, txt in enumerate(df["text"]):
                                # Handle null values
                                if pd.isna(txt):
                                    hasil_list.append("N/A")
                                    conf_list.append(0.0)
                                else:
                                    pred, conf, probs = predict_text(str(txt))
                                    
                                    if pred is None:
                                        hasil_list.append("Error")
                                        conf_list.append(0.0)
                                    else:
                                        hasil = "Depresi" if probs[1] > threshold else "Tidak Depresi"
                                        hasil_list.append(hasil)
                                        conf_list.append(conf)
                                
                                progress_bar.progress((idx + 1) / total)
                            
                            # Add results to dataframe
                            df["prediksi"] = hasil_list
                            df["confidence"] = conf_list
                            
                            st.success("✅ Prediksi selesai!")
                            
                            # Show results
                            st.subheader("📈 Hasil Prediksi")
                            st.dataframe(df, use_container_width=True)
                            
                            # Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                depresi_count = (df["prediksi"] == "Depresi").sum()
                                st.metric("Depresi", depresi_count)
                            with col2:
                                tidak_count = (df["prediksi"] == "Tidak Depresi").sum()
                                st.metric("Tidak Depresi", tidak_count)
                            with col3:
                                error_count = (df["prediksi"] == "Error").sum()
                                st.metric("Error", error_count)
                            
                            # Download results
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Hasil CSV",
                                data=csv,
                                file_name="hasil_prediksi.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
                logger.error(f"File processing error: {str(e)}")

# ============================================================
# DEBUG MODE
# ============================================================

elif menu == "Debug Model":
    st.subheader("🐛 Debug Model")
    
    with st.expander("ℹ️ Informasi Model"):
        st.write(f"**Model Name**: {MODEL_NAME}")
        st.write(f"**Max Length**: {MAX_LEN}")
        st.write(f"**Labels**: {label_map}")
    
    debug_text = st.text_area("Masukkan teks untuk debug", height=200)
    
    if st.button("🔍 Debug"):
        if debug_text.strip():
            with st.spinner("⏳ Memproses..."):
                pred, conf, probs = predict_text(debug_text)
                
                if pred is not None:
                    st.subheader("📊 Debug Output")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Index", pred)
                    with col2:
                        st.metric("Label", label_map[pred])
                    with col3:
                        st.metric("Confidence", f"{conf:.2f}%")
                    
                    st.subheader("🎯 Probabilities")
                    st.write(f"Tidak Depresi (0): {probs[0]:.6f}")
                    st.write(f"Depresi (1): {probs[1]:.6f}")
                    
                    st.subheader("📈 Visualization")
                    st.bar_chart({
                        "Tidak Depresi": probs[0],
                        "Depresi": probs[1]
                    })
                else:
                    st.warning("⚠️ Tidak dapat melakukan prediksi.")
        else:
            st.warning("⚠️ Masukkan teks untuk di-debug.")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "🔬 **Penelitian**: Model ini digunakan untuk penelitian akademik. "
    "Bukan untuk diagnosis medis profesional. "
    "Konsultasikan dengan ahli kesehatan mental jika diperlukan."
)
