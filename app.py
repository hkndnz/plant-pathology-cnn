import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# -------------------------------------------------------------
# 1. Sabit Tanımlar ve Model Yükleme
# -------------------------------------------------------------

# Plant Pathology 2020 Sınıf İsimleri
CLASS_NAMES = ['healthy', 'multiple_diseases', 'rust', 'scab']
IMAGE_SIZE = 224 # Modelin eğitildiği resim boyutuyla eşleşmeli

# Modeli yükleme fonksiyonu (Uygulamanın başlangıcında sadece bir kez çalışır)
@st.cache_resource
def load_classification_model():
    """Kaydedilmiş Keras (Görüntü Sınıflandırma) modelini yükler."""
    try:
        # LÜTFEN KENDİ KAYITLI MODEL DOSYA ADINIZI BURAYA GİRİN
        # Örneğin: 'plant_pathology_model.h5' veya 'model.keras'
        MODEL_PATH = 'plant_pathology_model.keras' 
        
        # Keras modelini yükleme
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Hata: Model dosyası yüklenemedi. Lütfen '{MODEL_PATH}' dosyasının varlığını kontrol edin. Hata: {e}")
        return None

# Modeli yükle
model = load_classification_model()

# -------------------------------------------------------------
# 2. Ön İşleme ve Tahmin Fonksiyonları
# -------------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Yüklenen resmi model tahmini için hazırlar."""
    # 1. Yeniden Boyutlandırma
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # 2. Numpy dizisine dönüştürme
    image_array = np.array(image, dtype=np.float32)
    # 3. Normalizasyon (0-255 -> 0-1 aralığına)
    image_array = image_array / 255.0
    # 4. Batch boyutu ekleme (Modelin istediği format: [1, IMAGE_SIZE, IMAGE_SIZE, 3])
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def make_prediction(model: tf.keras.Model, processed_image: np.ndarray):
    """Hazırlanmış resim üzerinde tahmin yapar."""
    predictions = model.predict(processed_image)
    # Tahmin, olasılık dizisi olarak döner (örn: [0.01, 0.05, 0.90, 0.04])
    return predictions[0]

# -------------------------------------------------------------
# 3. Streamlit Arayüzü
# -------------------------------------------------------------

st.set_page_config(
    page_title="Elma Yaprağı Hastalık Sınıflandırması (Plant Pathology 2020)",
    page_icon="🌿",
    layout="centered"
)

st.title("🍎 Elma Yaprağı Hastalık Teşhisi")
st.markdown("Convolutional Neural Network (CNN) modeli kullanılarak yüklenen elma yaprağının sağlık durumu (veya hastalığı) tahmin edilir.")
st.markdown("---")

if model:
    # Kullanıcıdan resim yüklemesini iste
    uploaded_file = st.file_uploader(
        "Lütfen bir elma yaprağı görseli yükleyin (.jpg veya .png):", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Resim yüklendi
        
        # Resmi PIL formatında aç
        image = Image.open(uploaded_file)
        
        # Yüklenen resmi göster
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Yüklenen Görsel", use_column_width=True)

        with col2:
            st.info("Tahmin Başlatılıyor...")
            
            # Ön İşleme
            processed_image = preprocess_image(image)
            
            # Tahmin
            with st.spinner('Model tahmin yapıyor...'):
                predictions = make_prediction(model, processed_image)
            
            # Sonuçları alma
            predicted_class_index = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_class_index]
            confidence = predictions[predicted_class_index] * 100
            
            # Sonucu formatlama
            if predicted_class == 'healthy':
                st.success(f"Teşhis: 🍏 SAĞLIKLI Yaprak")
            else:
                st.error(f"Teşhis: 🚨 {predicted_class.upper()} Hastalığı")
            
            st.subheader(f"Güvenilirlik: %{confidence:.2f}")

            st.markdown("---")
            st.subheader("Tüm Olasılık Dağılımı")
            
            # Olasılıkları bar grafiği olarak gösterme
            
            # Pandas DataFrame oluşturma
            df_results = pd.DataFrame({
                'Hastalık Sınıfı': [c.capitalize() for c in CLASS_NAMES],
                'Olasılık (%)': predictions * 100
            })
            
            # Bar grafiği
            st.bar_chart(
                df_results.set_index('Hastalık Sınıfı'),
                height=300
            )

else:
    st.warning("Model yüklenemedi. Lütfen notebook'unuzda modeli eğitip doğru adla kaydettiğinizden emin olun (plant_pathology_model.keras).")

# -------------------------------------------------------------
# Hugging Face Talimatları
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ## Hugging Face Spaces İçin
    Bu uygulamayı Hugging Face'e dağıtmak için aşağıdaki dosyaları deponuza yükleyin:
    1. **`app.py`** (Bu dosya)
    2. **`plant_pathology_model.keras`** (Eğittiğiniz model dosyası, adına dikkat edin)
    3. **`requirements.txt`** (Gerekli kütüphaneleri içerir)
    """
)

st.sidebar.code(
    """
# requirements.txt içeriği:
tensorflow
streamlit
Pillow
numpy
"""
)
