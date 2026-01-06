import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- AYARLAR ---
MODEL_PATH = 'results/custom_cnn_results/custom_cnn_adam.keras'
PREDICT_FOLDER = 'predict-images'  # Resimlerin olduğu klasör
CLASS_NAMES = ['Early_Blight', 'Healthy', 'Late_Blight']

def load_trained_model():
    """Modeli diskten bir kere yükler."""
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
        return None
    print(f"Model yükleniyor: {MODEL_PATH}...")
    return tf.keras.models.load_model(MODEL_PATH)

def predict_single_image(model, image_path, file_name):
    """Yüklenmiş model ile tek bir resim için tahmin yapar."""
    
    if not os.path.exists(image_path):
        print(f"HATA: Resim bulunamadı: {image_path}")
        return

    # Resmi oku
    img = cv2.imread(image_path)
    if img is None:
        print(f"UYARI: '{file_name}' okunamadı. Resim formatını kontrol edin.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Modelin beklediği boyuta getir (224x224)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    
    
    # Batch boyutuna getir: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = tf.expand_dims(img_resized, 0)

    # Tahmin Yap
    predictions = model.predict(img_array, verbose=0) # verbose=0 sessiz mod
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Sonucu Konsola Yaz
    print(f"Dosya: {file_name:<10} | Tahmin: {predicted_class:<15} | Güven: %{confidence:.2f}")

    # Görselleştirme (İsteğe bağlı - Çok resim varsa kapatılabilir)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.title(f"Dosya: {file_name}\nTahmin: {predicted_class} (%{confidence:.2f})")
    plt.axis("off")
    plt.show()

def main():
    # 1. Klasör Kontrolü
    if not os.path.exists(PREDICT_FOLDER):
        print(f"HATA: '{PREDICT_FOLDER}' klasörü bulunamadı. Lütfen oluşturun.")
        return

    # 2. Klasördeki Dosyaları Listele
    files = os.listdir(PREDICT_FOLDER)
    
    # Sadece resim dosyalarını filtrele (jpg, png, jpeg)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sıralama yapalım ki eb-1, eb-2 diye düzgün gitsin
    image_files.sort()

    if not image_files:
        print(f"UYARI: '{PREDICT_FOLDER}' klasöründe hiç resim bulunamadı.")
        return

    # 3. Modeli Yükle (Sadece 1 Kere!)
    model = load_trained_model()
    if model is None:
        return

    print(f"\n--- {PREDICT_FOLDER} klasöründeki {len(image_files)} resim test ediliyor ---\n")

    # 4. Döngü ile Hepsini Tahmin Et
    for file_name in image_files:
        full_path = os.path.join(PREDICT_FOLDER, file_name)
        predict_single_image(model, full_path, file_name)

if __name__ == "__main__":
    main()