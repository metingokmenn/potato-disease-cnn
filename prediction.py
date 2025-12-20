import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- AYARLAR ---
# Hangi modeli kullanacaksak onun yolunu buraya yazıyoruz
# Örn: 'models/custom_cnn_model.keras' veya 'models/mobilenet_model.keras'
MODEL_PATH = 'models/mobilenet_model.keras'  
CLASS_NAMES = ['Early_Blight', 'Healthy', 'Late_Blight']

def predict_image(image_path):
    # 1. Modeli Yükle
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
        return

    print(f"Model yükleniyor: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Resmi Yükle ve İşle (Preprocessing)
    if not os.path.exists(image_path):
        print(f"HATA: Resim bulunamadı: {image_path}")
        return
    
    # Resmi oku
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Modelin beklediği boyuta getir (224x224)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Batch boyutuna getir: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = tf.expand_dims(img_resized, 0)

    # 3. Tahmin Yap
    print("Tahmin yapılıyor...")
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    # 4. Sonucu Göster
    print(f"------------------------------------------------")
    print(f"SONUÇ: {predicted_class}")
    print(f"GÜVEN: %{confidence:.2f}")
    print(f"------------------------------------------------")

    # Görselleştirme
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(f"Tahmin: {predicted_class} (%{confidence:.2f})")
    plt.axis("off")
    plt.show()

# --- TEST KISMI ---
if __name__ == "__main__":
    # Buraya test etmek istediğiniz resmin yolunu yazın.
    # Örnek 1: Veri setinden rastgele bir resim yolu
    # image_path = "dataset/Late_Blight/bir_resim_ismi.jpg"
    
    # Örnek 2: Google'dan indirdiğiniz bir resim (deneme.jpg diye kaydedin)
    target_image = "test_image.jpg" 
    
    # Eğer test_image.jpg yoksa uyarı ver
    if os.path.exists(target_image):
        predict_image(target_image)
    else:
        print(f"Lütfen proje klasörüne '{target_image}' adında bir patates yaprağı resmi koyun.")