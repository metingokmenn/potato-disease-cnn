import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

MODEL_PATH = 'results/mobilenet_adam.keras'  
CLASS_NAMES = ['Early_Blight', 'Healthy', 'Late_Blight']

def predict_image(image_path):
    """
    Patates yaprağı görüntüsü üzerinde hastalık tahmini yapar.
    
    Bu fonksiyon, eğitilmiş bir derin öğrenme modelini kullanarak patates yaprağı 
    görüntülerini analiz eder ve hastalık sınıflandırması yapar. Görüntüyü 
    ön işleme tabi tutar, model ile tahmin yapar ve sonuçları görselleştirir.
    
    Parametreler:
        image_path (str): Tahmin yapılacak görüntü dosyasının yolu.
                         Örnek: "test_image.jpg" veya "dataset/Late_Blight/image.jpg"
    
    Returns:
        None: Fonksiyon sonuçları konsola yazdırır ve görselleştirme gösterir.
              Hata durumunda erken çıkış yapar.
    
    Örnek Kullanım:
        >>> predict_image("test_image.jpg")
        Model yükleniyor: models/mobilenet_model.keras...
        Tahmin yapılıyor...
        SONUÇ: Late_Blight
        GÜVEN: %95.23
    """
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
        return

    print(f"Model yükleniyor: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    if not os.path.exists(image_path):
        print(f"HATA: Resim bulunamadı: {image_path}")
        return
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    img_array = tf.expand_dims(img_resized, 0)

    print("Tahmin yapılıyor...")
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"------------------------------------------------")
    print(f"SONUÇ: {predicted_class}")
    print(f"GÜVEN: %{confidence:.2f}")
    print(f"------------------------------------------------")

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(f"Tahmin: {predicted_class} (%{confidence:.2f})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    target_image = "test_image.jpg" 
    if os.path.exists(target_image):
        predict_image(target_image)
    else:
        print(f"Lütfen proje klasörüne '{target_image}' adında bir patates yaprağı resmi koyun.")