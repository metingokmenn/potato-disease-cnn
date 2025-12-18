import tensorflow as tf
import os
from config import Config
from data_loader import load_data
from models import build_model
from evaluation import plot_history, evaluate_model

def main():
    # 1. Ortam Kontrolü
    print(f"Model Tipi: {Config.MODEL_TYPE}")
    print(f"Debug Modu: {Config.DEBUG_MODE}")
    
    # Mac M serisi işlemciler için optimizasyon uyarısı
    if not tf.config.list_physical_devices('GPU'):
        print("UYARI: GPU bulunamadı. CPU kullanılıyor.")
    
    # 2. Veri Yükleme
    train_ds, val_ds, class_names = load_data()
    
    # 3. Model Oluşturma
    model = build_model(len(class_names))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # 4. Eğitim
    print("\nEğitim Başlıyor...")
    # Debug modundaysak epoch sayısını zorla 1 yap
    epochs = 1 if Config.DEBUG_MODE else Config.EPOCHS
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    # 5. Sonuçları Kaydet ve Raporla
    print("\nSonuçlar Analiz Ediliyor...")
    
    # Loss/Accuracy grafikleri
    plot_history(history)
    
    # Confusion Matrix ve F1-Score tabloları
    # Not: Debug modunda validation seti çok küçük olduğu için bu kısım hata verebilir veya anlamsız olabilir.
    # Bu yüzden sadece gerçek eğitimde veya yeterli veriyle çalışırken anlamlıdır.
    if not Config.DEBUG_MODE:
        evaluate_model(model, val_ds, class_names)
    else:
        print("Debug modunda detaylı metrik analizi (Confusion Matrix) atlanıyor.")
        
    # Modeli Kaydet
    if not Config.DEBUG_MODE:
        model_path = os.path.join(Config.RESULTS_DIR, f"{Config.MODEL_TYPE}_model.keras")
        model.save(model_path)
        print(f"Model kaydedildi: {model_path}")

if __name__ == "__main__":
    main()