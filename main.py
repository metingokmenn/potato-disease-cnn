import tensorflow as tf
import os
from config import Config
from data_loader import load_data
from models import build_model
from evaluation import plot_history, evaluate_set, plot_comparison 

def get_optimizer():
    """Config dosyasına göre optimizer nesnesi döndürür."""
    if Config.OPTIMIZER == 'adam':
        print(f"Optimizasyon: ADAM (LR: {Config.LEARNING_RATE})")
        return tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    
    elif Config.OPTIMIZER == 'sgd_momentum':
        print(f"Optimizasyon: SGD with Momentum (LR: {Config.LEARNING_RATE}, Momentum: {Config.MOMENTUM})")
        return tf.keras.optimizers.SGD(learning_rate=Config.LEARNING_RATE, momentum=Config.MOMENTUM)
    
    elif Config.OPTIMIZER == 'rmsprop':
        print(f"Optimizasyon: RMSprop (LR: {Config.LEARNING_RATE})")
        return tf.keras.optimizers.RMSprop(learning_rate=Config.LEARNING_RATE)
    
    else:
        print("Bilinmeyen optimizer, varsayılan olarak Adam kullanılıyor.")
        return tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)

def main():
    # 1. Ortam Kontrolü
    print(f"Model Tipi: {Config.MODEL_TYPE}")
    
    if not tf.config.list_physical_devices('GPU'):
        print("UYARI: GPU bulunamadı. CPU kullanılıyor.")
    
    # 2. Veri Yükleme
    train_ds, val_ds, test_ds, class_names = load_data()
    
    # 3. Model Kurulumu
    model = build_model(len(class_names))
    
    # --- YENİ: Optimizer Seçimi ---
    optimizer = get_optimizer()
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # --- YENİ: Gelişmiş Callbacks (Optimizasyonun Kalbi) ---
    callbacks = []
    if not Config.DEBUG_MODE:
        # A. Learning Rate Scheduler:
        # Eğer 'val_loss' 3 epoch boyunca iyileşmezse, öğrenme hızını 0.2 ile çarp (küçült).
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1 # Konsola "Learning rate düşürüldü" yazar
        )
        
        # B. Early Stopping:
        # Eğer 'val_loss' 10 epoch boyunca hiç iyileşmezse eğitimi durdur.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True, # En iyi sonucu geri yükle
            verbose=1
        )
        
        callbacks = [lr_scheduler, early_stopping]

    # 4. Eğitim
    print("\nEğitim Başlıyor...")
    epochs = 1 if Config.DEBUG_MODE else Config.EPOCHS
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks # Callbacks buraya eklendi
    )
    
    # ... (Geri kalan kodlar aynı: Sonuç Analizi ve Kayıt) ...
    print("\n--- SONUÇLAR ANALİZ EDİLİYOR ---")
    plot_history(history)
    
    if not Config.DEBUG_MODE:
        print("Eğitim seti metrikleri...")
        train_metrics = evaluate_set(model, train_ds, class_names, set_name="Training")
        
        print("Test seti metrikleri...")
        test_metrics = evaluate_set(model, test_ds, class_names, set_name="Test")
        
        plot_comparison(train_metrics, test_metrics)
    
    if not Config.DEBUG_MODE:
        # Dosya adına optimizer ismini de ekleyelim ki karışmasın
        save_name = f"{Config.MODEL_TYPE}_{Config.OPTIMIZER}.keras"
        model_path = os.path.join(Config.RESULTS_DIR, save_name)
        model.save(model_path)
        print(f"Model kaydedildi: {model_path}")

if __name__ == "__main__":
    main()