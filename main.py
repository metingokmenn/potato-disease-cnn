import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import Config
from data_loader import load_data
from models import build_model
from evaluation import plot_history, evaluate_set, plot_comparison 

def get_optimizer():
    """
    Config dosyasına göre optimizer nesnesi oluşturur ve döndürür.
    
    Bu fonksiyon, Config sınıfındaki OPTIMIZER ayarına göre uygun 
    optimizasyon algoritmasını seçer ve yapılandırır. Desteklenen 
    optimizasyon algoritmaları: Adam, SGD with Momentum, RMSprop.
    
    Parametreler:
        Hiçbir parametre almaz. Tüm ayarlar Config sınıfından alınır.
    
    Returns:
        tf.keras.optimizers.Optimizer: Yapılandırılmış optimizer nesnesi.
    
    Desteklenen Optimizer'lar:
        - 'adam': Adam optimizer (varsayılan)
        - 'sgd_momentum': SGD optimizer with momentum
        - 'rmsprop': RMSprop optimizer
    
    Örnek Kullanım:
        >>> optimizer = get_optimizer()
        Optimizasyon: ADAM (LR: 0.001)
    """
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
    """
    Ana eğitim fonksiyonu - Model eğitiminin tüm sürecini yönetir.
    
    Bu fonksiyon, patates hastalığı sınıflandırma modelinin eğitim sürecini 
    baştan sona yönetir. Veri yükleme, model oluşturma, eğitim, değerlendirme 
    ve sonuç kaydetme işlemlerini gerçekleştirir.
    
    İşlem Adımları:
        1. Ortam kontrolü (GPU/CPU kontrolü)
        2. Veri yükleme ve bölme (train/val/test)
        3. Model oluşturma ve derleme
        4. Callback'lerin yapılandırılması (Learning Rate Scheduler, Early Stopping)
        5. Model eğitimi
        6. Sonuç analizi ve görselleştirme
        7. Model kaydetme
    
    Parametreler:
        Hiçbir parametre almaz. Tüm ayarlar Config sınıfından alınır.
    
    Returns:
        None: Fonksiyon sonuçları konsola yazdırır ve dosyalara kaydeder.
    
    Callback'ler:
        - ReduceLROnPlateau: Validation loss 3 epoch boyunca iyileşmezse 
                             öğrenme hızını 0.2 ile çarpar (küçültür)
        - EarlyStopping: Validation loss 8 epoch boyunca iyileşmezse 
                         eğitimi durdurur ve en iyi ağırlıkları geri yükler
    
    Örnek Kullanım:
        >>> main()
        Model Tipi: mobilenet
        --- Veri Yükleniyor: dataset ---
        Eğitim Başlıyor...
        ...
    """
    print(f"Model Tipi: {Config.MODEL_TYPE}")
    
    if not tf.config.list_physical_devices('GPU'):
        print("UYARI: GPU bulunamadı. CPU kullanılıyor.")
    
    train_ds, val_ds, test_ds, class_names = load_data()
    
    model = build_model(len(class_names))
    
    optimizer = get_optimizer()
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    callbacks = []
    if not Config.DEBUG_MODE:
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
        
        callbacks = [lr_scheduler, early_stopping]

    print("\nEğitim Başlıyor...")
    epochs = 1 if Config.DEBUG_MODE else Config.EPOCHS
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print("\n--- SONUÇLAR ANALİZ EDİLİYOR ---")
    plot_history(history)
    
    if not Config.DEBUG_MODE:
        print("Eğitim seti metrikleri...")
        train_metrics = evaluate_set(model, train_ds, class_names, set_name="Training")
        
        print("Test seti metrikleri...")
        test_metrics = evaluate_set(model, test_ds, class_names, set_name="Test")
        
        plot_comparison(train_metrics, test_metrics)
    
    if not Config.DEBUG_MODE:
        save_name = f"{Config.MODEL_TYPE}_{Config.OPTIMIZER}.keras"
        model_path = os.path.join(Config.RESULTS_DIR, save_name)
        model.save(model_path)
        print(f"Model kaydedildi: {model_path}")

if __name__ == "__main__":
    main()