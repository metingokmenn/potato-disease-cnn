import os

class Config:
    # Dosya Yolları
    DATASET_DIR = "dataset"
    RESULTS_DIR = "results"
    
    # Görüntü Ayarları
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    
    # Eğitim Hiperparametreleri
    BATCH_SIZE = 32
    EPOCHS = 30           # Optimizasyonu görmek için biraz artırdık
    LEARNING_RATE = 0.001
    
    # --- OPTİMİZASYON AYARLARI (YENİ) ---
    # Seçenekler: 'adam', 'sgd_momentum', 'rmsprop'
    OPTIMIZER = 'adam'    
    MOMENTUM = 0.9        # Sadece SGD kullanılıyorsa etkilidir
    
    # Model Seçimi
    # 'custom_cnn' veya 'mobilenet'
    MODEL_TYPE = 'custom_cnn' 
    
    DEBUG_MODE = False    # Windows'ta FALSE yapın
    SEED = 42
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)