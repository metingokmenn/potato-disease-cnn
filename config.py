import os

class Config:
    # Dosya Yolları
    DATASET_DIR = "dataset"
    RESULTS_DIR = "results"
    
    # Görüntü Ayarları
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    
    # Eğitim Hiperparametreleri
    BATCH_SIZE = 32
    EPOCHS = 20           # Windows'ta artıracağız (örn: 50)
    LEARNING_RATE = 0.001
    
    # Model Seçimi
    # 'custom_cnn' -> Kendi yazdığımız model
    # 'mobilenet'  -> Transfer learning
    MODEL_TYPE = 'custom_cnn' 
    
    # Donanım/Test
    DEBUG_MODE = True     # Mac'te test ederken TRUE, Windows'ta FALSE olacak
    SEED = 42             # Bilimsel tekrarlanabilirlik için sabit tohum
    
    # Oluştur
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)