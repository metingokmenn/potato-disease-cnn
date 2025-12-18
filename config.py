import os

class Config:
    # --- DOSYA YOLLARI ---
    # Windows'ta klasör yapınızın "dataset" şeklinde olduğundan emin olun
    DATASET_DIR = "dataset"
    RESULTS_DIR = "results"
    
    # --- GÖRÜNTÜ AYARLARI ---
    # 224x224, ResNet ve MobileNet gibi modellerin standart giriş boyutudur
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    
    # --- EĞİTİM HİPERPARAMETRELERİ (SAVAŞ MODU) ---
    # RTX 3060 (12GB VRAM) için 32 güvenlidir. 
    # Eğer "OOM (Out of Memory)" hatası almazsanız 64 yapıp hızı artırabilirsiniz.
    BATCH_SIZE = 32
    
    # Epoch sayısını yüksek tutuyoruz (50). 
    # Merak etmeyin, main.py içindeki "EarlyStopping" sayesinde 
    # model öğrenmeyi durdurursa (örn. 20. epochta) eğitim otomatik bitecektir.
    EPOCHS = 50           
    
    # Başlangıç öğrenme hızı
    LEARNING_RATE = 0.001
    
    # --- OPTİMİZASYON VE MODEL ---
    # Seçenekler: 'adam', 'sgd_momentum', 'rmsprop'
    # İlk deneme (Baseline) için 'adam' kullanın.
    # İkinci deneme için bunu 'sgd_momentum' yapıp tekrar çalıştırın.
    OPTIMIZER = 'adam'    
    
    # SGD kullanırsanız bu momentum değeri devreye girer
    MOMENTUM = 0.9        
    
    # Model Seçimi: 'custom_cnn' (Kendi modelimiz) veya 'mobilenet' (Transfer Learning)
    # Önce kendi modelimizi eğitiyoruz.
    MODEL_TYPE = 'custom_cnn' 
    
    # --- KRİTİK AYARLAR ---
    # Tam eğitim için BURASI MUTLAKA FALSE OLMALI
    DEBUG_MODE = False    
    
    # Bilimsel tekrarlanabilirlik için sabit tohum
    SEED = 42             
    
    # Sonuç klasörünü oluştur
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)