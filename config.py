import os

class Config:
    """
    Proje yapılandırma sınıfı.
    
    Bu sınıf, patates hastalığı sınıflandırma projesi için tüm yapılandırma 
    parametrelerini içerir. Dosya yolları, görüntü ayarları, eğitim 
    hiperparametreleri ve model seçenekleri burada tanımlanır.
    """
    
    DATASET_DIR = "dataset"
    """
    str: Veri setinin bulunduğu klasör yolu.
         Windows'ta klasör yapısının "dataset" şeklinde olduğundan emin olun.
    """
    
    RESULTS_DIR = "results"
    """
    str: Eğitim sonuçlarının (modeller, grafikler, raporlar) kaydedileceği klasör yolu.
    """
    
    IMG_HEIGHT = 224
    """
    int: Görüntülerin yüksekliği (piksel).
         224x224, ResNet ve MobileNet gibi modellerin standart giriş boyutudur.
    """
    
    IMG_WIDTH = 224
    """
    int: Görüntülerin genişliği (piksel).
         224x224, ResNet ve MobileNet gibi modellerin standart giriş boyutudur.
    """
    
    CHANNELS = 3
    """
    int: Görüntü kanal sayısı (RGB için 3).
    """
    
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    """
    tuple: Görüntü boyutu (yükseklik, genişlik) tuple'ı.
    """

    BATCH_SIZE = 16
    """
    int: Eğitim sırasında kullanılacak batch (toplu işlem) boyutu.
         RTX 3060 (12GB VRAM) için 32 güvenlidir. 
         Eğer "OOM (Out of Memory)" hatası almazsanız 64 yapıp hızı artırabilirsiniz.
         BATCH_SIZE = 32 ile denendi ve Out of Memory hatası alındı.
    """
    
    
    EPOCHS = 30
    """
    int: Eğitim sırasında kullanılacak maksimum epoch (dönem) sayısı.
         Epoch sayısını yüksek tutuyoruz.
         main.py içindeki EarlyStopping callback sayesinde model öğrenmeyi durdurursa eğitim otomatik bitecektir.
    """
    
    LEARNING_RATE = 0.001
    """
    float: Başlangıç öğrenme hızı (learning rate).
           Modelin ağırlıklarını güncellerken kullanılan adım büyüklüğü.
    """
    

    OPTIMIZER = 'adam'
    """
    str: Kullanılacak optimizasyon algoritması.
         Seçenekler: 'adam', 'sgd_momentum', 'rmsprop'
    """
    
    MOMENTUM = 0.9
    """
    float: SGD optimizasyonu için momentum değeri (0.0 ile 1.0 arası).
    """
    
    MODEL_TYPE = 'mobilenet'
    """
    str: Kullanılacak model tipi.
         Seçenekler: 'custom_cnn' (Kendi modelimiz) veya 'mobilenet' (Transfer Learning)
    """
    
    DEBUG_MODE = False
    """
    bool: Hata ayıklama modu.
          True olduğunda veri seti küçültülür ve hızlı test için kullanılır.
          Tam eğitim için False, kodların doğru çalıştığını görmek için True kullanılır.
    """
    
    SEED = 42
    """
    int: Rastgele sayı üreteci için sabit tohum değeri.
         Bilimsel tekrarlanabilirlik için sabit tohum kullanılır.
    """
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)