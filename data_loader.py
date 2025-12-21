import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from config import Config

def load_data():
    """
    Veri setini yükler ve eğitim, doğrulama ve test setlerine ayırır.
    
    Bu fonksiyon, patates hastalığı görüntü veri setini diskten yükler ve 
    üç ayrı veri setine böler: Eğitim (%80), Doğrulama (%10) ve Test (%10).
    Veri setleri TensorFlow Dataset formatında döndürülür ve performans 
    optimizasyonu için ön işleme tabi tutulur.
    
    Parametreler:
        Hiçbir parametre almaz. Tüm ayarlar Config sınıfından alınır.
    
    Returns:
        tuple: Dört elemanlı tuple:
            - train_ds (tf.data.Dataset): Eğitim veri seti (%80)
            - val_ds (tf.data.Dataset): Doğrulama veri seti (%10)
            - test_ds (tf.data.Dataset): Test veri seti (%10)
            - class_names (list): Sınıf isimlerinin listesi
    
    İşlem Adımları:
        1. Ana ayrım: Veri seti %80 eğitim ve %20 diğerleri olarak ayrılır
        2. İkincil ayrım: %20'lik parça yarı yarıya bölünür (%10 val, %10 test)
        3. Performans optimizasyonu: Shuffle ve prefetch işlemleri uygulanır
    
    Örnek Kullanım:
        >>> train_ds, val_ds, test_ds, class_names = load_data()
        --- Veri Yükleniyor: dataset ---
        Sınıflar: ['Early_Blight', 'Healthy', 'Late_Blight']
        Veri Dağılımı: Eğitim: %80 | Doğrulama: %10 | Test: %10
    """
    print(f"--- Veri Yükleniyor: {Config.DATASET_DIR} ---")
    
    train_ds = image_dataset_from_directory(
        Config.DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=Config.SEED,
        image_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        label_mode='categorical'
    )
    
    val_and_test_ds = image_dataset_from_directory(
        Config.DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=Config.SEED,
        image_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    print(f"Sınıflar: {class_names}")

    if Config.DEBUG_MODE:
        print("\n!!! DEBUG MODU: Veri seti küçültülüyor !!!")
        train_ds = train_ds.take(1)
        val_and_test_ds = val_and_test_ds.take(1) 
        val_ds = val_and_test_ds
        test_ds = val_and_test_ds
    else:
        val_batches = tf.data.experimental.cardinality(val_and_test_ds)
        test_size = val_batches // 2
        
        test_ds = val_and_test_ds.take(test_size)
        val_ds = val_and_test_ds.skip(test_size)
        
        print(f"Veri Dağılımı: Eğitim: %80 | Doğrulama: %10 | Test: %10")
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names