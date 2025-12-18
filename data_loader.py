import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from config import Config

def load_data():
    print(f"--- Veri Yükleniyor: {Config.DATASET_DIR} ---")
    
    # 1. Ana Ayrım: %80 Eğitim, %20 Diğerleri
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

    # DEBUG MODU İÇİN ÖZEL DURUM
    if Config.DEBUG_MODE:
        print("\n!!! DEBUG MODU: Veri seti küçültülüyor !!!")
        train_ds = train_ds.take(1)
        val_and_test_ds = val_and_test_ds.take(1) 
        # Debug modunda val ve test aynı tek batch olsun, hata vermesin
        val_ds = val_and_test_ds
        test_ds = val_and_test_ds
    else:
        # 2. İkincil Ayrım: %20'lik parçayı yarı yarıya böl (%10 Val, %10 Test)
        val_batches = tf.data.experimental.cardinality(val_and_test_ds)
        test_size = val_batches // 2
        
        test_ds = val_and_test_ds.take(test_size)
        val_ds = val_and_test_ds.skip(test_size)
        
        print(f"Veri Dağılımı: Eğitim: %80 | Doğrulama: %10 | Test: %10")

    # 3. Performans Optimizasyonu
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Artık 3 parça döndürüyoruz
    return train_ds, val_ds, test_ds, class_names