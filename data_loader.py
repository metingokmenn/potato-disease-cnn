import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from config import Config

def load_data():
    print(f"--- Veri Yükleniyor: {Config.DATASET_DIR} ---")
    
    # Eğitim Seti
    train_ds = image_dataset_from_directory(
        Config.DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=Config.SEED,
        image_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        label_mode='categorical'
    )
    
    # Doğrulama (Validation) Seti
    val_ds = image_dataset_from_directory(
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

    # DEBUG MODU: Hızlı test için veriyi kırp
    if Config.DEBUG_MODE:
        print("\n!!! DEBUG MODU: Sadece 1 batch veri kullanılıyor !!!\n")
        train_ds = train_ds.take(1)
        val_ds = val_ds.take(1)

    # Performans Optimizasyonu (Pipeline)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names