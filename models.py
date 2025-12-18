import tensorflow as tf
from tensorflow.keras import layers, models, applications
from config import Config

# Veri Artırma (Data Augmentation) - Özgünlük için önemli
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

def build_model(num_classes):
    inputs = tf.keras.Input(shape=Config.IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x) # Normalizasyon

    if Config.MODEL_TYPE == 'custom_cnn':
        # --- Model 1: Özgün Basit CNN ---
        # Rapor için not: Bu mimariyi kendimiz tasarladık.
        
        # Blok 1
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        
        # Blok 2
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        
        # Blok 3
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        
        # Sınıflandırma Kafası
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x) # Overfitting önleme
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="Custom_CNN_Model")
        
    elif Config.MODEL_TYPE == 'mobilenet':
        # --- Model 2: Transfer Learning (Karşılaştırma) ---
        base_model = applications.MobileNetV2(
            input_shape=Config.IMG_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False # Ağırlıkları dondur
        
        # Preprocessing MobileNet'e özel olmalı
        x = applications.mobilenet_v2.preprocess_input(inputs) # Rescaling yerine bu
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="MobileNetV2_Transfer")
        
    return model