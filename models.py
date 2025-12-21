import tensorflow as tf
from tensorflow.keras import layers, models, applications
from config import Config

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

def build_model(num_classes):
    """
    Patates hastalığı sınıflandırma modeli oluşturur.
    
    Bu fonksiyon, Config sınıfındaki MODEL_TYPE ayarına göre iki farklı 
    model mimarisinden birini oluşturur:
    - 'custom_cnn': Özgün tasarlanmış basit CNN mimarisi
    - 'mobilenet': Transfer learning ile MobileNetV2 tabanlı model
    
    Her iki model de veri artırma (data augmentation) ve normalizasyon 
    katmanları içerir. Model, belirtilen sınıf sayısına göre softmax 
    aktivasyonlu çıkış katmanına sahiptir.
    
    Parametreler:
        num_classes (int): Sınıflandırılacak sınıf sayısı.
                          Örnek: 3 (Early_Blight, Healthy, Late_Blight)
    
    Returns:
        tf.keras.Model: Derlenmemiş Keras modeli.
                        Model tipine göre "Custom_CNN_Model" veya 
                        "MobileNetV2_Transfer" isimli model döner.
    
    Model Mimarileri:
        1. Custom CNN:
           - 3 Conv2D bloğu (32, 64, 128 filtre)
           - MaxPooling2D katmanları
           - Dense katmanlar (128 nöron)
           - Dropout (0.5) ile overfitting önleme
        
        2. MobileNetV2 Transfer Learning:
           - ImageNet ağırlıklı MobileNetV2 taban modeli (dondurulmuş)
           - GlobalAveragePooling2D katmanı
           - Dropout (0.2) katmanı
           - Özel sınıflandırma kafası
    
    Örnek Kullanım:
        >>> model = build_model(3)
        >>> model.summary()
    """
    inputs = tf.keras.Input(shape=Config.IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x) # Normalizasyon

    if Config.MODEL_TYPE == 'custom_cnn':
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="Custom_CNN_Model")
        
    elif Config.MODEL_TYPE == 'mobilenet':
        base_model = applications.MobileNetV2(
            input_shape=Config.IMG_SIZE + (3,),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        x = applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="MobileNetV2_Transfer")
        
    return model