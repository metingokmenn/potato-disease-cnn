"""
TensorFlow GPU kontrolü için yardımcı script.

Bu script, sistemde TensorFlow'un kullanabileceği GPU sayısını kontrol eder.
Eğitim öncesi GPU erişimini doğrulamak için kullanılabilir.
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))