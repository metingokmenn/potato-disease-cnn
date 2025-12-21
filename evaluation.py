import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import Config
import os

def plot_history(history):
    """
    Eğitim süreci boyunca Loss ve Accuracy değişimini görselleştirir.
    
    Bu fonksiyon, model eğitimi sırasında kaydedilen loss ve accuracy 
    değerlerini kullanarak eğitim ve doğrulama metriklerinin zaman içindeki 
    değişimini gösteren grafikler oluşturur. Grafikler PNG formatında kaydedilir.
    
    Parametreler:
        history (tf.keras.callbacks.History): Model.fit() metodundan dönen 
                                              eğitim geçmişi nesnesi.
                                              İçinde 'accuracy', 'val_accuracy', 
                                              'loss', 'val_loss' anahtarları bulunur.
    
    Returns:
        None: Grafikler dosyaya kaydedilir ve konsola bilgi yazdırılır.
    
    Oluşturulan Grafikler:
        - Sol panel: Training & Validation Accuracy
        - Sağ panel: Training & Validation Loss
    
    Örnek Kullanım:
        >>> history = model.fit(train_ds, validation_data=val_ds, epochs=10)
        >>> plot_history(history)
        Eğitim grafikleri kaydedildi: results/history_graphs.png
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training & Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training & Validation Loss')
    
    save_path = os.path.join(Config.RESULTS_DIR, 'history_graphs.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Eğitim grafikleri kaydedildi: {save_path}")

def get_predictions_and_labels(model, ds):
    """
    Verilen veri setinden model tahminlerini ve gerçek etiketleri toplar.
    
    Bu fonksiyon, TensorFlow Dataset üzerinde döngü yaparak her batch için 
    model tahminlerini alır ve gerçek etiketlerle birlikte numpy dizilerine 
    dönüştürür. Kategorik etiketler argmax ile sınıf indekslerine çevrilir.
    
    Parametreler:
        model (tf.keras.Model): Eğitilmiş Keras modeli.
        ds (tf.data.Dataset): Tahmin yapılacak veri seti.
                              Her eleman (images, labels) tuple'ı içermelidir.
    
    Returns:
        tuple: İki numpy dizisi içeren tuple:
            - y_true (np.ndarray): Gerçek sınıf indeksleri (1D array)
            - y_pred (np.ndarray): Tahmin edilen sınıf indeksleri (1D array)
    
    Örnek Kullanım:
        >>> y_true, y_pred = get_predictions_and_labels(model, test_ds)
        >>> print(f"Toplam örnek: {len(y_true)}")
    """
    y_true = []
    y_pred = []
    
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    return np.array(y_true), np.array(y_pred)

def evaluate_set(model, ds, class_names, set_name="Test"):
    """
    Belirli bir veri seti için detaylı performans metriklerini hesaplar ve görselleştirir.
    
    Bu fonksiyon, verilen veri seti üzerinde model performansını değerlendirir.
    Confusion matrix oluşturur, classification report hesaplar ve bunları 
    dosyalara kaydeder. Weighted average metriklerini döndürür.
    
    Parametreler:
        model (tf.keras.Model): Değerlendirilecek eğitilmiş Keras modeli.
        ds (tf.data.Dataset): Değerlendirme yapılacak veri seti.
        class_names (list): Sınıf isimlerinin listesi.
                           Örnek: ['Early_Blight', 'Healthy', 'Late_Blight']
        set_name (str): Veri setinin adı (rapor ve dosya adları için kullanılır).
                       Varsayılan: "Test"
                       Örnek: "Training", "Validation", "Test"
    
    Returns:
        dict: Performans metriklerini içeren sözlük:
            - 'Accuracy' (float): Genel doğruluk skoru
            - 'Precision' (float): Weighted average precision
            - 'Recall' (float): Weighted average recall
            - 'F1-Score' (float): Weighted average F1-score
    
    Oluşturulan Dosyalar:
        - {set_name}_confusion_matrix.png: Confusion matrix görseli
        - {set_name}_classification_report.txt: Detaylı metrik raporu
    
    Örnek Kullanım:
        >>> metrics = evaluate_set(model, test_ds, class_names, "Test")
        --- Test Seti Değerlendiriliyor ---
        Test raporu ve matrisi kaydedildi.
    """
    print(f"\n--- {set_name} Seti Değerlendiriliyor ---")
    
    y_true, y_pred = get_predictions_and_labels(model, ds)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin (Predicted)')
    plt.ylabel('Gerçek (Actual)')
    plt.title(f'{set_name} Seti - Confusion Matrix')
    
    cm_path = os.path.join(Config.RESULTS_DIR, f'{set_name}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    
    txt_path = os.path.join(Config.RESULTS_DIR, f'{set_name}_classification_report.txt')
    with open(txt_path, 'w') as f:
        f.write(report_str)
        
    print(f"{set_name} raporu ve matrisi kaydedildi.")
    
    return {
        'Accuracy': report_dict['accuracy'],
        'Precision': report_dict['weighted avg']['precision'],
        'Recall': report_dict['weighted avg']['recall'],
        'F1-Score': report_dict['weighted avg']['f1-score']
    }

def plot_comparison(train_metrics, test_metrics):
    """
    Eğitim ve test seti metriklerini yan yana sütun grafiği ile karşılaştırır.
    
    Bu fonksiyon, eğitim ve test setleri için hesaplanan performans metriklerini 
    (Accuracy, Precision, Recall, F1-Score) görsel olarak karşılaştırır. 
    Yan yana çubuk grafikler oluşturur ve her çubuğun üzerine değerleri yazar.
    
    Parametreler:
        train_metrics (dict): Eğitim seti metrikleri.
                             Anahtarlar: 'Accuracy', 'Precision', 'Recall', 'F1-Score'
                             Değerler: float (0-1 arası)
        test_metrics (dict): Test seti metrikleri.
                            Anahtarlar: 'Accuracy', 'Precision', 'Recall', 'F1-Score'
                            Değerler: float (0-1 arası)
    
    Returns:
        None: Grafik dosyaya kaydedilir ve konsola bilgi yazdırılır.
    
    Oluşturulan Grafik:
        - 4 metrik için yan yana çubuk grafikler
        - Mavi çubuklar: Training metrikleri
        - Kırmızı çubuklar: Test metrikleri
        - Her çubuğun üzerinde değer gösterimi
    
    Örnek Kullanım:
        >>> train_metrics = {'Accuracy': 0.95, 'Precision': 0.94, ...}
        >>> test_metrics = {'Accuracy': 0.92, 'Precision': 0.91, ...}
        >>> plot_comparison(train_metrics, test_metrics)
        Karşılaştırma grafiği kaydedildi: results/train_vs_test_comparison.png
    """
    
    metrics = list(train_metrics.keys())
    train_vals = list(train_metrics.values())
    test_vals = list(test_metrics.values())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_vals, width, label='Training', color='#3498db')
    rects2 = ax.bar(x + width/2, test_vals, width, label='Test', color='#e74c3c')

    ax.set_ylabel('Skor (0-1 Arası)')
    ax.set_title('Eğitim vs Test Performans Karşılaştırması')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    save_path = os.path.join(Config.RESULTS_DIR, 'train_vs_test_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Karşılaştırma grafiği kaydedildi: {save_path}")