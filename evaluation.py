import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import Config
import os

def plot_history(history):
    """Eğitim süreci boyunca Loss ve Accuracy değişimini çizer."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training & Validation Accuracy')

    # Loss
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
    """Verilen veri setinden tahminleri ve gerçek etiketleri toplar."""
    y_true = []
    y_pred = []
    
    # İlerleme çubuğu gibi davranması için enumerate kullanılabilir ama
    # TF dataset olduğu için doğrudan döngüye giriyoruz.
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    return np.array(y_true), np.array(y_pred)

def evaluate_set(model, ds, class_names, set_name="Test"):
    """
    Belirli bir veri seti (Train veya Test) için metrikleri hesaplar,
    raporu kaydeder ve Confusion Matrix çizer.
    Geriye 'weighted avg' metriklerini döndürür.
    """
    print(f"\n--- {set_name} Seti Değerlendiriliyor ---")
    
    y_true, y_pred = get_predictions_and_labels(model, ds)
    
    # 1. Confusion Matrix
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
    
    # 2. Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    
    # Raporu TXT olarak kaydet
    txt_path = os.path.join(Config.RESULTS_DIR, f'{set_name}_classification_report.txt')
    with open(txt_path, 'w') as f:
        f.write(report_str)
        
    print(f"{set_name} raporu ve matrisi kaydedildi.")
    
    # Karşılaştırma grafiği için genel skorları döndür
    return {
        'Accuracy': report_dict['accuracy'],
        'Precision': report_dict['weighted avg']['precision'],
        'Recall': report_dict['weighted avg']['recall'],
        'F1-Score': report_dict['weighted avg']['f1-score']
    }

def plot_comparison(train_metrics, test_metrics):
    """Train ve Test metriklerini yan yana sütun grafiği ile karşılaştırır."""
    
    metrics = list(train_metrics.keys()) # ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_vals = list(train_metrics.values())
    test_vals = list(test_metrics.values())
    
    x = np.arange(len(metrics))  # Etiket konumları
    width = 0.35  # Çubuk genişliği

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_vals, width, label='Training', color='#3498db')
    rects2 = ax.bar(x + width/2, test_vals, width, label='Test', color='#e74c3c')

    # Yazılar ve Etiketler
    ax.set_ylabel('Skor (0-1 Arası)')
    ax.set_title('Eğitim vs Test Performans Karşılaştırması')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1) # Y ekseni 0 ile 1.1 arasında olsun
    ax.legend()

    # Çubukların üzerine değerleri yaz
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    save_path = os.path.join(Config.RESULTS_DIR, 'train_vs_test_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Karşılaştırma grafiği kaydedildi: {save_path}")