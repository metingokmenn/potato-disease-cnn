import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from config import Config
import os

def plot_history(history):
    # Loss ve Accuracy Grafikleri
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    save_path = os.path.join(Config.RESULTS_DIR, 'training_graphs.png')
    plt.savefig(save_path)
    print(f"Grafikler kaydedildi: {save_path}")
    plt.close()

def evaluate_model(model, val_ds, class_names):
    print("\n--- Detaylı Performans Analizi Başlıyor ---")
    
    # Tahminleri topla
    y_true = []
    y_pred = []

    # Batch'ler halinde tahmin yap
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # 1. Classification Report (Precision, Recall, F1)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nSınıflandırma Raporu:")
    print(report)
    
    # Raporu dosyaya yaz
    with open(os.path.join(Config.RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 2. Confusion Matrix (Karmaşıklık Matrisi)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gerçek (Actual)')
    plt.title('Confusion Matrix')
    
    save_path = os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Confusion Matrix kaydedildi: {save_path}")
    plt.close()