# Patates Hastalığı Sınıflandırma Projesi

Bu proje, derin öğrenme (deep learning) teknikleri kullanarak patates yapraklarında görülen hastalıkları otomatik olarak sınıflandırmayı amaçlamaktadır. Proje, Convolutional Neural Network (CNN) ve Transfer Learning yaklaşımlarını kullanarak üç farklı sınıfı ayırt edebilir: Erken Yanıklık (Early Blight), Sağlıklı (Healthy) ve Geç Yanıklık (Late Blight).

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Veri Seti Yapısı](#veri-seti-yapısı)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Yapılandırma](#yapılandırma)
- [Sonuçlar](#sonuçlar)
- [Sorun Giderme](#sorun-giderme)

## 🎯 Proje Hakkında

Bu proje, tarım alanında görüntü işleme ve makine öğrenmesi tekniklerini kullanarak patates bitkilerindeki hastalıkları otomatik olarak tespit etmeyi hedefler. Proje, iki farklı model mimarisi sunar:

1. **Custom CNN**: Özgün tasarlanmış basit convolutional neural network
2. **MobileNetV2**: Transfer learning ile ImageNet ağırlıklı MobileNetV2 tabanlı model

Her iki model de veri artırma (data augmentation), learning rate scheduling ve early stopping gibi gelişmiş tekniklerle optimize edilmiştir.

## ✨ Özellikler

- ✅ İki farklı model mimarisi desteği (Custom CNN ve MobileNetV2)
- ✅ Üç farklı optimizer seçeneği (Adam, SGD with Momentum, RMSprop)
- ✅ Otomatik veri bölme (80% eğitim, 10% doğrulama, 10% test)
- ✅ Gelişmiş callback'ler (Learning Rate Scheduler, Early Stopping)
- ✅ Detaylı performans metrikleri ve görselleştirmeler
- ✅ Confusion matrix ve classification report oluşturma
- ✅ Tek görüntü tahmin desteği
- ✅ Veri seti temizleme araçları

## 📦 Gereksinimler

### Yazılım Gereksinimleri

- Python 3.8 veya üzeri
- TensorFlow 2.x
- CUDA ve cuDNN (GPU desteği için opsiyonel)

### Python Kütüphaneleri

Proje gereksinimleri `requirements.txt` dosyasında listelenmiştir:

```
tensorflow
numpy
matplotlib
scikit-learn
seaborn
pandas
opencv-python
```

## 🚀 Kurulum

### 1. Projeyi İndirin

```bash
git clone <repository-url>
cd potato-disease-cnn
```

### 2. Sanal Ortam Oluşturun (Önerilen)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Gereksinimleri Yükleyin

```bash
pip install -r requirements.txt
```

### 4. GPU Desteğini Kontrol Edin (Opsiyonel)

```bash
python tensorflow_check.py
```

Bu komut, sisteminizde kaç adet GPU bulunduğunu gösterir. GPU yoksa model CPU üzerinde çalışacaktır.

## 📁 Veri Seti Yapısı

Proje, aşağıdaki klasör yapısını bekler:

```
dataset/
├── Early_Blight/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Healthy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Late_Blight/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**Önemli Notlar:**
- Klasör isimleri tam olarak `Early_Blight`, `Healthy` ve `Late_Blight` olmalıdır
- Görüntü formatları: JPG, JPEG, PNG desteklenir
- Görüntüler otomatik olarak 224x224 boyutuna yeniden boyutlandırılır

## 🛠️ Kullanım

### Adım 1: Veri Seti Hazırlığı

Eğer veri setiniz macOS'tan aktarıldıysa veya bozuk dosyalar içeriyorsa, önce temizlik yapın:

**macOS hayalet dosyalarını temizleme:**
```bash
python clean_mac_files.py
```

**Derinlemesine temizlik (bozuk görüntüleri kontrol etme):**
```bash
python deep_clean.py
```

### Adım 2: Yapılandırma Ayarları

`config.py` dosyasını açarak proje ayarlarını düzenleyin:

```python
# Model seçimi
MODEL_TYPE = 'mobilenet'  # 'custom_cnn' veya 'mobilenet'

# Optimizer seçimi
OPTIMIZER = 'adam'  # 'adam', 'sgd_momentum', 'rmsprop'

# Eğitim parametreleri
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

# Debug modu (hızlı test için)
DEBUG_MODE = False  # Tam eğitim için False olmalı
```

### Adım 3: Model Eğitimi

Ana eğitim scriptini çalıştırın:

```bash
python main.py
```

Eğitim sırasında:
- Veri seti otomatik olarak yüklenir ve bölünür
- Model oluşturulur ve derlenir
- Eğitim başlar ve ilerleme konsola yazdırılır
- Callback'ler otomatik olarak çalışır (learning rate scheduling, early stopping)

### Adım 4: Sonuçları İnceleme

Eğitim tamamlandıktan sonra, `results/` klasöründe şu dosyalar oluşturulur:

- `{model_type}_{optimizer}.keras`: Eğitilmiş model dosyası
- `history_graphs.png`: Eğitim süreci grafikleri (loss ve accuracy)
- `Training_confusion_matrix.png`: Eğitim seti confusion matrix
- `Test_confusion_matrix.png`: Test seti confusion matrix
- `Training_classification_report.txt`: Eğitim seti detaylı metrikler
- `Test_classification_report.txt`: Test seti detaylı metrikler
- `train_vs_test_comparison.png`: Eğitim ve test metriklerinin karşılaştırması

### Adım 5: Tahmin Yapma

Eğitilmiş model ile yeni görüntüler üzerinde tahmin yapmak için:

1. `prediction.py` dosyasını açın
2. `MODEL_PATH` değişkenini eğitilmiş model yoluna ayarlayın:
   ```python
   MODEL_PATH = 'results/mobilenet_adam.keras'
   ```
3. Test görüntüsünü proje klasörüne koyun (örn: `test_image.jpg`)
4. Scripti çalıştırın:
   ```bash
   python prediction.py
   ```

Alternatif olarak, Python'da doğrudan kullanabilirsiniz:

```python
from prediction import predict_image
predict_image("test_image.jpg")
```

## 📂 Proje Yapısı

```
potato-disease-cnn/
├── config.py              # Proje yapılandırma ayarları
├── data_loader.py         # Veri yükleme ve bölme fonksiyonları
├── models.py              # Model mimarileri (Custom CNN, MobileNetV2)
├── main.py                # Ana eğitim scripti
├── evaluation.py          # Performans değerlendirme ve görselleştirme
├── prediction.py          # Tek görüntü tahmin scripti
├── clean_mac_files.py     # macOS hayalet dosya temizleme
├── deep_clean.py         # Bozuk görüntü kontrolü ve temizleme
├── tensorflow_check.py    # GPU kontrolü
├── requirements.txt      # Python bağımlılıkları
├── dataset/              # Veri seti klasörü
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
└── results/              # Eğitim sonuçları (otomatik oluşturulur)
    ├── *.keras           # Eğitilmiş modeller
    ├── *.png             # Grafikler
    └── *.txt             # Metrik raporları
```

## ⚙️ Yapılandırma

### Config.py Parametreleri

| Parametre | Açıklama | Varsayılan Değer |
|-----------|----------|------------------|
| `DATASET_DIR` | Veri seti klasör yolu | `"dataset"` |
| `RESULTS_DIR` | Sonuçlar klasör yolu | `"results"` |
| `IMG_HEIGHT` | Görüntü yüksekliği (piksel) | `224` |
| `IMG_WIDTH` | Görüntü genişliği (piksel) | `224` |
| `BATCH_SIZE` | Batch boyutu | `16` |
| `EPOCHS` | Maksimum epoch sayısı | `30` |
| `LEARNING_RATE` | Öğrenme hızı | `0.001` |
| `OPTIMIZER` | Optimizer tipi | `'adam'` |
| `MODEL_TYPE` | Model mimarisi | `'mobilenet'` |
| `DEBUG_MODE` | Hata ayıklama modu | `False` |
| `SEED` | Rastgele sayı tohumu | `42` |

### Model Tipleri

**Custom CNN:**
- 3 Conv2D bloğu (32, 64, 128 filtre)
- MaxPooling2D katmanları
- Dense katmanlar (128 nöron)
- Dropout (0.5) ile overfitting önleme

**MobileNetV2:**
- ImageNet ağırlıklı MobileNetV2 taban modeli (dondurulmuş)
- GlobalAveragePooling2D katmanı
- Dropout (0.2) katmanı
- Özel sınıflandırma kafası

### Optimizer Seçenekleri

- **Adam**: Adaptif öğrenme hızı, genellikle en iyi performans
- **SGD with Momentum**: Momentum değeri 0.9 ile klasik optimizasyon
- **RMSprop**: Adaptif öğrenme hızı, RNN'ler için popüler

## 📊 Sonuçlar

Eğitim tamamlandıktan sonra, `results/` klasöründe şu çıktılar oluşturulur:

### Model Dosyası
- Format: `.keras`
- İsimlendirme: `{model_type}_{optimizer}.keras`
- Örnek: `mobilenet_adam.keras`

### Görselleştirmeler
- **history_graphs.png**: Epoch bazında loss ve accuracy grafikleri
- **Training_confusion_matrix.png**: Eğitim seti karışıklık matrisi
- **Test_confusion_matrix.png**: Test seti karışıklık matrisi
- **train_vs_test_comparison.png**: Eğitim ve test metriklerinin karşılaştırmalı grafiği

### Metrik Raporları
- **Training_classification_report.txt**: Eğitim seti için precision, recall, F1-score
- **Test_classification_report.txt**: Test seti için precision, recall, F1-score

## 🔧 Sorun Giderme

### GPU Bulunamıyor

**Sorun:** `UYARI: GPU bulunamadı. CPU kullanılıyor.`

**Çözüm:**
- CUDA ve cuDNN'in doğru kurulu olduğundan emin olun
- TensorFlow GPU sürümünün yüklü olduğunu kontrol edin: `pip install tensorflow-gpu`
- GPU sürücülerinin güncel olduğundan emin olun

### Out of Memory (OOM) Hatası

**Sorun:** `ResourceExhaustedError: OOM when allocating tensor`

**Çözüm:**
- `config.py` dosyasında `BATCH_SIZE` değerini küçültün (örn: 32 → 16)
- Model tipini `custom_cnn` olarak değiştirin (daha az parametre)
- Görüntü boyutunu küçültün (224 → 128)

### Veri Seti Bulunamıyor

**Sorun:** `FileNotFoundError: dataset klasörü bulunamadı`

**Çözüm:**
- `dataset/` klasörünün proje kök dizininde olduğundan emin olun
- Klasör isimlerinin doğru olduğunu kontrol edin: `Early_Blight`, `Healthy`, `Late_Blight`

### Bozuk Görüntü Hatası

**Sorun:** Eğitim sırasında görüntü decode hatası

**Çözüm:**
```bash
python deep_clean.py
```

Bu script, bozuk görüntü dosyalarını otomatik olarak tespit eder ve siler.

### Model Dosyası Bulunamıyor (Tahmin için)

**Sorun:** `HATA: Model dosyası bulunamadı`

**Çözüm:**
- `prediction.py` dosyasındaki `MODEL_PATH` değişkenini kontrol edin
- Model dosyasının `results/` klasöründe olduğundan emin olun
- Dosya adının doğru olduğunu kontrol edin (örn: `mobilenet_adam.keras`)

## 📝 Notlar

- Eğitim süresi, veri seti boyutuna ve kullanılan donanıma bağlıdır
- GPU kullanımı eğitim süresini önemli ölçüde kısaltır
- Early stopping sayesinde model gereksiz yere uzun süre eğitilmez
- Learning rate scheduler, modelin daha iyi öğrenmesine yardımcı olur
- Debug modu (`DEBUG_MODE = True`), hızlı test için veri setini küçültür ve kodlarda hata olup olmadığını görmek için kullanılır

