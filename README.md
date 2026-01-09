# Potato Disease Classification Project

This project aims to automatically classify diseases seen in potato leaves using deep learning techniques. Using Convolutional Neural Network (CNN) and Transfer Learning approaches, the project can distinguish between three different classes: Early Blight, Healthy, and Late Blight.

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## 🎯 About the Project

This project aims to automatically detect diseases in potato plants using image processing and machine learning techniques in the field of agriculture. The project offers two different model architectures:

1. **Custom CNN**: Uniquely designed simple convolutional neural network
2. **MobileNetV2**: MobileNetV2-based model with ImageNet weights using transfer learning

Both models are optimized with advanced techniques such as data augmentation, learning rate scheduling, and early stopping.

## ✨ Features

- ✅ Support for two different model architectures (Custom CNN and MobileNetV2)
- ✅ Three different optimizer options (Adam, SGD with Momentum, RMSprop)
- ✅ Automatic data splitting (80% training, 10% validation, 10% test)
- ✅ Advanced callbacks (Learning Rate Scheduler, Early Stopping)
- ✅ Detailed performance metrics and visualizations
- ✅ Confusion matrix and classification report generation
- ✅ Single image prediction support
- ✅ Dataset cleaning tools

## 📦 Requirements

### Software Requirements

- Python 3.8 or higher
- TensorFlow 2.x
- CUDA and cuDNN (optional for GPU support)

### Python Libraries

Project requirements are listed in the `requirements.txt` file:

```
tensorflow
numpy
matplotlib
scikit-learn
seaborn
pandas
opencv-python
```

## 🚀 Installation

### 1. Clone the Project

```bash
git clone <repository-url>
cd potato-disease-cnn
```

### 2. Create a Virtual Environment (Recommended)

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

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Check GPU Support (Optional)

```bash
python tensorflow_check.py
```

This command shows how many GPUs are available on your system. If there is no GPU, the model will run on the CPU.

## 📁 Dataset Structure

The project expects the following folder structure:

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

**Important Notes:**

- Folder names must be exactly `Early_Blight`, `Healthy`, and `Late_Blight`
- Image formats: JPG, JPEG, PNG are supported
- Images are automatically resized to 224x224 dimensions

## 🛠️ Usage

### Step 1: Dataset Preparation

If your dataset was transferred from macOS or contains corrupt files, perform cleaning first:

**Cleaning macOS ghost files:**

```bash
python clean_mac_files.py
```

**Deep cleaning (checking for corrupt images):**

```bash
python deep_clean.py
```

### Step 2: Configuration Settings

Edit the project settings by opening the `config.py` file:

```python
# Model selection
MODEL_TYPE = 'mobilenet'  # 'custom_cnn' or 'mobilenet'

# Optimizer selection
OPTIMIZER = 'adam'  # 'adam', 'sgd_momentum', 'rmsprop'

# Training parameters
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

# Debug mode (for quick testing)
DEBUG_MODE = False  # Should be False for full training
```

### Step 3: Model Training

Run the main training script:

```bash
python main.py
```

During training:

- The dataset is automatically loaded and split
- The model is created and compiled
- Training starts and progress is printed to the console
- Callbacks run automatically (learning rate scheduling, early stopping)

### Step 4: Reviewing Results

After training is complete, the following files are created in the `results/` folder:

- `{model_type}_{optimizer}.keras`: Trained model file
- `history_graphs.png`: Training process graphs (loss and accuracy)
- `Training_confusion_matrix.png`: Training set confusion matrix
- `Test_confusion_matrix.png`: Test set confusion matrix
- `Training_classification_report.txt`: Training set detailed metrics
- `Test_classification_report.txt`: Test set detailed metrics
- `train_vs_test_comparison.png`: Comparison of training and test metrics

### Step 5: Making Predictions

To make predictions on new images with the trained model:

1. Open the `prediction.py` file
2. Set the `MODEL_PATH` variable to the path of the trained model:
   ```python
   MODEL_PATH = 'results/mobilenet_adam.keras'
   ```
3. Place the test image in the project folder (e.g., `test_image.jpg`)
4. Run the script:
   ```bash
   python prediction.py
   ```

Alternatively, you can use it directly in Python:

```python
from prediction import predict_image
predict_image("test_image.jpg")
```

## 📂 Project Structure

```
potato-disease-cnn/
├── config.py              # Project configuration settings
├── data_loader.py         # Data loading and splitting functions
├── models.py              # Model architectures (Custom CNN, MobileNetV2)
├── main.py                # Main training script
├── evaluation.py          # Performance evaluation and visualization
├── prediction.py          # Single image prediction script
├── clean_mac_files.py     # macOS ghost file cleaning
├── deep_clean.py         # Corrupt image check and cleaning
├── tensorflow_check.py    # GPU check
├── requirements.txt      # Python dependencies
├── dataset/              # Dataset folder
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
└── results/              # Training results (automatically generated)
    ├── *.keras           # Trained models
    ├── *.png             # Graphs
    └── *.txt             # Metric reports
```

## ⚙️ Configuration

### Config.py Parameters

| Parameter       | Description              | Default Value |
| --------------- | ------------------------ | ------------- |
| `DATASET_DIR`   | Dataset folder path      | `"dataset"`   |
| `RESULTS_DIR`   | Results folder path      | `"results"`   |
| `IMG_HEIGHT`    | Image height (pixels)    | `224`         |
| `IMG_WIDTH`     | Image width (pixels)     | `224`         |
| `BATCH_SIZE`    | Batch size               | `16`          |
| `EPOCHS`        | Maximum number of epochs | `30`          |
| `LEARNING_RATE` | Learning rate            | `0.001`       |
| `OPTIMIZER`     | Optimizer type           | `'adam'`      |
| `MODEL_TYPE`    | Model architecture       | `'mobilenet'` |
| `DEBUG_MODE`    | Debug mode               | `False`       |
| `SEED`          | Random number seed       | `42`          |

### Model Types

**Custom CNN:**

- 3 Conv2D blocks (32, 64, 128 filters)
- MaxPooling2D layers
- Dense layers (128 neurons)
- Overfitting prevention with Dropout (0.5)

**MobileNetV2:**

- ImageNet weighted MobileNetV2 base model (frozen)
- GlobalAveragePooling2D layer
- Dropout (0.2) layer
- Custom classification head

### Optimizer Options

- **Adam**: Adaptive learning rate, generally best performance
- **SGD with Momentum**: Classic optimization with momentum value of 0.9
- **RMSprop**: Adaptive learning rate, popular for RNNs

## 📊 Results

After training is complete, the following outputs are generated in the `results/` folder:

### Model File

- Format: `.keras`
- Naming: `{model_type}_{optimizer}.keras`
- Example: `mobilenet_adam.keras`

### Visualizations

- **history_graphs.png**: Loss and accuracy graphs per epoch
- **Training_confusion_matrix.png**: Training set confusion matrix
- **Test_confusion_matrix.png**: Test set confusion matrix
- **train_vs_test_comparison.png**: Comparative graph of training and test metrics

### Metric Reports

- **Training_classification_report.txt**: Precision, recall, F1-score for the training set
- **Test_classification_report.txt**: Precision, recall, F1-score for the test set

## 🔧 Troubleshooting

### GPU Not Found

**Issue:** `WARNING: GPU not found. Using CPU.`

**Solution:**

- Ensure CUDA and cuDNN are correctly installed
- Check if TensorFlow GPU version is installed: `pip install tensorflow-gpu`
- Ensure GPU drivers are up to date

### Out of Memory (OOM) Error

**Issue:** `ResourceExhaustedError: OOM when allocating tensor`

**Solution:**

- Decrease the `BATCH_SIZE` value in the `config.py` file (e.g., 32 → 16)
- Change the model type to `custom_cnn` (fewer parameters)
- Reduce the image size (224 → 128)

### Dataset Not Found

**Issue:** `FileNotFoundError: dataset folder not found`

**Solution:**

- Ensure the `dataset/` folder is in the project root directory
- Check that folder names are correct: `Early_Blight`, `Healthy`, `Late_Blight`

### Corrupt Image Error

**Issue:** Image decode error during training

**Solution:**

```bash
python deep_clean.py
```

This script automatically detects and deletes corrupt image files.

### Model File Not Found (For Prediction)

**Issue:** `ERROR: Model file not found`

**Solution:**

- Check the `MODEL_PATH` variable in the `prediction.py` file
- Ensure the model file is in the `results/` folder
- Check that the filename is correct (e.g., `mobilenet_adam.keras`)

## 📝 Notes

- Training time depends on the dataset size and the hardware used
- GPU usage significantly reduces training time
- Early stopping prevents the model from being trained for unnecessarily long periods
- Learning rate scheduler helps the model learn better
- Debug mode (`DEBUG_MODE = True`) reduces the dataset size for quick testing and is used to see if there are errors in the code
