# 🏥 Pneumonia CNN Classification

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A deep learning project that analyzes chest X-rays to detect pneumonia. Developed using EfficientNet-B0 model with Grad-CAM visual explanations and web interface.

## 📊 Dataset Statistics

| Set | Normal | Pneumonia | Total |
|-----|--------|-----------|--------|
| **Train** | 4,755 | 5,088 | 9,843 |
| **Validation** | 1,238 | 1,845 | 3,083 |
| **Test** | 1,242 | 1,864 | 3,106 |
| **TOTAL** | **7,235** | **8,797** | **16,032** |

## 🚀 Features

- **EfficientNet-B0** transfer learning
- **Grad-CAM** visual explanations
- **Web interface** (Gradio)
- **Confidence threshold** system (80%)
- **Data augmentation** techniques
- **Class balancing** for imbalanced data

## 🛠️ Technologies

- **Python 3.x**
- **PyTorch** - Deep learning framework
- **Torchvision** - Image processing
- **Gradio** - Web interface
- **OpenCV** - Grad-CAM visualization
- **Scikit-learn** - Metrics calculation
- **Matplotlib/Seaborn** - Plotting

## 📁 Project Structure

```
pneumonia-cnn-classification/
├── main.py                 # Model training and testing
├── app.py                  # Gradio web interface
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
├── README.md              # This file
├── dataset/               # Dataset
│   ├── train/             # Training set (9,843 images)
│   │   ├── NORMAL/        # 4,755 normal X-rays
│   │   └── PNEUMONIA/     # 5,088 pneumonia X-rays
│   ├── val/               # Validation set (3,083 images)
│   │   ├── NORMAL/        # 1,238 normal X-rays
│   │   └── PNEUMONIA/     # 1,845 pneumonia X-rays
│   └── test/              # Test set (3,106 images)
│       ├── NORMAL/        # 1,242 normal X-rays
│       └── PNEUMONIA/     # 1,864 pneumonia X-rays
└── results/               # Trained models and plots
    ├── model_efficientnet_b0.pth
    ├── confusion_matrix.png
    └── training_plots.png
```

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/justm3lih/pneumonia-cnn-classification.git
cd pneumonia-cnn-classification
```

2. **Install required libraries:**
```bash
pip install -r requirements.txt
```

**Manual installation:**
```bash
pip install torch torchvision gradio opencv-python scikit-learn matplotlib seaborn pillow
```

3. **Prepare the dataset:**
   - Place train/val/test folders in `dataset/` directory
   - Each folder should contain NORMAL and PNEUMONIA subfolders

## 🎯 Usage

### Model Training
```bash
python main.py
```

### Web Interface
```bash
python app.py
```

The web interface will open at `http://localhost:7860`.

## 📈 Model Performance

- **Model:** EfficientNet-B0 (Transfer Learning)
- **Image Size:** 224x224
- **Batch Size:** 32
- **Epochs:** 10
- **Optimizer:** Adam (lr=0.001)

### Training Features:
- **Data Augmentation:** Random flip, rotation, color jitter, Gaussian blur, random erasing
- **Class Weights:** Weighting for imbalanced dataset
- **Transfer Learning:** Only final layer training

## 🔍 Grad-CAM Explanation

The model visualizes which regions it focuses on for pneumonia detection using Grad-CAM:
- **Red regions:** Areas the model pays attention to
- **Blue regions:** Less important areas

## ⚠️ Important Notes

- This project is for **educational and demo purposes only**
- **Should NOT be used for medical diagnosis**
- Results are for reference only
- Consult a medical professional for real medical conditions

## 📊 Results

After model training:
- Loss and accuracy plots in `results/` folder
- Confusion matrix visualization
- Trained model file (16MB)

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👨‍💻 Developer

**Osman Melih Çınar**  
📧 [mlhcnr1903@icloud.com](mailto:mlhcnr1903@icloud.com)  
💼 [LinkedIn](https://www.linkedin.com/in/osman-melih-%C3%A7%C4%B1nar-76312332a/)

## 🙏 Acknowledgments

- [Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- PyTorch and Torchvision teams
- Gradio team

---

## 🇹🇷 Türkçe Özet
Bu proje, göğüs röntgenlerinden zatürre (pneumonia) tespiti için EfficientNet-B0 tabanlı bir CNN modelidir. 
Grad-CAM ile görsel açıklama, Gradio tabanlı web arayüzü ve veri artırma teknikleri içerir.