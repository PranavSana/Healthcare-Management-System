# Healthcare Management System (HMS)

A comprehensive web-based healthcare management platform that leverages machine learning to predict and diagnose medical conditions using medical imaging (MRI, CT scans) and patient data.

## 📋 Project Overview

This Healthcare Management System is designed to assist healthcare professionals in diagnosing three major health conditions:

1. **Brain Tumor Detection** - Uses CNN to classify brain MRI scans
2. **Lung Cancer Detection** - Uses CNN to classify CT scans
3. **Heart Disease Prediction** - Uses machine learning on patient health metrics

The system provides a user-friendly web interface built with Flask, allowing medical professionals to upload scans and receive predictions with confidence scores.

---

## 🎯 Features

- **Multi-disease Detection System**: Support for brain tumors, lung cancer, and heart disease
- **Deep Learning Models**: 
  - CNN-based models for image classification (brain tumors & lung cancer)
  - Machine learning model for heart disease prediction
- **Web-based Interface**: Built with Flask for easy access and deployment
- **Real-time Predictions**: Get instant diagnostic predictions
- **Professional Dashboard**: Clean UI for medical professionals
- **Image Upload Support**: Drag-and-drop interface for medical images

---

## 📁 Project Structure

```
HMS/
├── app.py                          # Main Flask application
├── README.md                       # This file
├── requirements.txt                # Python dependencies
│
├── models/                         # ML/DL models and training scripts
│   ├── heart.csv                   # Heart disease dataset
│   ├── heart_model.pkl             # Trained heart disease model
│   ├── train_heart_model.py        # Heart model training script
│   │
│   ├── brain_tumor/                # Brain tumor detection
│   │   ├── brain_tumor_cnn.h5      # Trained model (H5 format)
│   │   ├── brain_tumor_model.keras # Trained model (Keras format)
│   │   ├── predict_brain.py        # Inference script
│   │   ├── train_brain_model.py    # Training script
│   │   ├── test_image_loader.py    # Utility for loading test images
│   │   ├── Data/                   # MRI scan dataset (NOT in GitHub - see note below)
│   │   │   ├── Testing/
│   │   │   │   ├── glioma/
│   │   │   │   ├── meningioma/
│   │   │   │   ├── notumor/
│   │   │   │   └── pituitary/
│   │   │   └── Training/
│   │   │       ├── glioma/
│   │   │       ├── meningioma/
│   │   │       ├── notumor/
│   │   │       └── pituitary/
│   │   └── binary_data/            # Binary classification dataset (NOT in GitHub)
│   │       ├── test/
│   │       ├── train/
│   │       └── valid/
│   │
│   └── lung_cancer/                # Lung cancer detection
│       ├── lung_cancer_cnn.keras   # Trained model
│       ├── train_lung_model.py     # Training script
│       ├── test_image_loader.py    # Utility for loading test images
│       ├── Data/                   # CT scan dataset (NOT in GitHub - see note below)
│       │   ├── test/
│       │   ├── train/
│       │   └── valid/
│       └── binary_data/            # Binary classification dataset (NOT in GitHub)
│           ├── test/
│           ├── train/
│           └── valid/
│
├── utils/                          # Utility functions
│   ├── brain_predict.py            # Brain tumor prediction utilities
│   ├── lung_predict.py             # Lung cancer prediction utilities
│   ├── predict.py                  # Heart disease prediction utilities
│   └── preprocess.py               # Data preprocessing functions
│
├── templates/                      # HTML templates
│   ├── index.html                  # Home page
│   ├── dashboard.html              # Main dashboard
│   ├── disease1.html               # Brain tumor detection page
│   ├── disease2.html               # Lung cancer detection page
│   ├── disease3.html               # Heart disease prediction page
│   └── result.html                 # Results display page
│
└── static/                         # Static files
    ├── css/
    │   └── style.css               # Application styling
    ├── js/
    │   └── script.js               # Client-side scripts
    └── uploads/                    # Temporary uploaded file storage
```

---

## ⚠️ Important Note: Medical Image Data

**The CT scan and MRI image files are NOT included in this GitHub repository** due to:
- **File Size Constraints**: Medical imaging datasets are extremely large (several GB)
- **Data Privacy**: Medical data requires proper handling and compliance

### What's Included in GitHub:
✅ All Python source code  
✅ Trained model files (.h5, .keras)  
✅ Complete folder structure  
✅ Training scripts  
✅ Configuration files  

### What's NOT Included:
❌ MRI scan images (brain_tumor/Data/ and brain_tumor/binary_data/)  
❌ CT scan images (lung_cancer/Data/ and lung_cancer/binary_data/)  
❌ Original medical imaging datasets  

### To Use This Project:
1. Clone the repository
2. The empty folder structure will be preserved with `.gitkeep` files
3. Download or obtain the medical imaging datasets separately
4. Place them in the respective `Data/` directories
5. Run the training scripts if you want to retrain the models

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/PranavSana/Healthcare-Management-System.git
cd HMS
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

---

## 📊 Models Used

### Brain Tumor Classification
- **Model Type**: Convolutional Neural Network (CNN)
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Input**: MRI scan images
- **Framework**: TensorFlow/Keras

### Lung Cancer Classification
- **Model Type**: Convolutional Neural Network (CNN)
- **Classes**: Adenocarcinoma, Large Cell Carcinoma, Normal, Squamous Cell Carcinoma
- **Input**: CT scan images
- **Framework**: TensorFlow/Keras

### Heart Disease Prediction
- **Model Type**: Machine Learning Classifier
- **Input**: Patient health metrics (age, cholesterol, blood pressure, etc.)
- **Output**: Risk score for heart disease
- **Framework**: scikit-learn

---

## 📋 Usage

1. **Navigate to the dashboard**
2. **Select the disease type** you want to check:
   - Brain Tumor Detection
   - Lung Cancer Detection
   - Heart Disease Prediction
3. **Upload medical data**:
   - For brain/lung: Upload MRI or CT scan image
   - For heart: Enter patient health metrics
4. **View Results**: Get predictions with confidence scores and clinical insights

---

## 📦 Dependencies

See `requirements.txt` for complete list. Key packages:
- Flask - Web framework
- TensorFlow/Keras - Deep learning
- NumPy - Numerical computing
- Pandas - Data manipulation
- scikit-learn - Machine learning
- Pillow - Image processing
- OpenCV - Computer vision

---

## 📝 Files in GitHub Repository

### Core Application
- `app.py` - Main Flask application
- `requirements.txt` - Python package dependencies

### Models & Training
- `models/train_heart_model.py` - Heart disease model training
- `models/brain_tumor/train_brain_model.py` - Brain tumor model training
- `models/brain_tumor/predict_brain.py` - Brain tumor inference
- `models/lung_cancer/train_lung_model.py` - Lung cancer model training
- `models/heart.csv` - Heart disease dataset

### Utilities
- `utils/brain_predict.py` - Brain prediction functions
- `utils/lung_predict.py` - Lung prediction functions
- `utils/predict.py` - Heart disease prediction functions
- `utils/preprocess.py` - Preprocessing utilities

### Frontend
- `templates/index.html` - Home page
- `templates/dashboard.html` - Dashboard
- `templates/disease1.html` - Brain tumor page
- `templates/disease2.html` - Lung cancer page
- `templates/disease3.html` - Heart disease page
- `templates/result.html` - Results display
- `static/css/style.css` - Styling
- `static/js/script.js` - JavaScript functionality

---

## 🔒 License

This project is provided as-is for educational and research purposes.

---

## 👨‍💻 Author

Pranav Sana

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

---

## ⚡ Performance Notes

- Brain Tumor Detection: ~2-5 seconds per scan
- Lung Cancer Detection: ~2-5 seconds per scan  
- Heart Disease Prediction: <1 second

---

## 🔧 Troubleshooting

**Models not loading?**
- Ensure TensorFlow/Keras is properly installed
- Check model file paths in the code

**Images not uploading?**
- Verify the `static/uploads/` directory exists
- Check file size limits in Flask configuration

**Import errors?**
- Run `pip install -r requirements.txt` again
- Ensure you're using the correct Python virtual environment

---

## 📞 Support

For issues or questions, please check the code comments or create an issue on GitHub.
