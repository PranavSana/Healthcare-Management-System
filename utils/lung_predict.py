import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "models/lung_cancer/lung_cancer_cnn.keras"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

CLASS_LABELS = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]

DISPLAY_NAMES = {
    "adenocarcinoma":       "Adenocarcinoma",
    "large.cell.carcinoma": "Large Cell Carcinoma",
    "normal":               "Normal",
    "squamous.cell.carcinoma": "Squamous Cell Carcinoma"
}

def predict_lung_cancer(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)[0]

    class_index = np.argmax(predictions)
    confidence  = predictions[class_index]
    label       = CLASS_LABELS[class_index]

    # All class probabilities as a dict  {display_name: percentage_string}
    all_probs = {
        DISPLAY_NAMES[cls]: f"{float(predictions[i]) * 100:.1f}"
        for i, cls in enumerate(CLASS_LABELS)
    }

    return {
        "status":     "Lung Cancer Detected" if label != "normal" else "No Lung Cancer",
        "type":       DISPLAY_NAMES[label],
        "confidence": f"{confidence:.2f}",
        "all_probs":  all_probs
    }