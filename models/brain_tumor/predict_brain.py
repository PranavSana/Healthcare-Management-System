import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "models/brain_tumor/brain_tumor_cnn.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")

# Change this image path to test
IMAGE_PATH = "models/brain_tumor/Data/Testing/pituitary/"

# Pick one image automatically
import os
image_file = os.listdir(IMAGE_PATH)[0]
image_path = os.path.join(IMAGE_PATH, image_file)

# Load image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)[0][0]

if prediction > 0.5:
    result = "BRAIN TUMOR DETECTED"
else:
    result = "NO BRAIN TUMOR DETECTED"

print("Prediction:", prediction)
print("Result:", result)

# Display image
plt.imshow(img[0])
plt.title(result)
plt.axis("off")
plt.show()
