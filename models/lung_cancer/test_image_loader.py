import os
import cv2
import matplotlib.pyplot as plt

# Base path to training data
BASE_DIR = "models/lung_cancer/Data/train"

# Pick one class to test (normal)
class_name = "normal"
class_path = os.path.join(BASE_DIR, class_name)

# Get first image in that folder
image_name = os.listdir(class_path)[0]
image_path = os.path.join(class_path, image_name)

# Load image using OpenCV
image = cv2.imread(image_path)

# Convert from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image to standard CNN input size
image = cv2.resize(image, (224, 224))

# Display image
plt.imshow(image)
plt.title(f"Sample CT Image - {class_name}")
plt.axis("off")
plt.show()

print("Image loaded successfully")
print("Image shape:", image.shape)
