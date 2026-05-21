import os
import cv2
import matplotlib.pyplot as plt

# Change this if needed
BASE_DIR = "models/brain_tumor/binary_data/train"

# Pick one sample image
tumor_dir = os.path.join(BASE_DIR, "tumor")
sample_image = os.listdir(tumor_dir)[0]
image_path = os.path.join(tumor_dir, sample_image)

# Load image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize
img = cv2.resize(img, (224, 224))

print("Image loaded successfully")
print("Image shape:", img.shape)

# Display image
plt.imshow(img)
plt.title("Sample Brain MRI - Tumor")
plt.axis("off")
plt.show()
