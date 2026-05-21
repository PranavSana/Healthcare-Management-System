import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# -----------------------
# PARAMETERS
# -----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

train_dir = "models/lung_cancer/Data/train"
valid_dir = "models/lung_cancer/Data/valid"
test_dir  = "models/lung_cancer/Data/test"

# -----------------------
# DATA GENERATORS
# -----------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("\nClass Indices:", train_generator.class_indices)

# -----------------------
# LOAD MOBILENET
# -----------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# -----------------------
# CUSTOM HEAD
# -----------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------
# COMPILE
# -----------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------
# TRAIN
# -----------------------
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS
)

# -----------------------
# TEST
# -----------------------
test_loss, test_acc = model.evaluate(test_generator)
print("\nTest Accuracy:", test_acc)

# -----------------------
# SAVE MODEL
# -----------------------
model.save("models/lung_cancer/lung_cancer_cnn.keras")

print("Model saved successfully")