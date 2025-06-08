import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Input
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

# ----------- SET PATHS -----------
train_data_dir = "PATH_TO_YOUR_TRAIN_DATA"
dev_data_dir = "PATH_TO_YOUR_DEV_DATA"

# ----------- IMAGE DATA GENERATOR -----------
img_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

dev_generator = datagen.flow_from_directory(
    dev_data_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# ----------- MODEL -----------
model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0),
    metrics=['accuracy']
)

# ----------- CALLBACKS -----------
checkpoint_cb = ModelCheckpoint("best_model_doubleconv.h5", save_best_only=True)
earlystop_cb = EarlyStopping(patience=5, restore_best_weights=True)

# ----------- TRAINING -----------
history = model.fit(
    train_generator,
    validation_data=dev_generator,
    epochs=50,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# ----------- CONVERT TO TFLITE (FLOAT16) -----------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("best_model_doubleconv.tflite", "wb") as f:
    f.write(tflite_model)

# ----------- PLOT ACCURACY & LOSS -----------
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
