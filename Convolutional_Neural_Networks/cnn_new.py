"""Convolutional Neural Network for Cat vs Dog Image Classification.

A binary image classifier built with TensorFlow/Keras that distinguishes
between cat and dog images using a two-layer CNN architecture with data
augmentation.

Requirements:
    pip install tensorflow numpy
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def build_model() -> Sequential:
    """Build and compile a binary-classification CNN."""
    model = Sequential([
        keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_data_generators():
    """Create augmented training and rescaled test data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    training_set = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "training_set"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    test_set = test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "test_set"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    return training_set, test_set


def train(model: Sequential, training_set, test_set) -> None:
    """Train the CNN on the image dataset."""
    model.fit(
        training_set,
        steps_per_epoch=8000 // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_set,
        validation_steps=2000 // BATCH_SIZE,
    )


def predict_single_image(model: Sequential, training_set) -> str:
    """Run inference on a single test image and return 'cat' or 'dog'."""
    img_path = os.path.join(DATASET_DIR, "single_prediction", "cat_or_dog_1.jpg")
    test_image = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    test_image = keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # rescale to match training

    result = model.predict(test_image)
    class_indices = training_set.class_indices
    # Invert the mapping: {index: class_name}
    idx_to_class = {v: k for k, v in class_indices.items()}

    predicted_idx = int(result[0][0] > 0.5)
    prediction = idx_to_class.get(predicted_idx, "unknown")
    print(f"Prediction: {prediction}")
    return prediction


if __name__ == "__main__":
    classifier = build_model()
    classifier.summary()

    training_set, test_set = get_data_generators()
    train(classifier, training_set, test_set)

    prediction = predict_single_image(classifier, training_set)
    print(f"The image is a: {prediction}")
