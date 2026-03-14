# 🐱🐶 Cat vs Dog — CNN Image Classifier

A binary image classifier that distinguishes cats from dogs using a Convolutional Neural Network built with TensorFlow / Keras.

## What It Does

Trains a two-layer CNN on a labelled dataset of cat and dog images, then predicts the class of unseen images. Data augmentation (shear, zoom, horizontal flip) is applied during training to improve generalisation.

### Architecture

```
Input (64 × 64 × 3)
  → Conv2D 32 filters (3×3) + ReLU → MaxPool (2×2)
  → Conv2D 32 filters (3×3) + ReLU → MaxPool (2×2)
  → Flatten
  → Dense 128 (ReLU)
  → Dense 1 (Sigmoid)
```

Compiled with Adam optimiser and binary cross-entropy loss.

## Dataset

The model expects the following directory layout inside `Convolutional_Neural_Networks/dataset/`:

```
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
├── test_set/
│   ├── cats/
│   └── dogs/
└── single_prediction/
    └── cat_or_dog_1.jpg
```

A `single_prediction.zip` is included in the repo. Training and test sets must be downloaded separately (see original links below or use any cats-vs-dogs dataset such as the [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)).

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Language |
| 🔶 TensorFlow / Keras | Deep-learning framework |
| 🔢 NumPy | Array operations |

## Getting Started

```bash
# 1. Clone
git clone https://github.com/stabgan/Convolutional-Neural-Network-using-Keras.git
cd Convolutional-Neural-Network-using-Keras

# 2. Install dependencies
pip install tensorflow numpy

# 3. Place dataset (see layout above)

# 4. Train & predict
python Convolutional_Neural_Networks/cnn_new.py
```

## ⚠️ Known Issues

- `steps_per_epoch` and `validation_steps` are set for a ~8 000 / ~2 000 image split. Adjust if your dataset differs.
- `ImageDataGenerator` is a legacy API; for new projects prefer `keras.utils.image_dataset_from_directory` with `tf.data` pipelines.

## License

See [LICENSE](LICENSE).
