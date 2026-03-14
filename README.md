# 🐱🐶 Cats vs Dogs — CNN Image Classifier

A Convolutional Neural Network built with Keras (TensorFlow backend) that classifies images of cats and dogs. Trained on a large dataset of labeled pet images using data augmentation and binary cross-entropy loss.

---

## 🏗️ Architecture

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3 kernel, ReLU, input 64×64×3 |
| MaxPooling2D | 2×2 pool |
| Conv2D | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 pool |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dense (output) | 1 unit, Sigmoid (binary) |

Optimizer: Adam · Loss: Binary Cross-Entropy · Metric: Accuracy

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🐍 Python 3.8+ | Language |
| 🔶 TensorFlow / Keras | Deep learning framework |
| 🔢 NumPy | Array operations |

---

## 📦 Dependencies

```
tensorflow>=2.0
numpy
```

Install with:

```bash
pip install tensorflow numpy
```

---

## 🚀 How to Run

1. **Download the dataset** and place it under `Convolutional_Neural_Networks/dataset/`:
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

2. **Train & predict**:
   ```bash
   cd Convolutional_Neural_Networks
   python cnn_new.py
   ```

   The script trains for 25 epochs and then predicts on a single test image.

---

## ⚠️ Known Issues

- The dataset is not included in this repo — you need to supply your own cats/dogs image set.
- `steps_per_epoch` and `validation_steps` are hardcoded; adjust them to match your dataset size.
- Training on CPU can be very slow; a GPU with CUDA support is recommended.
- The model uses a simple two-conv-layer architecture — accuracy can be improved with deeper networks, transfer learning, or larger input sizes.

---

## 📄 License

See [LICENSE](LICENSE) for details.
