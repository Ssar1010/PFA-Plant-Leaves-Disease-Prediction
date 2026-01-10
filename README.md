# 🌿 Plant Disease Classification using VGG16 Transfer Learning

## Overview
This project implements a deep learning–based image classification system for plant disease and condition recognition using **transfer learning with VGG16**. The model leverages pretrained ImageNet weights and is adapted to a domain-specific plant image dataset through controlled fine-tuning.

The pipeline emphasizes **reproducibility, structured experimentation, and rigorous evaluation**, making it suitable for academic use and professional portfolios.

---

## Objectives
- Build a robust image classification pipeline for plant datasets
- Apply transfer learning using a pretrained convolutional neural network
- Perform deterministic dataset splitting and data augmentation
- Fine-tune high-level convolutional features
- Evaluate performance using standard classification metrics

---

## Dataset
- **Source**: Plant image dataset hosted on Kaggle
- **Structure**: One directory per class
- **Split ratios**:
  - Training: 70%
  - Validation: 20%
  - Test: 10%

Dataset splitting is performed programmatically using a fixed random seed to ensure reproducibility.

---

## Methodology

### Data Preprocessing
- Images resized to **224 × 224** pixels
- Normalization using `preprocess_input` (ImageNet standard)
- Data augmentation applied only to the training set:
  - Random rotations
  - Width and height shifts
  - Zoom transformations
  - Horizontal flipping

---

### Model Architecture
- **Base Model**: VGG16 (pretrained on ImageNet)
- Top classification layers removed
- Custom classifier head:
  - Flatten layer
  - Dense layer (256 units, ReLU)
  - Dropout (0.5)
  - Softmax output layer

All convolutional layers are initially frozen.

---

### Training Strategy

#### Phase 1: Feature Extraction
- Frozen VGG16 backbone
- Optimizer: Adam
- Learning rate: `1e-4`
- Loss function: Categorical Cross-Entropy

#### Phase 2: Fine-Tuning
- Only **Block 5** of VGG16 is unfrozen
- Learning rate reduced to `1e-5`
- Objective: refine high-level features while maintaining training stability

---

## Evaluation
Model performance is assessed using:
- Training and validation accuracy curves
- Training and validation loss curves
- Test set accuracy and loss
- Confusion matrix
- Classification report (precision, recall, F1-score)

All evaluation artifacts are saved to the working directory.

---

## Inference
The trained model supports single-image inference.  
Input images are preprocessed using the same pipeline as the training data before prediction.

---

## Outputs
- Trained model weights
- Fine-tuned model weights
- Accuracy and loss plots
- Confusion matrix visualization
- Classification report (`.txt`)

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- VGG16 (ImageNet pretrained)

---

## Reproducibility
- Fixed random seed
- Deterministic dataset splitting
- Explicit preprocessing and training configuration
- Saved weights and evaluation metrics

---

## Conclusion
This project demonstrates a principled application of transfer learning for plant disease classification. The methodology combines structured preprocessing, controlled fine-tuning, and comprehensive evaluation, resulting in a reliable and extensible image classification system.

---

## References
1. Simonyan, K., Zisserman, A. *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556  
2. Chollet, F. *Deep Learning with Python*. Manning Publications  
3. TensorFlow Keras Documentation — Transfer Learning and Fine-Tuning  
4. Goodfellow, I., Bengio, Y., Courville, A. *Deep Learning*. MIT Press
