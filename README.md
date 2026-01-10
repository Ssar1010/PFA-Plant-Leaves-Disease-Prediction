# 🌿 Plant Disease Classification Using VGG16 Transfer Learning

## Project Overview
In this project, I developed a deep learning–based image classification system for plant disease and condition recognition using **transfer learning with the VGG16 convolutional neural network**. The objective was to leverage pretrained ImageNet features and adapt them efficiently to a domain-specific plant image dataset.

The work emphasizes **methodological rigor, reproducibility, and proper evaluation**, aligning with academic and professional machine learning standards.

---

## Objectives
- Design a robust image classification pipeline for plant datasets  
- Apply transfer learning using a pretrained convolutional neural network  
- Implement deterministic dataset splitting and data augmentation  
- Perform controlled fine-tuning of high-level convolutional layers  
- Evaluate model performance using standard classification metrics  

---

## Dataset
- **Source**: Open-source plant image dataset available on **Kaggle**  
- **License**: Publicly accessible for research and educational use  
- **Selection strategy**:  
  From the original dataset, I deliberately selected **six classes** relevant to my study objectives. This subset was chosen to ensure balanced experimentation, manageable computational complexity, and focused evaluation.

- **Organization**: One directory per class  
- **Splitting strategy**:
  - Training set: 70%
  - Validation set: 20%
  - Test set: 10%

Dataset splitting is performed programmatically using a fixed random seed to ensure **reproducibility and experimental consistency**.

---

## Methodology

### Data Preprocessing
- Images resized to **224 × 224 pixels**, matching VGG16 input requirements  
- Normalization using the ImageNet preprocessing function  
- Data augmentation applied exclusively to the training set:
  - Random rotations  
  - Width and height shifts  
  - Zoom transformations  
  - Horizontal flipping  

This approach improves generalization and reduces overfitting.

---

### Model Architecture
- **Base Model**: VGG16 pretrained on ImageNet  
- Fully connected classification layers removed  
- Custom classifier head composed of:
  - Flatten layer  
  - Dense layer with 256 neurons (ReLU activation)  
  - Dropout layer (rate = 0.5)  
  - Softmax output layer  

All convolutional layers are initially frozen to preserve pretrained representations.

---

### Training Strategy

#### Phase 1: Feature Extraction
- VGG16 backbone frozen  
- Optimizer: Adam  
- Learning rate: 1e-4  
- Loss function: Categorical Cross-Entropy  

#### Phase 2: Fine-Tuning
- Only the final convolutional block (Block 5) is unfrozen  
- Learning rate reduced to 1e-5  
- Purpose: refine high-level semantic features while maintaining training stability  

Early stopping and model checkpointing are employed to prevent overfitting and retain the best-performing weights.

---

## Evaluation
Model performance is evaluated using:
- Training and validation accuracy curves  
- Training and validation loss curves  
- Final test set accuracy and loss  
- Confusion matrix  
- Classification report including precision, recall, and F1-score per class  

All evaluation artifacts are saved for traceability and analysis.

---

## Inference
The trained model supports **single-image inference**. Input images are preprocessed using the same pipeline applied during training to ensure consistency.

---

## Outputs
- Trained model weights  
- Fine-tuned model weights  
- Accuracy and loss plots  
- Confusion matrix visualization  
- Classification report (text file)  

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
- Fixed random seeds for Python, NumPy, and TensorFlow  
- Deterministic dataset splitting  
- Explicit preprocessing and training configuration  
- Saved model weights and evaluation metrics  

---

## Conclusion
This project demonstrates a principled and reproducible application of transfer learning for plant disease classification using an **open-source Kaggle dataset** and a **carefully selected six-class subset**. By combining structured data preprocessing, controlled fine-tuning, and comprehensive evaluation, the system achieves reliable performance and provides a solid foundation for extension to larger datasets or additional plant categories.

---

## References
1. Simonyan, K., Zisserman, A. *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556  
2. Chollet, F. *Deep Learning with Python*. Manning Publications  
3. TensorFlow Keras Documentation — Transfer Learning and Fine-Tuning  
4. Goodfellow, I., Bengio, Y., Courville, A. *Deep Learning*. MIT Press
