# Hybrid-Model-For-Cost-Effective-SCA-Detection

## Abstract

Sickle Cell Anemia (SCA) is an inherited blood disorder characterized by the production of abnormal hemoglobin, which causes red blood cells to become rigid and sickle-shaped. Early identification and screening are important for the timely management of the disease and prevention of associated complications.

This project explores the application of deep learning and computer vision techniques for the automated classification of microscopic blood smear images to identify sickle cell-related abnormalities. Multiple individual deep learning architectures and hybrid model approaches are investigated, evaluated, and compared to identify a suitable model for the proposed classification task.

The project also incorporates image preprocessing techniques to improve input quality and Grad-CAM-based visual explanations to understand the regions influencing model predictions. An interactive application is developed to provide a user-friendly interface for image submission, classification, and visualization of model explanations.

The repository contains the complete project workflow, including dataset preparation, preprocessing experiments, individual model implementations, hybrid architectures, model evaluation, explainability analysis, and the final application.

---

## 1. Project Overview

The primary objective of this project is to investigate the use of deep learning-based image classification techniques for Sickle Cell Anemia screening using microscopic blood smear images.

Rather than relying on a single architecture, the project explores multiple approaches to understand their performance, limitations, and suitability for the classification task.

The major components of the project include:

- Dataset collection, exploration, and preparation.
- Image cleaning and duplicate detection.
- Image preprocessing and enhancement.
- Implementation and evaluation of individual deep learning models.
- Development and experimentation with hybrid AI architectures.
- Comparative analysis of different model configurations.
- Selection and evaluation of the final model.
- Model interpretability using Grad-CAM.
- Development of an interactive application for image-based prediction.

## 2. Objectives

The key objectives of this project are:

1. To investigate deep learning approaches for classifying microscopic blood smear images.
2. To prepare and preprocess the image dataset for model development.
3. To experiment with multiple individual deep learning architectures.
4. To explore hybrid model architectures and their potential advantages.
5. To evaluate and compare different models using appropriate classification metrics.
6. To identify a suitable model based on experimental evaluation.
7. To incorporate explainable AI techniques for visual interpretation of model predictions.
8. To develop an interactive application demonstrating the proposed approach.

## 3. Project Methodology

The overall development process follows a series of stages, beginning with dataset preparation and progressing towards model development, evaluation, and application deployment.

### 3.1 Dataset Collection and Exploration

The project uses microscopic blood smear images for the classification task.

The dataset undergoes initial exploration to understand its structure, image formats, class distribution, and other characteristics relevant to model development.

### 3.2 Data Cleaning and Validation

Data preparation includes:

- Checking image readability and validity.
- Identifying duplicate images.
- Removing any possible noise that is present (blurry photos, dim lighting, etc.)
- Performing SHA-256-based file hashing.
- Performing pixel-level duplicate checks.
- Examining image formats and dimensions.
- Identifying unsuitable or corrupted images.
- Preparing the dataset for subsequent experiments.

### 3.3 Image Preprocessing

Image preprocessing techniques are explored to prepare the input images for deep learning models.

The preprocessing workflow includes techniques such as:

- Image loading and resizing.
- Image normalization.
- Histogram equalization.
- Contrast enhancement.
- Model-specific input preprocessing.

The exact preprocessing configuration may differ between model experiments depending on the architecture and its input requirements.

### 3.4 Individual Model Development

Multiple individual deep learning architectures are implemented and evaluated independently.

These experiments help establish baseline performance and provide a basis for comparison with the hybrid approaches.

### 3.5 Hybrid Model Development

Hybrid architectures are explored by combining different deep learning approaches or feature representations.

The objective is to investigate whether combining complementary model capabilities can improve classification performance.

### 3.6 Model Evaluation and Comparison

The developed models are evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- ROC-AUC, where applicable

Confusion matrices and training/validation performance curves are also used to examine model behaviour.

### 3.7 Explainable AI Using Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is incorporated to provide visual explanations of model predictions.

Grad-CAM generates heatmaps that highlight image regions contributing to a model's prediction, helping examine whether the model focuses on relevant visual features.

### 3.8 Application Development

An interactive application is developed to demonstrate the final model.

The application provides functionality for image input, prediction, and visualization of Grad-CAM explanations.

---

## 4. Models Explored

The project investigates multiple model configurations throughout the experimentation process.

The models are organized into the following categories:

| Category | Description |
|---|---|
| Individual Models | Standalone deep learning architectures evaluated independently |
| Hybrid Models | Architectures combining multiple models or feature representations |
| Final Model | Selected model used in the final application |

Individual and hybrid model experiments are maintained separately to preserve the development history and enable comparison between different approaches.

Detailed architecture descriptions, training configurations, and evaluation results will be documented in the respective experiment records.

---

## 5. Dataset

The project uses microscopic blood smear images for image-based classification.

Dataset preparation involves image validation, duplicate detection, preprocessing, and dataset splitting.

### Dataset Information

| Attribute | Description |
|---|---|
| Data Type | Microscopic blood smear images |
| Input Format | Image files |
| Task | Image classification |
| Classes | negative, positive_labelled, positive_unlabelled  |
| Original Dataset Size | 1,052 images |
| Dataset Source |  |

**Note:** Different experiments may use different numbers of images depending on image format compatibility, preprocessing requirements, and data validation procedures. These differences are documented in the respective experiments.

The dataset is not redistributed through this repository unless its licensing and sharing conditions permit it. Instructions for obtaining the dataset will be provided where applicable.

---

## 6. Technologies and Tools

The project uses the following technologies and tools:

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| TensorFlow / Keras | Deep learning model development |
| NumPy | Numerical operations |
| Pandas | Data manipulation and analysis |
| OpenCV | Image processing |
| Matplotlib | Data visualization |
| Scikit-learn | Dataset splitting and evaluation metrics |
| Grad-CAM | Model explainability |
| Streamlit | Interactive application development |
| Jupyter Notebook | Experimentation and model development |
| Git and GitHub | Version control and project management |

The exact libraries and versions required to run the project are specified in `requirements.txt`.

---


## 7. Models Developed and Evaluated

A total of five individual deep learning models and two intermediate hybrid fusion configurations were implemented and evaluated during the project. Based on these experiments, a final dual-branch hybrid architecture was also developed and evaluated.

### 7.1 Individual Deep Learning Models

The following five individual architectures were investigated:

| Model | Test Images | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| VGG-16 | 95 | 86.3% | 95.0% | 85.1% | 89.8% | 0.943 |
| DenseNet-121 | 95 | 90.5% | 92.7% | 94.0% | 93.3% | 0.953 |
| InceptionV3 | 91 | 96.7% | 95.5% | 100.0% | 97.7% | 1.000 |
| EfficientNet-B0 | 158 | 96.2% | 96.3% | 99.2% | 97.7% | 0.973 |
| OptiSCDNet-inspired MobileNetV2 variant | 158 | 93.0% | 97.6% | 93.9% | 95.7% | 0.960 |

### 7.2 Intermediate Hybrid Model Experiments

Two intermediate feature-fusion configurations were developed to investigate the effectiveness of combining complementary feature representations.

| Hybrid Model | Test Images | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| DenseNet-121 + VGG-16 | 106 | 88.7% | 94.1% | 88.9% | 91.4% | 0.943 |
| OptiSCDNet-inspired CNN + EfficientNet-B0 | 149 | 85.2% | 99.1% | 83.5% | 90.6% | 0.971 |

### 7.3 Proposed Final Hybrid Architecture

The final proposed architecture uses a dual-branch feature extraction approach combining:

- **Branch 1:** ImageNet-pretrained EfficientNet-B0.
- **Branch 2:** A custom residual CNN inspired by OptiSCDNet, incorporating separable convolutions and squeeze-and-excitation channel attention.

The feature representations from both branches are concatenated and passed through a classification head to produce the final binary classification output.

| Metric | Result |
|---|---:|
| Test Images | 156 |
| Accuracy | 85.3% |
| Precision | 95.7% |
| Recall (Sensitivity) | 86.2% |
| Specificity | 80.8% |
| F1-Score | 90.7% |
| ROC-AUC | 0.922 |
| PR-AUC | 0.979 |
| Total Parameters | 4.96 Million |

The final model uses a 0.5 classification threshold and was evaluated on a held-out test set containing 26 normal and 130 sickle images.

### 7.4 Experimental Observations

The experiments provided insights into the performance of different architectures and fusion strategies.

- DenseNet-121 improved upon VGG-16 on their shared test set, achieving higher accuracy, recall, and F1-score.
- EfficientNet-B0 achieved 96.2% accuracy on its 158-image test set.
- InceptionV3 achieved 96.7% accuracy; however, its test set contained positive samples exclusively from the annotated folder, making the result susceptible to annotation-related shortcut learning.
- The intermediate fusion experiments did not demonstrate a consistent improvement over their individual components.
- The proposed final hybrid achieved 85.3% accuracy and a ROC-AUC of 0.922, without demonstrating a measurable performance improvement over the strongest individual backbones.

**Note:** The models were not all evaluated on identical test sets. Therefore, the reported results should be interpreted as experiment-specific performance rather than a strict ranking of all architectures. Comparisons between models sharing the same test set are more directly meaningful.

---

## 8. Explainability and Visualization

Model interpretability is an important component of this project.

Grad-CAM is used to generate heatmaps that highlight regions of the input image that contribute to the model's prediction.

These visualizations provide additional insight into the model's behaviour and help examine its attention patterns.

The explainability implementation and related visualizations are maintained separately from the core model implementation.

---

## 9. Limitations

The project has certain limitations that should be considered when interpreting its results:

- Model performance depends on the quality, size, and diversity of the available dataset.
- Variations in image acquisition, staining, illumination, and image quality may influence predictions.
- Different architectures may require different preprocessing configurations.
- Image-level dataset splitting does not necessarily eliminate patient-level or slide-level data leakage.
- Grad-CAM visualizations provide explanations of model behaviour but do not establish clinical validity.
- The developed system is intended for research and educational purposes and is not a substitute for established clinical diagnostic procedures.

Further validation using larger, more diverse, and independently collected datasets would be necessary before considering any clinical application.

---

## 10. Future Scope

Potential future improvements include:

- Evaluation using larger and more diverse datasets.
- Exploration of additional deep learning and hybrid architectures.
- Improved generalization across different imaging conditions.
- More extensive explainability and model interpretation.
- Evaluation on independently collected test datasets.
- Optimization of inference speed and application performance.
- Further investigation of the model's suitability for screening applications.

---

## 11. Disclaimer

This project is developed for academic research and educational purposes.

The developed models are experimental and are not intended to replace professional medical evaluation, laboratory investigations, or confirmatory diagnostic tests.

Predictions generated by the application should not be interpreted as a definitive diagnosis of Sickle Cell Anemia.

---

## 12. Authors

**Project Team**

- Bhavana Mudadla
- Varshitha Kurra
- Amrita Chakravorty
- Lakshya Gupta

**Project Domain:** Deep Learning, Computer Vision, Medical Image Analysis
