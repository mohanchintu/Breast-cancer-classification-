

# Breast Cancer Classification and Clustering

This project is designed to classify and cluster breast cancer data using machine learning techniques, specifically t-SNE for dimensionality reduction, K-Means for clustering, and Support Vector Machines (SVM) for classification.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Clustering](#clustering)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Classification](#classification)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)

## Introduction
This project uses machine learning techniques to analyze breast cancer data, aiming to classify the diagnosis as either malignant (M) or benign (B). The workflow includes data preprocessing, clustering using K-Means, dimensionality reduction using PCA, and classification using an ensemble of SVM classifiers.

## Installation
To run this code, you will need Python 3.x and the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Data
The dataset used is assumed to be a CSV file named `data.csv`, containing features related to breast cancer diagnoses. The target variable is `diagnosis`, where `M` represents malignant and `B` represents benign.

## Preprocessing
The preprocessing steps include:
1. **Loading the data**: The data is loaded from a CSV file using Pandas.
2. **Mapping the target variable**: The `diagnosis` column is mapped to numerical values (`M` to `1` and `B` to `0`).
3. **Scaling the features**: Standardization is applied to the features for better performance of machine learning algorithms.

## Clustering
t-SNE is used for dimensionality reduction and visualization of clusters:
- **K-Means Clustering**: Clustering is performed on the data using K-Means with `k=2`, representing two clusters for the diagnosis types.

## Dimensionality Reduction
Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset:
- **PCA**: The data is reduced to retain 95% of the variance, significantly reducing the feature space while preserving most of the data's information.

## Classification
Two Support Vector Machines (SVM) with different kernels (`linear` and `rbf`) are used in an ensemble method:
- **Voting Classifier**: Combines the outputs of the linear and RBF SVMs to make final predictions.

## Evaluation
The model's performance is evaluated using accuracy, confusion matrix, sensitivity (recall), and specificity:
- **Confusion Matrix**: Provides insights into the number of true positives, true negatives, false positives, and false negatives.
- **Specificity**: Calculated from the confusion matrix, representing the proportion of actual negatives that are correctly identified.

## Conclusion
This project demonstrates the application of various machine learning techniques to classify and cluster breast cancer data effectively. The ensemble of SVM classifiers achieves high accuracy and balanced sensitivity and specificity.

## Example Outputs
- **Accuracy**: The model's accuracy on the test data.
- **Confusion Matrix**: 
    ```
    [[TN FP]
     [FN TP]]
    ```
- **Sensitivity/Recall**: Sensitivity of the classifier.
- **F1 Score**: The F1 score of the classifier.
- **Specificity**: Specificity of the classifier.
