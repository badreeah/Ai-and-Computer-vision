# Ai-and-Computer-vision
# Machine Learning Projects: Census Income Prediction & Brain Tumor Detection

## Overview

This repository showcases two machine learning projects I developed using **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)** algorithms. These projects focus on solving real-world classification problems using supervised learning approaches.

---

## 1. Census Income Prediction

**Objective:**  
Predict whether an individual's annual income exceeds $50K based on demographic and employment attributes from the UCI Adult dataset.

**Techniques Used:**  
- Preprocessing: Handling missing values, label encoding, feature scaling  
- Models: K-Nearest Neighbors (KNN), Support Vector Machine (SVM)  
- Evaluation: Confusion Matrix, Accuracy, Precision, Recall, F1-score

**Dataset:**  
[UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

**Results:**  
- SVM outperformed KNN in terms of accuracy and generalization.  
- Feature scaling significantly improved KNN performance.

---

## 2. MRI Brain Tumor Detection

**Objective:**  
Classify MRI images to detect the presence of a brain tumor.

**Techniques Used:**  
- Image preprocessing: Resizing, normalization, grayscale conversion  
- Feature Extraction: Histogram of Oriented Gradients (HOG) , Local Binary Patterns (LBP)
- Models: Support Vector Machine (SVM), Random forest , Ensemble Model (VotingClassifier) 
- Evaluation: Accuracy, Precision, Recall, F1-score

**Dataset:**  
[MRI Brain Tumor Dataset (Kaggle)](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

**Results:**  
-SVM: Based on accuracy and classification report, the SVM model is effective but limited by its linear nature.

-Random Forest: Strong performance but may be computationally demanding.

-Ensemble: Significantly improves performance across all metrics, benefiting from both the precision of SVM and the flexibility of Random Forest.
---

## Tools & Libraries

- Python  
- Scikit-learn  
- NumPy, Pandas  
- OpenCV, Matplotlib  
- Jupyter Notebook

---

## Author

**Badreeah homoud almutairi**  
Computer Science Student  
📧 443011783@sm.imamu.edu.sa 
---

## License

This project is licensed under the MIT License. Feel free to use, modify, and share it.
