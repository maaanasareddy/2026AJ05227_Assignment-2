# Breast Cancer Classification - ML Assignment 2

## a. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on features computed from digitized images of fine needle aspirate (FNA) of breast masses. These features describe characteristics of the cell nuclei present in the image.

This is a **binary classification** problem where the goal is to accurately classify tumors to assist in medical diagnosis.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset

| Property | Value |
|---|---|
| **Source** | UCI Machine Learning Repository / Kaggle |
| **Instances** | 569 |
| **Features** | 30 numeric features |
| **Target Variable** | `diagnosis` (M = Malignant, B = Benign) |
| **Class Distribution** | Benign: 357 (62.7%), Malignant: 212 (37.3%) |
| **Missing Values** | None |

**Features:** Ten real-valued features are computed for each cell nucleus, and for each feature the mean, standard error (SE), and "worst" (mean of the three largest values) are recorded, giving 30 features total:

- `radius` - mean of distances from center to points on the perimeter
- `texture` - standard deviation of gray-scale values
- `perimeter` - perimeter of the cell nucleus
- `area` - area of the cell nucleus
- `smoothness` - local variation in radius lengths
- `compactness` - (perimeter^2 / area - 1.0)
- `concavity` - severity of concave portions of the contour
- `concave points` - number of concave portions of the contour
- `symmetry` - symmetry of the cell nucleus
- `fractal dimension` - "coastline approximation" - 1

**Preprocessing:**
- Dropped `id` column (not a predictive feature)
- Encoded target: B = 0, M = 1
- Applied StandardScaler for feature normalization
- Train/Test split: 80/20 with stratified sampling (random_state=42)
- Training set: 455 samples, Test set: 114 samples

## c. GitHub Repository Link

**Repository:** https://github.com/maaanasareddy/2026AJ05227_Assignment-2

**Repository Structure:**
```
2026AJ05227_Assignment-2/
├── app.py                  # Streamlit web application
├── train_models.py         # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test data (114 samples)
└── model/                  # Saved trained models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── support_vector_machine.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── results.json
```

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | **0.9737** | **0.9947** | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Logistic Regression performs excellently with an accuracy of 96.49% and an AUC of 0.9960, which is nearly perfect. It shows a strong balance between precision (0.9750) and recall (0.9286), resulting in a high F1 score of 0.9512. The model benefits from the linearly separable nature of many of the features in this dataset and the feature scaling applied. It misclassified only 4 samples (1 false positive and 3 false negatives). This is a reliable and interpretable model for this classification task. |
| **Decision Tree** | The Decision Tree classifier has the second-lowest performance overall with 92.98% accuracy. It has equal precision and recall (0.9048) and an AUC of 0.9246, which is the lowest among all models. It made 8 misclassifications (4 false positives and 4 false negatives). Decision Trees tend to overfit on training data and may not generalize as well without pruning or ensemble methods. However, it offers good interpretability and requires no feature scaling. |
| **KNN** | KNN achieves 95.61% accuracy with very high precision (0.9744) but slightly lower recall (0.9048), suggesting it is conservative in predicting malignancy. The AUC of 0.9823 indicates strong discrimination ability. It misclassified 5 samples (1 false positive and 4 false negatives). KNN benefits significantly from the StandardScaler preprocessing, as it is a distance-based algorithm. The choice of K=5 provides a good balance between bias and variance. |
| **Naive Bayes** | Gaussian Naive Bayes has the lowest accuracy (92.11%) and F1 score (0.8889) among all models. Despite this, it achieves a surprisingly high AUC of 0.9891, indicating that its probability estimates are well-calibrated even though its hard classifications are not as accurate. It made 9 misclassifications (3 false positives and 6 false negatives). The lower recall (0.8571) means it misses more malignant cases, which is a concern in medical diagnosis. The assumption of feature independence may not hold perfectly for this dataset. |
| **Support Vector Machine** | SVM is the best-performing model with the highest accuracy (97.37%), F1 score (0.9630), and MCC (0.9442). Notably, it achieves **perfect precision (1.0000)** with zero false positives, while maintaining a strong recall of 0.9286. Its AUC of 0.9947 is near-perfect. The RBF kernel effectively captures the non-linear decision boundary in the scaled feature space. Only 3 samples were misclassified (all false negatives). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** is the clear winner for the Breast Cancer Wisconsin dataset. It achieves the highest scores across nearly all metrics - accuracy (97.37%), F1 (0.9630), MCC (0.9442), and perfect precision (1.0000). In a medical diagnosis context, having zero false positives (no benign tumors incorrectly labeled as malignant) is valuable, while its 3 false negatives are the joint-lowest among all models. Logistic Regression is a close second and offers better interpretability, making it a strong alternative for clinical use. |

## e. Live Streamlit App

**Streamlit App Link:** [To be updated after deployment]

### App Features:
1. **CSV Upload** - Upload test data in CSV format for evaluation
2. **Model Selection** - Dropdown to select any of the 5 trained models
3. **Evaluation Metrics** - Displays Accuracy, AUC, Precision, Recall, F1, and MCC
4. **Confusion Matrix** - Visual heatmap of the confusion matrix
5. **ROC Curve** - Receiver Operating Characteristic curve
6. **Classification Report** - Detailed per-class metrics
7. **Compare All Models** - Side-by-side comparison of all 5 models with charts

## f. Technologies Used

- **Python 3.12**
- **scikit-learn** - Machine Learning models and evaluation
- **Streamlit** - Interactive web application
- **pandas** - Data manipulation
- **NumPy** - Numerical operations
- **matplotlib & seaborn** - Data visualization
- **joblib** - Model serialization
