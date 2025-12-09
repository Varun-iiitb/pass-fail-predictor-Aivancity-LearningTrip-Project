# Pass–Fail Predictor

This project implements a complete machine learning pipeline to classify students into performance categories (`GradeClass`) using academic, behavioral, and demographic features.

## Overview

- **Dataset:** `Student_performance.csv`
- **Objective:** Predict the `GradeClass` of students
- **Key Components:**
  - Outlier handling and categorical encoding
  - Class balancing with SMOTE
  - Feature selection using Random Forest importances
  - Weighted soft-voting ensemble model
  - Probability calibration
  - Class-wise threshold tuning for improved F1-score

---

## Models Used

The classification system uses the following models:

- Logistic Regression (with Standard Scaling)
- Decision Tree
- Random Forest
- XGBoost

These models are combined using a **weighted soft-voting ensemble classifier**.

---

## Pipeline Steps

### 1. **Preprocessing**
- Cap outliers in `StudyTimeWeekly` and `Absences`
- Frequency-encode categorical variables
- Remove original categorical columns after encoding

### 2. **Train–Test Split**
- Split the data (80–20) with stratification to maintain class balance

### 3. **Class Balancing**
- Apply **SMOTE** to oversample minority classes in the training set

### 4. **Feature Selection**
- Use Random Forest feature importances  
- Select features above the **median importance threshold**

### 5. **Model Training**
- Train individual models
- Build a **weighted soft-voting ensemble** for final predictions

### 6. **Probability Calibration**
- Apply **isotonic regression** using 5-fold cross-validation  
- Improves reliability of predicted probabilities

### 7. **Threshold Optimization**
- Tune decision thresholds **for each class** using a validation set  
- Enhances F1-score and performance balance

### 8. **Final Prediction**
- Predict using calibrated probabilities + optimized thresholds

### 9. **Evaluation**
- Accuracy
- Classification report
- Confusion matrix

### 10. **Visualization**
Includes:
- Class distribution plot
- Feature importances
- Confusion matrix heatmap
- Correlation analysis of selected features
- Category-wise feature breakdown across grade classes

---

## Dependencies

Install all required libraries using:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib shap
```

## Running the Code

Ensure the dataset `Student_performance.csv` is in the working directory. Then run the script:

```bash
python student_performance_classifier.py
```

## Output

* Printed accuracy and classification report
* Visual plots of:

  * Grade class distribution
  * Feature importances
  * Confusion matrix
  * Correlated feature relationships
  * Categorical feature breakdown by class

## Author

Developed by Adithya Kommuri, Shivansh Shah, Tejas Kollipara, Varun Ekambaranath

for student analytics and educational outcome prediction.


