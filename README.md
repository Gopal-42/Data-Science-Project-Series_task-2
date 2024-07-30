# Data-Science-Project-Series_task-2
# Breast Cancer Prediction 
# Author: Gopal Krishna 
# Batch: July
# Domain: DATA SCIENCE 

## Aim
The aim of this project is to predict the presence of breast cancer using machine learning techniques in Python. We use a well-known dataset to train and evaluate models, ultimately aiming to provide accurate predictions to aid in early detection.

## Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow/keras (optional for deep learning)

## Dataset
We use the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository. This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Example usage
df = load_data()
```

## Data Processing
Data processing involves cleaning, handling missing values, and splitting the data into training and testing sets:
```python
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Example usage
X_train, X_test, y_train, y_test = preprocess_data(df)
```

## Model Training
We train machine learning models such as Logistic Regression using the preprocessed data:
```python
from sklearn.linear_model import LogisticRegression

def build_model(X_train, y_train):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model

# Example usage
model = build_model(X_train, y_train)
```

## Conclusion
This project demonstrates how to predict breast cancer using Python and machine learning techniques. By following the outlined steps, one can load, preprocess, and analyze breast cancer data, train models, and make predictions to assist in early diagnosis.

---
