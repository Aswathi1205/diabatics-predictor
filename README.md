

# 🩺 Diabetes Prediction using Logistic Regression

This project is a simple machine learning pipeline built in Python to predict whether a person is likely to have diabetes using the **Pima Indians Diabetes Dataset**. It uses **Logistic Regression** as the classification model.

## 📊 Dataset

The dataset used is the **Pima Indians Diabetes Database** from the UCI Machine Learning Repository, available at:
[https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

### Features (Columns):

* **Pregnancies**: Number of times pregnant
* **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
* **BloodPressure**: Diastolic blood pressure (mm Hg)
* **SkinThickness**: Triceps skin fold thickness (mm)
* **Insulin**: 2-Hour serum insulin (mu U/ml)
* **BMI**: Body mass index (weight in kg/(height in m)^2)
* **DiabetesPedigreeFunction**: Diabetes pedigree function
* **Age**: Age (years)
* **Outcome**: Class variable (0 or 1), where 1 means the patient has diabetes

## ⚙️ Features

* Data loading and preprocessing
* Feature scaling using `StandardScaler`
* Train-test split (80-20)
* Logistic Regression model training
* Model evaluation using:

  * Accuracy
  * Confusion Matrix
  * Classification Report
* Function for predicting diabetes based on new user input

## 🧪 Example Prediction

```python
example_input = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
result = predict_diabetes(example_input)
print(result)
```

**Output:**

```
The person is likely to have diabetes.
```

## 🛠️ Requirements

* Python 3.x
* pandas
* scikit-learn

You can install the dependencies using:

```bash
pip install pandas scikit-learn
```

## 🚀 Running the Project

1. Clone the repository or copy the script.
2. Make sure all dependencies are installed.
3. Run the Python file.

```bash
python diabetes_prediction.py
```

## 📈 Model Evaluation Output (Sample)

```
Accuracy: 0.77
Confusion Matrix:
[[90 17]
 [18 29]]
Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.84      0.83       107
           1       0.63      0.62      0.62        47

    accuracy                           0.77       154
   macro avg       0.73      0.73      0.73       154
weighted avg       0.77      0.77      0.77       154
```

## 📌 Notes

* Some features like `Insulin` or `SkinThickness` may contain zero values, which are not valid; in a production scenario, missing value treatment would be important.
* The model can be improved with hyperparameter tuning or by using advanced algorithms.


