


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
           'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)


X = data.drop('Outcome', axis=1)
y = data['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

def predict_diabetes(input_features):

    input_features = pd.DataFrame([input_features], columns=columns[:-1])
    input_features = scaler.transform(input_features)


    prediction = model.predict(input_features)


    if prediction[0] == 1:
        return "The person is likely to have diabetes."
    else:
        return "The person is unlikely to have diabetes."



example_input = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
result = predict_diabetes(example_input)
print(result)