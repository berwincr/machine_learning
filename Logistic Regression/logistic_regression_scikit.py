
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = np.array([
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 4],
    [6, 5]
])

y = np.array([0, 0, 0, 1, 1, 1])
model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)
print("Weights (Coefficients):", model.coef_)
print("Bias (Intercept):", model.intercept_)

print("\nPredicted Values:", y_pred)
print("Actual Values   :", y)
accuracy = accuracy_score(y, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))


print("\nClassification Report:")
print(classification_report(y, y_pred))
new_student = np.array([[7, 6]])

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nPrediction: Student will Pass")
else:
    print("\nPrediction: Student will Fail")