import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("StudentPerformanceFactors.csv")
data.head()
data.info()
data.describe()
data.isnull().sum()
data.duplicated().sum()
data.drop_duplicates(inplace=True)
data.dropna


print(data.head())

X = data.drop(['Exam_Score','Gender'], axis = 1)
y = data['Exam_Score']

X = pd.get_dummies(X, drop_first=True)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train,y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


tolerance = 0.1 
accuracy = np.mean(np.abs((y_pred - y_test) / y_test) < tolerance)
print("Approximate Accuracy:", accuracy * 100, "%")




plt.scatter(y_test, y_test, alpha=0.7, color="blue", label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.show()

# 2. Residuals (errors)
residuals = y_test - y_test
plt.hist(residuals, bins=20, color="purple", alpha=0.7)
plt.title("Residual Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()


