import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3]])
y = np.array([2, 3, 4])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predictions
X_test = np.array([[4], [5]])
predictions = model.predict(X_test)
print("Predictions:", predictions)
