import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = 3 * np.random.rand(100,2)
Y = 4 + 2 + X[:, 0] + X[:, 1] + np.random.rand(100)

model = LinearRegression()
model.fit(X,Y)

coefficients = model.coef_
intercept = model.intercept_
print(f"Coefficents: {coefficients}")
print(f"Intercepts: {intercept}")