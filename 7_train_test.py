from sklearn.model_selection import train_test_split
import numpy as np


# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1)
Y = 2 * X.squeeze() + 1 + 0.1 * np.random.rand(100) 
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Print the shapes of the datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)
