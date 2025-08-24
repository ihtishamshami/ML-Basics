import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
data = np.random.rand(100, 2) * 100 + 500

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print("Original Data:\n", data[:5])
print("Scaled Data:\n", scaled_data[:5])
