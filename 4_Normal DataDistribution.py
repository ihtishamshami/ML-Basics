import numpy as np
import matplotlib.pyplot as plt

mean  = 50
std_dav = 10
sample_size = 1000

normal_data = np.random.normal(loc=mean, scale=std_dav, size=sample_size)

plt.hist(normal_data, bins=30, edgecolor='black', density=True)

plt.title('Normal DIstribution')
plt.xlabel('values')
plt.ylabel('Probability Density')
plt.show()