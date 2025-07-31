import numpy as np

def cal_percentile(data,percentile):
    return np.percentile(data,percentile)


data = [10,15,20,25,30]
res = cal_percentile(data, percentile=75)
print(res)
