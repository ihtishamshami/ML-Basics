from statistics import mode


def cal_mean(data):
    mean = sum(data)/len(data)
    return mean

def cal_mid(data):
    data.sort()
    n = len(data)
    mid = n // 2
    if n % 2 == 0:
        median = (data[mid-1] + data[mid])/2
    else:
        median = data[mid]
    return median



dataset  = [12,45,12,36,10,20]
R1 = cal_mean(dataset)
R2 = cal_mid(dataset)
R3 = mode(dataset)

print("Mean",R1)
print("Median", R2)
print("Mode", R3)