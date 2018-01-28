import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import h5py
import time
from sklearn.neighbors import NearestNeighbors
from preprocess_shuttle import preprocess_shuttle

# import data
print('importing data files...')
data = preprocess_shuttle()
print('finish importing data!')

trainX = data[:, :-1]
trainY = data[:, -1]

start = time.time()
# fit the model
clf = NearestNeighbors(n_neighbors=5)
y_pred = clf.fit(trainX)
print(y_pred.shape)

end = time.time()

total = len(trainX)
R_count = 0
corr_count = 0
miss_count = 0

total_out = 0
total_norm = 0


for i in range(0, total):
    if(trainY[i] == 1):
        total_out += 1
    else:
        total_norm += 1

print("norm: ", total_norm)
print("outliers: ", total_out)
print(total)
for i in range(0, total):
    if(y_pred[i] == -1):
        R_count += 1
        if(trainY[i] == 1):
            corr_count += 1
    else:
        if(trainY[i] == 1):
            miss_count += 1

print ("reported", R_count)
print ("correct", corr_count)
print ("miss", miss_count)
print ("kNN:", end - start)
