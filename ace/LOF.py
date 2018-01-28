import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import h5py
import time
from sklearn.neighbors import LocalOutlierFactor

# # import data - shuttlelog
# filepath = 'data/shuttlelog/rand/shuttle_3.hdf5'
# f = h5py.File(filepath, 'r')
# data = f['shuttle']
# trainX = data[:, :-1]
# trainY = data[:, -1]

# import data - HEP
filepath = 'data/HEP_Daniel/hep.hdf5'
f = h5py.File(filepath, 'r')
trainX = f['features']
trainY = f['targets']


start = time.time()
# fit the model
N = [5, 10, 50]
for n in N:
    print('N = ', n)
    clf = LocalOutlierFactor(n_neighbors=n)
    y_pred = clf.fit_predict(trainX)

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
    print("total:" total)
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
    print ("LOF:", end - start)
