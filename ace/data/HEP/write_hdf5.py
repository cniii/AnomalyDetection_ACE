# wirte given file to a hdf5 file
import numpy as np
import h5py

# filepath = "shuttlelog/"
# input = open(filepath + "shuttle.tst", "r")
# entries = 14500
# attri = 10

filepath = ""
input = open(filepath + "public_train.csv", "r")
entries = 100000
attri = 17

normal = 0
ab = 0

# f = h5py.File('shuttle_test.hdf5', 'w')
# dset = f.create_dataset("shuttle_data", (entries, attri))

f = h5py.File('hep_train.hdf5', 'w')
dset = f.create_dataset("hep_data", (entries, attri))

s = input.readline()

for j in range(entries):
    s = input.readline()
    tokens = s.split(',')
    tokens = list(map(float, tokens))
    # print(tokens)
    arr = np.array(tokens)
    # print(arr.shape)
    dset[j] = arr
    if (arr[attri - 1] == 1):
        dset[j, attri - 1] = 0
        normal = normal + 1
    else:
        dset[j, attri - 1] = 1
        ab = ab + 1
input.close()

print("normal: ", normal, "abs: ", ab)

print(dset)
print("finished writing to hdf5")

# debugging: read the hdf5 file
# f = h5py.File("shuttle_test.hdf5", "r")

# for arr in f["shuttle_data"][:20]:
#     print(arr)
