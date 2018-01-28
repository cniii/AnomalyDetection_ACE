import numpy as np
import h5py


def preprocess_shuttle():
    # import data
    print('importing data files...')
    filepath = "raw_data/"
    train = h5py.File(filepath + "shuttle_test.hdf5", 'r')
    test = h5py.File(filepath + "shuttle_train.hdf5", 'r')

    trainset = np.asarray(train['shuttle_data'])
    testset = np.asarray(test['shuttle_data'])
    print('finish importing data!')

    # random sampling
    print("random sampling...")
    num_normal = 34987
    num_outlier = 879

    data = np.concatenate((trainset, testset))
    normal = data[np.nonzero(data[:, -1] == 0)]
    outlier = data[np.nonzero(data[:, -1] == 1)]
    normal = normal[np.random.choice(normal.shape[0], num_normal, replace=False), :]
    outlier = outlier[np.random.choice(outlier.shape[0], num_outlier, replace=False), :]
    processed = np.concatenate((outlier, normal))
    np.random.shuffle(processed)
    print('finish data preprocessing!')
    return processed


# writing to hdf5
pro = preprocess_shuttle()
f = h5py.File('rand/shuttle_3.hdf5', 'w')
f.create_dataset('shuttle', data=pro)
print('finished writing to hdf5')
f.close()

# # debuggin
# f = h5py.File("rand/shuttle_3.hdf5", "r")
# dset = f['shuttle'][:]
# print(dset[:10, :])
