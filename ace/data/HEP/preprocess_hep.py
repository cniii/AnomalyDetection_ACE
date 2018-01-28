import numpy as np
import h5py

# reads two h5 data set: one is class0(background) and the other is class1(signal)
# merge and shuffle the two classes to generate one dataset for learning.

f_0 = h5py.File('m100_valid_class0.h5', 'r')
f_1 = h5py.File('m100_valid_class1.h5', 'r')

# list keys
print('keys: ', list(f_0.keys()))

# class 0
c0_features = np.array(f_0['features'])
c0_fileidx = np.array(f_0['fileidx'])
c0_imass = np.array(f_0['imass'])
c0_targets = np.array(f_0['targets'])
c0_tmass = np.array(f_0['tmass'])
c0_weights = np.array(f_0['weights'])

# class 1
c1_features = np.array(f_1['features'])
c1_fileidx = np.array(f_1['fileidx'])
c1_imass = np.array(f_1['imass'])
c1_targets = np.array(f_1['targets'])
c1_tmass = np.array(f_1['tmass'])
c1_weights = np.array(f_1['weights'])

# write them to one dataset
f = h5py.File('hep.hdf5', 'w')
# 1
features = np.vstack((c0_features, c1_features))
np.random.shuffle(features)
f.create_dataset('features', data=features)
# 2
fileidx = np.concatenate((c0_fileidx, c1_fileidx))
np.random.shuffle(fileidx)
f.create_dataset('fileidx', data=fileidx)
# 3
imass = np.concatenate((c0_imass, c1_imass))
np.random.shuffle(imass)
f.create_dataset('imass', data=imass)
# 4
targets = np.concatenate((c0_targets, c1_targets))
np.random.shuffle(targets)
f.create_dataset('targets', data=targets)
# 5
tmass = np.concatenate((c0_tmass, c1_tmass))
np.random.shuffle(tmass)
f.create_dataset('tmass', data=tmass)
# 6
weights = np.concatenate((c0_weights, c1_weights))
np.random.shuffle(weights)
f.create_dataset('weights', data=weights)

f.close()
