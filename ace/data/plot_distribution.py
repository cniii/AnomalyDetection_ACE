# plot distribution of the given dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt

# parameters of the dataset
# entries = 43500
# attri = 10
# filepath = "shuttle_test.hdf5"

entries = 10000
attri = 17
filepath = "hep_train.hdf5"

# import dataset
dset = h5py.File(filepath, "r")
test = np.asarray(dset["hep_data"])

# normal = []
# anmly = []
# # collecting data
# for k in range(entries):
#     if (test[k, -1] == 1):
#         anmly.append(test[k, :])
#     else:
#         normal.append(test[k, :])

# normal = np.array(normal)
# anmly = np.array(anmly)
# # print("number of normal", normal.shape)
# # print("number of anomaly", anmly.shape)
# print("finish sorting data")

# start ploting
fig = plt.figure(figsize=(80, 60))
fig.suptitle('HEP Training Set Distribution', fontsize=100)

name = ['DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltar_tau_lep', '  DER_pt_to', 'DER_sum_p', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt ', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt  ', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'isSkewed']

for k in range(attri):
    p = fig.add_subplot(4, 5, k + 1)
    # for shuttle dataset
    # if (k < attri - 1):
    #     name = "feature" + str(k)
    # else:
    #     name = "label"
    print(k)
    p.set_title(name[k], size=60)
    # plt.hist(normal[:, k], histtype='step', color='blue', label='normal', linewidth=4)
    # plt.hist(anmly[:, k], histtype='step', color='red', label='anomaly', linewidth=4)
    plt.hist(test[:, k], histtype='step', color='blue', linewidth=4)
    plt.legend(loc='upper right', prop={'size': 35})

fig.savefig("hep_distribution.jpg")
print("finish drawing the distribution")
