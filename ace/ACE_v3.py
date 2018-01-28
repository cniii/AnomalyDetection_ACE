import numpy as np
from numpy.random import multivariate_normal as mvn
import matplotlib.pyplot as plt
import h5py
import math
import time
from sklearn.preprocessing.data import StandardScaler

# [delete]prepare random sample once and use it through out all test for consistency
# data = preprocess_shuttle()

# -----------Shuttle----------------
# filepath = 'data/shuttlelog/rand/'
# dset = h5py.File(filepath + "shuttle_3.hdf5", 'r')
# data = dset['shuttle'][:]
# trainX = data[:, :-1]
# trainY = data[:, -1]

# # small scale testing
# data = data[np.random.choice(data.shape[0], 1000, replace=False)]

# -----------HEP------------
# import data - HEP
filepath = 'data/HEP_Daniel/hep.hdf5'
f = h5py.File(filepath, 'r')
trainX = f['features']
trainY = f['targets']


def SRP(x):
    # given a value x, returns a hash function of length dim(x)
    # return np.random.normal(0,1,x)
    vals = []
    for i in range(x):
        vals.append(np.random.normal(0, 1))
    return vals


class ACE:
    def __init__(self, D, K, L, alpha):
        self.D = D
        self.K = K
        self.L = L
        self.alpha = alpha
        # initalize hash - generate L Hj(x) with K independent SRPs each
        x = len(self.D[0])
        self.H = np.zeros((self.L, self.K, x))
        self.A = np.zeros((self.L, 2**self.K))
        self.mu = 0
        self.n = 0

        for j in range(self.L):
            self.H[j] = np.zeros((self.K, x))
            for i in range(self.K):
                self.H[j][i] = SRP(x)
            # Aj = new short arrays
            self.A[j] = np.zeros(2**self.K)

        # preprocessing:
        for x in self.D:
            mu_i = 0
            for j in range(self.L):
                # get the index in the A[j] array corresponding to the SRP-hashed x
                # to do this: get the K-length vector of SRP hashes for x
                # take the sign of each of the hashes, convert to (0 negative, 1 positive), concatenate to get binary index,
                # convert binary index to get array index
                val = np.sign(np.inner(self.H[j], x))
                val[val == -1] = 0
                bindex = ''.join(map(str, val.astype(int)))
                ind = int(bindex, 2)
                # increment Aj[Hj(x)]
                self.A[j][ind] += 1
                a = self.A[j][ind]
                mu_i += (2 * a + 1) / self.L
            self.mu = (self.n * self.mu + mu_i) / (self.n + 1)
            self.n += 1
        print('max collision', np.amax(self.A))

    def query(self, q):
        S = 0
        for j in range(self.L):
            val = np.sign(np.inner(self.H[j], q))
            val[val == -1] = 0
            bindex = ''.join(map(str, val.astype(int)))
            ind = int(bindex, 2)
            S += self.A[j][ind] / self.L
        if S <= (self.mu - alpha):
            # print('mu', self.mu)
            return (S, True)
        else:
            return (S, False)


K = 15
L = 50
al = [1, 15, 50]

for k in range(3):
    alpha = al[k]

    print('K=', K)
    print('L=', L)
    print('alpha=', alpha)

    start = time.time()

    # print("setting up ACE estimator...")
    estimator = ACE(trainX, K, L, alpha)
    # print("setup finished!")

    # print("estimating...")
    result = np.array([estimator.query(val)[1] for val in trainX])
    # print("finshed estimating!")

    end = time.time()

    print('ACE Runtime:', end - start)

    # print("analyzing results...")
    r_nor = 0
    r_out = 0
    cr_nor = 0
    cr_out = 0
    for k in range(len(result)):
        if result[k] == 1:
            r_out = r_out + 1
            if (trainY[k] == 1):
                cr_out = cr_out + 1
        else:
            r_nor = r_nor + 1
            if trainY[k] == 0:
                cr_nor = cr_nor + 1

    # print("finished collecting results!")
    print("reported normal: ", r_nor)
    print("correctly reported normal: ", cr_nor)

    print("reported outliers: ", r_out)
    print("correctly reported outliers: ", cr_out)
    print("=====END=======")
    print()


# print('Calculating Learning Results...')
# data = b_dat
# print(len(data))
# data = data[np.random.choice(data.shape[0], 50000, replace=False), :]  # 5000?
# outliers = s_dat
# # outliers = outliers[np.random.choice(outliers.shape[0], 10000, replace=False), :]

# mus = []
# norms = []
# xs = [12, 15, 17]
# outls = []

# K = 15
# L = 50
# alpha = 5
# print("K=", K, "L=", L)
# # print('estimating')
# start = time.time()
# est = ACE(data, K, L, alpha)
# end_ace = time.time()
# print("ACE runtime:", end_ace - start)
# outliers = s_dat

# # print('querying outliers')
# start_q = time.time()
# # outresmean = np.mean([est.query(val)[0] for val in outliers])
# outres = np.array([est.query(val)[0] for val in outliers])

# # print('querying normal data')
# normres = np.array([est.query(val)[0] for val in data])
# alphadict = {}
# alphadict['corrout'] = []
# alphadict['errout'] = []
# alphadict['corrnorm'] = []
# alphadict['errnorm'] = []
# end_q = time.time()
# print("query time:", end_q - start_q)

# alphas = np.linspace(0, 100, 20)
# # print('alphas len', len(alphas))
# for alpha in alphas:
#     # print(alpha)
#     corrout = outres[abs(outres - est.mu) < alpha]  # tp - signal efficiency
#     errout = outres[abs(outres - est.mu) >= alpha]  # fp
#     corrnorm = normres[abs(normres - est.mu) >= alpha]  # tn - background rejection
#     errnorm = normres[abs(normres - est.mu) < alpha]  # fn
#     alphadict['corrout'].append(len(corrout) / len(outliers))
#     alphadict['errout'].append(len(errout) / len(outliers))
#     alphadict['corrnorm'].append(len(corrnorm) / len(data))
#     alphadict['errnorm'].append(len(errnorm) / len(data))
#     # print(alphadict['corrout'])
#     # print(alphadict['errout'])
#     # print(alphadict['corrnorm'])
#     # print(alphadict['errnorm'])
# end = time.time()

# plt.clf()
# plt.plot(alphas, alphadict['corrout'], 'b', label='True positive rate')
# plt.plot(alphas, alphadict['errout'], 'r', label='False negative rate')
# plt.plot(alphas, alphadict['corrnorm'], 'k', label='True negative rate')
# plt.plot(alphas, alphadict['errnorm'], 'g', label='False positive rate')
# plt.title('Results, adv, K = 15, L = 50')
# plt.xlabel('$alpha$')
# plt.legend(loc='best')
# plt.show()

# print('total time for K=15 L=50', end - start)

# np.savetxt("shuttle_csv/se_15_50.csv", np.sort(alphadict['corrout']), delimiter=',')
# np.savetxt("shuttle_csv/bj_15_50.csv", np.sort(alphadict['corrnorm']), delimiter=',')

# print('done!')

# # plt.clf()
# # plt.plot(alphas,alphadict['corrout'],label='True positive rate')
# # plt.plot(alphas,alphadict['errout'],label='False negative rate')
# # plt.plot(alphas,alphadict['corrnorm'],label='True negative rate')
# # plt.plot(alphas,alphadict['errnorm'],label='False positive rate')
# # plt.title('Results, adv, K = 15, L = 50')
# # plt.xlabel('$alpha$')
# # plt.legend(loc='best')
# # plt.show()

# # save figures
# # plt.savefig('ACE_Alpha_Test')
# # plt.clf()
# # plt.plot(xs, np.asarray(mus) - np.asarray(outls), label = 'outlier deviation from mean')
# # plt.plot(xs, np.asarray(mus) - np.asarray(norms), label = 'inner point deviation from mean')
# # plt.legend(loc='best')
