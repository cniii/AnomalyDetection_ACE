import numpy as np
import matplotlib.pyplot as plt
import csv

signal = []
background = []
header = []


# length: 250000  dimension:  33
print('importing data...')
with open('../data/training.csv', newline='') as file:
	reader = csv.reader(file)
	header = next(reader)
	for row in reader:
		if (row[-1] == 's'):
			signal.append([float(x) for x in row[:-1]])
		else:
			background.append([float(x) for x in row[:-1]])

print('done importing!')

signal = np.array(signal)
background = np.array(background)

# plotting 20 features
indices = [1, 2, 3, 4, 8, 9 ,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

i = 0
fig = plt.figure(figsize=(80, 60))
fig.suptitle('Higgs Challenge Training Set Distribution', fontsize=90)

print('plotting...')
for k in indices:
    p = fig.add_subplot(4, 5, i + 1)
    p.set_title(header[k], size=60)
    plt.hist(background[:, k], histtype='step', color='blue', label='background', linewidth=4)
    plt.hist(signal[:, k], histtype='step', color='red', label='signal', linewidth=4)
    plt.legend(loc='upper right', prop={'size': 35})
    i += 1

print('done plotting!')
fig.savefig("higgs_distribution.png")
print("finish drawing the distribution!")	

