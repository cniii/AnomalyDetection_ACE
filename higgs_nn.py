import numpy as np
import matplotlib.pyplot as plt
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

'''
MLP for binary classification
'''
# import data
X = []
Y = []
print('importing data...')
with open('../data/training.csv', newline='') as file:
	reader = csv.reader(file)
	header = next(reader)
	for row in reader:
		X.append([float(x) for x in row[:-1]])
		if (row[-1] == 's'):
			Y.append(1)
		else:
			Y.append(0)

print('done importing!')
indices = [1, 2, 3, 4, 8, 9 ,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

X = np.array(X)
X = np.append(X[:, 1:5], X[:, 8:24], 1)
Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# building a sequential MLP model
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# fit the training data 
print('fitting and predicting the data...')
model.fit(x_train, y_train,
          epochs=50,
          batch_size=128)
predict = model.predict(x_test, batch_size=None)
print('done predicting!')

print('start plotting')
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr[0], tpr[0], _ = roc_curve(y_test[:], predict[:])
roc_auc[0] = auc(fpr[0], tpr[0])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), predict.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

fig = plt.figure()
fig.suptitle('Higgs Challenge MLP[3 layers, 50 epoch]', fontsize=10)
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
fig.savefig("higgs_mlp.png")

print('done plotting!')

