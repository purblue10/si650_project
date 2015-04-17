from sklearn import svm
import numpy as np
import csv

#X_train = open('output_test.csv', 'rb').readlines()
X_train = csv.reader(open('train.csv', 'rb'))
Y_train = csv.reader(open('train_class.csv', 'rb'))
X_test = csv.reader(open('output_test_x.csv', 'rb'))

X = []
for line in X_train:
	xline = []
	for i in range(9):
		xline.append(float(line[i]))
	X.append(xline)

Y = []
for line in Y_train:
	Y.append(int(line[0]))

X = np.array(X)
Y = np.array(Y)

x = []
for line in X_test:
	xline = []
	for i in range(9):
		xline.append(float(line[i]))
	x.append(xline)
x = np.array(x)

clf = svm.SVC()
clf.fit(X, Y)

predict = clf.predict(x)
output_csv = csv.writer(open('output_test_class.csv','wb',buffering=0))

for i in predict:
    output_csv.writerow([i])