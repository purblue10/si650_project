from sklearn import svm
import numpy as np
import csv

#X_train = open('output_test.csv', 'rb').readlines()
# X_train = csv.reader(open('train.csv', 'rb'))
# Y_train = csv.reader(open('train_class.csv', 'rb'))
# X_test = csv.reader(open('output_test_x.csv', 'rb'))

train = dp.form_matrix(train_path, type=1)

train_X = [ row[0:1]+row[2:]  for row in train]
train_y = [ row[0]  for row in train]

train_X = np.array(train_X)
train_y = np.array(train_y)


X = []
for line in X_train:
	xline = []
	for i in range(9):
		xline.append(float(line[i]))
	X.append(xline)

Y = []
for line in Y_train:
	Y.append(int(line[0]))



x = []
for line in X_test:
	xline = []
	for i in range(9):
		xline.append(float(line[i]))
	x.append(xline)
x = np.array(x)

clf = svm.SVC()
# clf = svm.SVC(kernel='rbf', C=500, gamma=0.001, cache_size=500)
clf.fit(X, Y)

predict = clf.predict(x)
output_csv = csv.writer(open('output_test_class.csv','wb', buffering=0))

for i in predict:
    output_csv.writerow([i])