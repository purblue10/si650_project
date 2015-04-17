from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import svm
import numpy as np
import csv
import data_process_forrating as dp


train_path = "./data/train.json"
test_path = "./data/test.json"

train = dp.form_matrix(train_path)
test = dp.form_matrix(test_path)

train_X = [ row[1][0:1]+row[1][2:]  for row in train]
train_y = [ row[0]  for row in train]
test_X = [ row[1][0:1]+row[1][2:]  for row in train]
test_y = [ row[0]  for row in test]


train_X = np.array(train_X[:1000])
train_y = np.array(train_y[:1000])


clf = svm.SVC(kernel='rbf', C=500, gamma=0.001, cache_size=500)
clf = svm.SVC()
clf.fit(train_X, train_y)


model = svm.SVC(kernel='rbf', cache_size=500)
param_grid = {
    "C": [0.001, 1, 1000],
    "gamma": [0.001, 1,  1000]
}
clf_grid = GridSearchCV(model, param_grid=param_grid, score_func=f1_score, cv=5)
clf_grid.fit(train_X, train_y)

clf_grid.best_estimator_
clf_grid.grid_scores_
clf_grid.best_score_

