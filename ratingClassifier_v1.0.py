from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import svm
import numpy as np
import csv
import data_process as dp
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer

train_path = "./data/train.json"
test_path = "./data/test.json"

train1, train2, trian3 = dp.form_matrix(train_path, type=0)
teat1, test2, test3 = dp.form_matrix(test_path, type=0)

train_X1 = [ row[1][0:1]+row[1][2:]  for row in train1]
train_y = [ row[0]  for row in train1]
test_X1 = [ row[1][0:1]+row[1][2:]  for row in test1]
test_y1 = [ row[0]  for row in test1]


train_X2 = []
for item in train2:
    dic = {}
    for tag in item[1]:
        if tag not in dic:
            dic[tag] = 1
        else:
            dic[tag] += 1
    train_X2.append(dic)

v = DictVectorizer(sparse=True)
train_X2 = v.fit_transform(train_X2)

imp = Imputer(missing_values=0, strategy='mean', axis=0)
train_X2 = imp.fit_transform(train_X2)


train_X1 = sparse.csr_matrix(train_X1)
train_X1 = train_X1.toarray()

train_X = np.hstack((train_X1, train_X2))

pca.explained_variance_ratio_

pca = PCA(n_components=3)
pca.fit(train_X.toarray())
train_X_pca = pca.transform(train_X.toarray())

logit = LogisticRegression(penalty="l2", dual=True, C=10)
logit_scores = testCV(logit, train_X, train_y, 5)


pca = PCA(n_components=3)
pca.fit(train_X)
train_X_pca = pca.transform(train_X)
test_X_pca = pca.transform(test_X.toarray())



clf = svm.SVC(kernel='rbf', C=1000, gamma=0.001, cache_size=500)
score_svm = testCV(clf, train_X_pca, train_y, 5)
clf.fit(train_X_pca, train_y)


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

