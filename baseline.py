import csv
import re 
import misc
import time
import fs
import data_process as dp

from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import chi2, SelectKBest
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



train_path = "./data/train.json"
test_path = "./data/test.json"

train = dp.form_matrix(train_path, type=3)
test = dp.form_matrix(test_path, type=3)


train_text, train_y = misc.getTextAndLabel(train3)
test_text, test_y = misc.getTextAndLabel(test3)


### vectorization
# vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', sublinear_tf=True)
# vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', lowercase=True, sublinear_tf=True, tokenizer=LemmaTokenizer())
vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', lowercase=True, sublinear_tf=True, tokenizer=LemmaTokenizer(), ngram_range=(1,2))
vectorizer.fit(train_text)
train_X3 = vectorizer.transform(train_text)
test_X3 = vectorizer.transform(test_text)


### Feature selection: chi-square
ch2, train_X_ch2, test_X_ch2 = fs.chisq(train_X3, train_y, test_X3, 2000)
