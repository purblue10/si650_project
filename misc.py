import csv
import re 
import time

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
from sklearn.metrics import f1_score

from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
   def __init__(self):
      self.wnl = WordNetLemmatizer()
   def __call__(self, doc):
      return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def check_special_characters(sent):
   while(re.search('[^a-zA-Z0-9\-\+\'\"\#\*\@\!\?\/\&\(\)\:\;\,\.\s]',sent)):
      sent = re.sub('[^a-zA-Z0-9\-\+\'\"\#\*\@\!\?\/\&\(\)\:\;\,\.\s]','',sent)
   if re.search('\( *',sent):
      sent = re.sub('\( *', ' (',sent)
   if re.search(' *\)',sent):
      sent = re.sub(' *\)', ') ',sent)
   if re.search(',',sent):
      sent = re.sub('\,',' ', sent)
   return sent

def condensing(sent):
   # pattern = re.compile('[a-zA-Z0-9][\-\+\'\"\#\*\@\!\?\&\:\;\.][a-zA-Z0-9]')
   # while True:
   #    result = pattern.search(sent)
   #    if result==None:
   #       break
   #    index = result.start()+1
   #    sent=sent[:index]+sent[index+1:]
   # return re.sub('[^a-zA-Z0-9\s]',' ',sent).strip()
   sent = re.sub('[^a-zA-Z0-9\s]',' ',sent).strip()
   return re.sub('[0-9]','',sent).strip()

def setenceProcessing(sent):
   sent = check_special_characters(sent)
   sent = condensing(sent)
   return sent

def readData(path):
   data = []
   with open(path,'r') as reader:
      for line in reader:
         text = line[2:]
         text = check_special_characters(text)
         text = condensing(text)
         tp = (int(line[0]), text)
         data.append(tp)
   X = [row[1] for row in data]
   y = [row[0] for row in data]
   return X,y

def writeResult(path, result):
   output = [ (i+1, result[i]) for i in range(len(result))]
   writer = csv.writer(open(path, 'wb', buffering=0))
   writer.writerows([("Id","Category")])
   writer.writerows(output)


def getTextAndLabel(data):
   data_label = [ row['label'] for row in data]
   data_text = []
   for row in data:
      text = ""
      for line in data[0]['text']:
         text += line[0]+" "
      data_text.append(text)
   data_text = [" ".join([ line[0] for line in row['text']]) for row in data]
   return data_text, data_label

def getParameters(clf, feature_model):
   # feature model
   if type(feature_model) is PCA or type(feature_model) is RandomizedPCA:
      feature = 'pca'
      feature_param = feature_model.get_params()['n_components']
   elif  type(feature_model) is SelectKBest:
      feature = 'ch2'
      feature_param = feature_model.get_params()['k']
   # classifier
   if type(clf) is OneVsRestClassifier:
      try:
         c = clf.get_params()['estimator__estimator'].C
         gamma = clf.get_params()['estimator__estimator'].gamma
      except(KeyError):
         c = clf.get_params()['estimator__C']
         gamma = clf.get_params()['estimator__gamma']
   elif type(clf) is GridSearchCV:
      c = clf.best_params_['estimator__C']
      gamma = clf.best_params_['estimator__gamma']
   return [feature, str(feature_param), str(c), str(gamma)]

# For prediction, i.e. generate submission files
# path: FSMethod_FSparam_C_gamma
def prediction(clf, test_X, feature_model):
   start_time = time.time()
   params = getParameters(clf, feature_model)
   path = "./result/submission/"+"_".join(params)+".txt"
   prediction = clf.predict(test_X)
   result = prediction.tolist()
   writeResult(path, result)
   print("prediction: --- %s seconds ---" % (time.time() - start_time))
   print("path: "+path)
   

# for cross validation log
# use empty classifier
def runCV(clf, train_X, train_y, feature_model, cvn):
   start_time = time.time()
   path = "./result/eval_log.txt"
   params = getParameters(clf, feature_model)
   if type(clf) is GridSearchCV:
      clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=float(params[2]), gamma=float(params[3])))
   scores = cross_validation.cross_val_score(clf, train_X, train_y, cv=cvn)
   mean = "{:.5f}".format(scores.mean())
   sd = "{:.5f}".format(scores.std()*2)
   line = ",".join(params)+" - accuracy: "+ str(mean) +" (+/- " + str(sd) + ")" +", "
   line += str(scores)+" \n"
   writer = open(path, 'a')
   writer.write(line)
   writer.close()
   print("cross validation: --- %s seconds ---" % (time.time() - start_time))
   print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
   print("params: "+str(params))
   return scores

def testCV(clf, train_X, train_y, cvn):
	start_time = time.time()
	scores = cross_validation.cross_val_score(clf, train_X, train_y, cv=cvn)
	mean = "{:.5f}".format(scores.mean())
	sd = "{:.5f}".format(scores.std()*2)
	line = "accuracy: "+ str(mean) +" (+/- " + str(sd) + ")" +", "
	print line
	print("cross validation: --- %s seconds ---" % (time.time() - start_time))
	return scores

def evaluation(prediction, test_label):
	n = len(test_label)
	answer = []
	for i in range(n):
		if prediction[i]==test_label[i]:
			answer.append((test_label[i], prediction[i],True))
		else:
			answer.append((test_label[i], prediction[i],False))
	numTrue = len([ row for row in answer if row[2]==True])
	numFalse = len([ row for row in answer if row[2]==False])
	numTrue_zero = len([ row for row in answer if row[2]==True and row[1]==0])
	numTrue_one = len([ row for row in answer if row[2]==True and row[1]==1])
	n_zero = len([ row for row in answer if row[1]==0])
	n_one = len([ row for row in answer if row[1]==1])
	print("Total Accuracy: %0.5f " % (numTrue/float(n)))
	print("0-label Accuracy: %0.5f (%d / %d)" % (numTrue_zero/float(n_zero), numTrue_zero, n_zero))
	print("1-label Accuracy: %0.5f (%d / %d)" % (numTrue_one/float(n_one), numTrue_one, n_one))
	return answer



# grades = {}
# for row in train:
#    g = row[1][1]
#    if g not in grades:
#       grades[g]=1
#    else:
#       grades[g]=1+grades[g]

# A+ : 844
# A : 2936
# A- : 1691

# B+ : 985
# B : 972
# B- : 410

# C+ : 219
# C : 234
# C- : 97

# D+ : 41
# D : 32
# D- : 18

# F : 22

# N/A : 21564
