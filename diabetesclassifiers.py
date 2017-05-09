# Gaussian Naive Bayes Classification
"""Source: http://machinelearningmastery.com/spot-check-classification-machine-learning-algorithms-python-scikit-learn/"""
import pandas
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
#takes all the column from 0-8
X = array[:,0:8]
#takes only 8th column
Y = array[:,8]
#seed = 7
kfold = model_selection.KFold(n_splits=9, random_state=None)
model = GaussianNB()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print (" Mean cross-validation score of Naive Bayes Classification")
print(results.mean())


# SVM Classification
import pandas
from sklearn import model_selection
from sklearn.svm import SVC
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = SVC();
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print (" Mean cross-validation score of SVM Classification")
print(results.mean())