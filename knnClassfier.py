#https://shankarmsy.github.io/stories/knn-sklearn.html
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn
from pprint import pprint
import urllib
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
raw_data = urllib.urlopen(url)
diab = np.genfromtxt(raw_data, delimiter=",")
print diab.shape
X,y = diab[:,:-1], diab[:,-1:].squeeze()
print X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X,y)
print X_train.shape, X_test.shape

#KNN with 3 neighbours
diab_knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = diab_knn.predict(X_test)
y_train_pred = diab_knn.predict(X_train)

#Let's get the score summary
print "Results with 3 Neighbors"
print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])

#KNN with 5 neighbours
diab_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred = diab_knn.predict(X_test)
y_train_pred = diab_knn.predict(X_train)

print "Results with 5 Neighbors"
print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])

#KNN with different algorithm
diab_knn = KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree").fit(X_train, y_train)
y_pred = diab_knn.predict(X_test)
y_train_pred = diab_knn.predict(X_train)

print "Results with 3 Neighbors and the Ball Tree algo"
print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])
