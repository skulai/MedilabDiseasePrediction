import numpy as np
import urllib
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
# url with dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:8]
y = dataset[:,8]

X_train, X_test, y_train, y_test = train_test_split(X,y)
print X_train.shape, X_test.shape

#KNN with 3 neighbours
diab_dt = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = diab_dt.predict(X_test)
y_train_pred = diab_dt.predict(X_train)

#Let's get the score summary
print "Results with 3 Neighbors"
print metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
print metrics.classification_report(y_train, y_train_pred, target_names=['No Diabetes', 'Diabetes'])

