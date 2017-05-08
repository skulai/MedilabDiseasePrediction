#this code uses an ensemble of different algorithms
import re
import numpy as np
import urllib
from sklearn.cross_validation import train_test_split
# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Cross validation
from sklearn import cross_validation, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

# Convert text to vector
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

names = ["KNN", "Decision Tree", "SVM", "Naive Bayes"]
classifiers = [KNeighborsClassifier(n_neighbors=3, weights='distance'),
DecisionTreeClassifier(random_state=0),
SVC(kernel="linear", C=1.0),
MultinomialNB()
]

classifier_types = []
for name, clf in zip(names, classifiers):
    print '\nMetric for ' + name
    cv_predicted = cross_val_predict(clf, X_train, y_train, cv=5)
    print metrics.classification_report(y_train, cv_predicted)
    scores = cross_validation.cross_val_score(clf, X_train, y_train)
    print '\nCross validation scores: ', scores.mean()

    clf.fit(X_train, y_train)
    label_predicted_for_test = clf.predict(X_test)
    print 'Label predicted by '+ name, label_predicted_for_test
    classifier_types.append((name, clf))
    
print classifier_types

print "Ensemble output:"
eclf1 = VotingClassifier(estimators=classifier_types, voting='hard', weights=[1,1,2,1])
eclf1 = eclf1.fit(X_train, y_train)
ensemble_label = eclf1.predict(X_test)
scores = cross_validation.cross_val_score(eclf1, X_train, y_train)
print '\nCross validation scores: ', scores.mean()

