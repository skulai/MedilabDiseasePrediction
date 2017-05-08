import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib
from sklearn import model_selection
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = pd.read_csv("heart_disease.csv",header=0)

features=list(data.columns[0:13])
model=DecisionTreeClassifier()
train, test = train_test_split(data, test_size = 0.3)
train_X = train[features]# taking the training data input 
print "train_X"
print train_X

print "train_Y"
train_Y = train.outcome# This is output of our training data
print train_Y
test_X= test[features] # taking test data inputs
test_Y =test.outcome   #output value of test dat
model.fit(train_X,train_Y)
y_pred=model.predict(test_X)

print metrics.classification_report(test_Y, y_pred, target_names=['No Heart Disease', 'Heart Disease'])



