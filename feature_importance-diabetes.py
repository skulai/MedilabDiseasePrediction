# Feature Importance - GIVEN IN DETAIL IN NOTES 1-http://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/
import csv
import numpy as np
import pandas as pandas;
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset


def main():
    filename = 'diabetes.csv'
    #dataset = loadCsv(filename)
    #dataset = datasets.load_iris()
    dataset = pandas.read_csv(filename);
    test_id_df = pandas.DataFrame(dataset,columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']);
    #print test_id_df
    dataset.data= test_id_df
    dataset.target = pandas.DataFrame(dataset,columns=['Outcome']);
	#print dataset[0:len(dataset)-1][:][0:8]
    #dataset.data = dataset[:-2]
    ##############################################
	#WAY-1
	# fit an Extra Trees model to the data
    model = ExtraTreesClassifier()
    model.fit(dataset.data, dataset.target)
    # display the relative importance of each attribute
    print(model.feature_importances_)
	
    ###################################################
	#WAY=2
    # create a base classifier used to evaluate a subset of attributes
    model = LogisticRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(model, 3)
    rfe = rfe.fit(dataset.data, dataset.target)
    # summarize the selection of the attributes
    print(rfe.support_)
    print(rfe.ranking_)
	
	###################################################
	#WAY-3
	# fit an Extra Trees model to the data
    model = SVR(kernel="linear")
    rfe = RFE(model, 3)
    rfe = rfe.fit(dataset.data, dataset.target)
    # display the relative importance of each attribute
    print(rfe.support_)
 
	
main()