#this code uses an ensemble of different algorithms
# here we will import the libraries used for machine learning
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
import pandas as pd
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

from flask import Flask, render_template, json, request
app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/diabetes")
def diabetesInput():
    return render_template('diabetes.html')
	
@app.route("/cancer")
def cancerInput():
    return render_template('cancer.html')

@app.route('/diabetesPredictionPage',methods=['POST'])
def diabetesClassifier():
    # read the posted values from the UI
    _glucose = request.form['glucose']
    _pressure = request.form['pressure']
    _insulin = request.form['insulin']
    _bmi = request.form['bmi']
    _age = request.form['age']
    _preg = request.form['pregnancy']

    # url with dataset
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    # download the file
    raw_data = urllib.urlopen(url)
    # load the CSV file as a numpy matrix
    dataset = np.loadtxt(raw_data, delimiter=",")
    # separate the data from the target attributes
    X = dataset[:,0:8]
    y = dataset[:,8]
    
    print "inputs"
    print _glucose,_pressure,_insulin,_bmi,_age,_preg

    X_test=[_preg,_glucose,_pressure,0,_insulin,_bmi,0.134,_age]

    diab_dt = DecisionTreeClassifier().fit(X, y)
    y_pred = diab_dt.predict(X_test)
    output= int(y_pred[0])
    if output == 0:
        return render_template('diabetesPrediction.html',value ="Congratulations!! You are safe from diabetes.Keep visiting nearby hospitals for regular checkups.")
    else:
        return render_template('diabetesPrediction.html',value ="There are high chances of having diabetes. Please visit nearby hospital soon.")

@app.route("/heartDisease")
def heartDiseaseInput():
    return render_template('heartDisease.html')

@app.route('/heartDiseasePredictionPage',methods=['POST'])
def heartDiseaseClassifier():
    # read the posted values from the UI
    _age = request.form['age']
    _sex = request.form['sex']
    _cpt = request.form['chest_pain_types']
    _bp = request.form['resting_BP']
    _cholesterol = request.form['serum_cholesterol']
    _sugar = request.form['bloodSugar']
    _restEcg = request.form['restEcg']
    _maxHeartRate = request.form['maxHeartRate']

    data = pd.read_csv("heart_disease.csv",header=0)
    features=list(data.columns[0:13])
    train, test = train_test_split(data, test_size = 0.1)
    X_train = train[features]
    y_train = train.outcome
    X_test=[_age,_sex,_cpt,_bp,_cholesterol,_sugar,_restEcg,_maxHeartRate,0,2.5,3,0,6]
    print "input"
    print X_test
    heart_dt = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
    y_pred = heart_dt.predict(X_test)
    output= int(y_pred[0])

    if output == 0:
        return render_template('heartDiseasePrediction.html',value ="Congratulations!! You are safe from heart disease.Keep visiting nearby hospitals for regular checkups.")
    else:
        return render_template('diabetesPrediction.html',value ="There are high chances of having heart disease. Please visit nearby hospital soon.")


@app.route('/cancerPredictionPage',methods=['POST'])
def cancerClassifier():

	# Any results you write to the current directory are saved as output.
	data = pd.read_csv("data.csv",header=0)# here header 0 means the 0 th row is our coloumn 
	# have a look at the data
	print(data.head(2))#
	data.drop("Unnamed: 32",axis=1,inplace=True)
	print data.columns
	# like this we also don't want the Id column for our analysis
	data.drop("id",axis=1,inplace=True)

	features_mean= list(data.columns[1:11])
	features_se= list(data.columns[11:20])
	features_worst=list(data.columns[21:31])

	data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
	data.describe()
	#print data[0: 10]
	sns.countplot(data['diagnosis'],label="Count")
	corr = data[features_mean].corr() # .corr is used for find corelation


			   
	#prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
	#prediction_var = features_mean
	#low acuracy|||above both have low accuracy

	prediction_var = features_worst
	# the accuracy for RandomForest invcrease it means the value are more catogrical in Worst part
	#lets get the important features
	model=RandomForestClassifier(n_estimators=100,criterion="entropy")



	#featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
	#print(featimp)
	# this is the property of Random Forest classifier that it provide us the importance 
	# of the features used

	#prediction_var = ['concave points_worst','radius_worst','area_worst','perimeter_worst','concavity_worst'] 
	prediction_var = ['concave points_worst','radius_worst','area_worst','perimeter_worst','concavity_worst'] 

	train, test = train_test_split(data, test_size = 0.3)

	#RANDOMFORREST
	train_X = train[prediction_var]# taking the training data input 
	train_Y = train.diagnosis# This is output of our training data
	# same we have to do for test
	test_X= test[prediction_var] # taking test data inputs
	test_Y =test.diagnosis   #output value of test dat
	#model=RandomForestClassifier(n_estimators=100, criterion="entropy")
	model.fit(train_X,train_Y)
	print("------------------------------------")
	# read the posted values from the UI
	_concave_points_worst = request.form['concave points_worst']
	_radius_worst = request.form['radius_worst']
	_area_worst = request.form['area_worst']
	_perimeter_worst = request.form['perimeter_worst']
	_concavity_worst = request.form['concavity_worst']


	print "inputs"
	print _concave_points_worst, _radius_worst,_area_worst, _perimeter_worst, _concavity_worst
	X_test=[_concave_points_worst, _radius_worst,_area_worst, _perimeter_worst, _concavity_worst]

	prediction=model.predict(X_test)
	print prediction
	#print metrics.accuracy_score(prediction,test_Y)
	output= int(prediction[0])
	if output == 0:
		return render_template('cancerPrediction.html',value ="Congratulations!! You are safe from cancer.Keep visiting nearby hospitals for regular checkups.  ")
	else:
		return render_template('cancerPrediction.html',value ="There are high chances of having cancer. Please visit nearby hospital soon.")

if __name__ == "__main__":
    app.run()


