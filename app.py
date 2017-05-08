#this code uses an ensemble of different algorithms
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
        return render_template('diabetesPrediction.html',value ="no diabetes")
    else:
        return render_template('diabetesPrediction.html',value ="diabetes")

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
        return render_template('heartDiseasePrediction.html',value ="no heart disease")
    else:
        return render_template('heartDiseasePrediction.html',value ="Heart Disease")


if __name__ == "__main__":
    app.run()


