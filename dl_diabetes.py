from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)


# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
dataset1 = numpy.loadtxt("user_input1.csv", delimiter=",")
#dataset2 = numpy.loadtxt("user_input2.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
#print X
#print "Y"
#print Y

U = dataset1[:,0:8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)

# calculate predictions
print "\n"
print "predictions on user input is below:"
print "\n"
print "Values enetered by 2 users are:"
print "[Pregnancy_count,plasma_glucose,BP,skin_thickness,serum_insulin,BMI,pedigree_function, age]"
print U
predictions1 = model.predict(U)
# round predictions
rounded = [round(x[0]) for x in predictions1]
print "\n"
print "Prediction Results for the users: O-No & 1-Yes"
print(rounded)
print "\n"
