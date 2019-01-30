import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import *

# Load data
with open ("trainingDataset.csv") as trainingFile:
	trainingCsv = csv.reader(trainingFile, delimiter=',')
	X_training=[]
	Y_training=[]
	for row in trainingCsv:
		l=[]
		for i in range(138):
			l.append(row[i])
		X_training.append(l)
		Y_training.append(row[139])

with open("testDataset.csv") as testFile:
	testCsv = csv.reader(testFile, delimiter=',')
	X_test=[]
	Y_test=[]
	for row in testCsv:
		l=[]
		for i in range(138):
			l.append(row[i])
		X_test.append(l)
		Y_test.append(row[139])

clf = svm.SVC(gamma='scale')
clf.fit(X_training,Y_training)
Pred = clf.predict(X_test)
s = 0.
n = 0
for i in range(len(Y_test)):
	n += 1
	if Y_test[i] == Pred[i]:
		s += 1
print(s/n)
