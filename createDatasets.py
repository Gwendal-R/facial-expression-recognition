import csv
import sys
from random import randint

completeDataset = csv.reader(open("completeDataset.csv"))

trainingDataset = csv.writer(open("trainingDataset.csv","w"))
devDataset = csv.writer(open("devDataset.csv","w"))
testDataset = csv.writer(open("testDataset.csv","w"))

n = int(sys.argv[1])

dataset = []

for row in completeDataset:
    dataset.append(row)

n = len(dataset) - 1
i = 0

while n > 0:
	if i < 0.65*len(dataset):
		rand = randint(0,n)
		trainingDataset.writerow(dataset[rand])
		del dataset[rand]
	elif i >= 0.65*len(dataset) and i < 0.75*len(dataset):
		rand = randint(0,n)
		devDataset.writerow(dataset[rand])
		del dataset[rand]
	else:
		rand = randint(0,n)
		testDataset.writerow(dataset[rand])
		del dataset[rand]
	n -= 1
	i += 1
