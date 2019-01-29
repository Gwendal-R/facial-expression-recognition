#!/bin bash

personnes="Thomas Raphael Gwendal Julien"

rm completeDataset.csv
touch completeDataset.csv trainingDataset.csv testDataset.csv devDataset.csv

for i in $personnes
do
    for j in $(ls datasets/$i/csv)
    do
        tail -n $(expr $(cat datasets/$i/csv/$j | wc -l) - 1) datasets/$i/csv/$j >> completeDataset.csv
    done
done

python createDatasets.py $(cat completeDataset.csv | wc -l)

python3 svm.py

rm trainingDataset.csv testDataset.csv devDataset.csv
