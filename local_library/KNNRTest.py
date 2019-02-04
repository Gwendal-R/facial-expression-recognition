import pandas as pd
import numpy as np
import math as mth

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

df = pd.read_csv('../datasets/mix.csv', na_values=['?'], header=None)

# df_with_dummies = pd.get_dummies(df)

X = df.drop(['Target'], axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = preprocessing.StandardScaler()
le = preprocessing.LabelEncoder()
train = scaler.fit_transform(X_train)
test = scaler.transform(X_test)
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# Own implemented Naive Bayes Classifier

# w => Gives more or less weight to attributes
w = [1, 1, 1, 1]

# C(X) = w0 + w1*x1 + ... + Wn*xn
C = np.dot(train, w)

PI = []
for c in np.nditer(C):
    PI.append(1 / (1 + mth.exp(-1 * c)))

# LL = np.sum(y*C-mth.ln(mth.exp(C) + 1))

# LL = 0.0
# for i in range(len(C)):
#     LL += y[i] * C[i] - mth.ln(mth.exp(C[i]) + 1)
# print(LL)

gradient = 0
for i in range(0, train.shape[0]-1):
    for j in range(0, train.shape[1]-1):
        gradient += train[i, j] * (y_train[i] - PI[i])
print(gradient)

# choisir un jeu de paramètre w au hasard


# Scikit Naive Bayes Classifier
clf.fit(train, y_train)
GaussianNB(priors=None, var_smoothing=1e-09)
y_pred = clf.predict(test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

