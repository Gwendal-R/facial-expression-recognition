import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.mixture import GaussianMixture


class MachineLearning:
    def __init__(self, training_file):
        self.df = pd.read_csv(str(training_file), delimiter=',', header=0)

        print(self.df)

        self.X = self.normalize(self.df.drop('Target', 1))[:, :-1]
        self.Y = self.df['Target']

        self.name = None
        self.clf = None
        self.param_grid = None
        self.cv = None
        self.tree = None

        self.scoring = {
            'accuracy_macro_score': metrics.make_scorer(metrics.accuracy_score)
        }

    @staticmethod
    def normalize(df):
        return MinMaxScaler(feature_range=(0, 1)).fit_transform(df)

    def svm_classifier(self):
        self.name = "SVM"
        self.clf = svm.OneClassSVM(kernel='rbf')
        gamma_range = np.logspace(0.09819, 0.09821, 5)
        degree_range = np.logspace(5.0, 5.5, 5)
        tol = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
        self.param_grid = dict(degree=degree_range, gamma=gamma_range, tol=tol)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.33, random_state=42)

        self.learning()

    def gmm_classifier(self):
        self.name = "GMM"
        self.clf = GaussianMixture()
        covar_type = ['spherical', 'diag', 'tied', 'full']
        n_components = [i for i in range(1, 20)]
        tol = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
        self.param_grid = dict(n_components=n_components, covariance_type=covar_type, tol=tol)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.33, random_state=0)

        self.learning()

    def learning(self):
        grid = GridSearchCV(self.cv, param_grid=self.param_grid, cv=self.cv, scoring=self.scoring,
                            refit="accuracy_macro_score")
        self.tree = grid.fit(self.X, self.Y)
        print('All results from the grid search')
        print("The best parameters are " + str(grid.best_params_) + " with a score of " + str(grid.best_score_) + ".")
        print('Mean accuracy score : ' + str(grid.cv_results_['mean_test_accuracy_macro_score']))

    def save_clf(self):
        with open(self.name + '.pkl', 'wb') as file:
            pickle.dump(self.tree, file)
