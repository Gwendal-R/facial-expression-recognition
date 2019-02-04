import pandas as pd
import numpy as np
import pickle

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


class MachineLearning:
    def __init__(self, file1, file2=None):
        self.dataframe1 = pd.read_csv(str(file1), delimiter=',')
        self.dataframe1.drop("nframe", 1)

        self.train = self.dataframe1
        train_dataset = self.normalize(self.train)

        self.x_train = train_dataset[:, :-1]
        self.y_train = self.train['Target']

        if file2 is not None:
            self.dataframe2 = pd.read_csv(str(file2), delimiter=',')
            self.dataframe2.drop("nframe", 1)

        self.test = self.dataframe2
        test_dataset = self.normalize(self.test)
        self.x_test = test_dataset[:, :-1]
        self.y_test = self.test['Target']

        self.clf = None
        self.tree = None
        self.name = None
        self.param_grid = None
        self.cv = None

        self.scoring = {
            'accuracy_macro_score': metrics.make_scorer(metrics.accuracy_score),
        }

    def normalize(self, df):
        return MinMaxScaler(feature_range=(0, 1)).fit_transform(df)

    def svm_classifier(self):
        """Utilisation du classifier OneClassSVM de sklearn  :
            http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
            voir pour utiliser une grid
        """

        self.name = "SVM"
        self.clf = svm.OneClassSVM(kernel='rbf')
        gamma_range = np.logspace(0.09819, 0.09821, 5)
        degree_range = np.logspace(5.0, 5.5, 5)
        tol = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
        self.param_grid = dict(degree=degree_range, gamma=gamma_range, tol=tol)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.33,
                                         random_state=42)
        self.learning()

    def gmm_classifier(self):
        """Utilisation du classifier GaussianMixture de sklearn  :
        http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        """
        self.name = "GMM"
        self.clf = GaussianMixture()
        covar_type = ['spherical', 'diag', 'tied', 'full']
        n_components = [i for i in range(1, 20)]
        tol = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
        self.param_grid = dict(n_components=n_components,
                               covariance_type=covar_type, tol=tol)
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.33,
                                         random_state=0)
        self.learning()

    def knnr_classifier(self):
        """Utilisation du classifier KNeighborsRegressor de sklearn  :
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
        """
        self.name = "KNNR"
        self.clf = KNeighborsRegressor()

        n_neighbors = [i for i in range(2, 5)]
        weights = ['uniform', 'distance']
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size = [i for i in range(5, 20)]
        self.param_grid = dict(n_neighbors=n_neighbors, weights=weights,
                               algorithm=algorithm, leaf_size=leaf_size)

        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.33,
                                         random_state=0)
        self.learning()

    def learning(self):
        grid = GridSearchCV(self.clf, param_grid=self.param_grid, cv=self.cv, scoring=self.scoring,
                            refit="accuracy_macro_score")

        self.tree = grid.fit(self.x_train, self.y_train)

        print('All results from the grid search')
        print("The best parameters are " + str(grid.best_params_) + " with a score of " + str(grid.best_score_) + ".")
        print('Mean accuracy score : ' + str(grid.cv_results_['mean_test_accuracy_macro_score']))

        return self.tree, self.clf

    def save_clf(self):
        with open(self.name + '.pkl', 'wb') as fid:
            pickle.dump(self.tree, fid)

    def predict(self, file):
        dataframe = pd.read_csv(file, delimiter=',')
        predict_data_x = self.normalize(dataframe.drop("nframe", 1).drop("Target", 1))
        predict_data_y = dataframe['Target']

        predictions = self.tree.predict(predict_data_x)
        print("PREDICITONS : ")
        print(predictions)
        confusion = metrics.confusion_matrix(predict_data_y, predictions)
        print(confusion)
        accuracy = metrics.accuracy_score(predict_data_y, predictions, normalize=True)
        print("Result for accuracy : " + str(accuracy))
