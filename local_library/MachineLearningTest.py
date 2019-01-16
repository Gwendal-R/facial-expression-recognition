import pickle
import os
import pandas as pd

from local_library.ExtractLandmarks import ExtractLandmarks
from local_library.ProcessVideo import ProcessVideo
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


class MachineLearningTest:
    def __init__(self, clf, content, normal, extract, verbose=False):
        self.tree = None
        self.load_clf(clf)

        if extract:
            process_video = ProcessVideo(extract_shape_facial_landmarks=True)

            if os.path.isdir(content):  # An image collection stored in a folder
                video = ExtractLandmarks(source_type='image_collection', file=content, save_repertory='./test_datasets',
                                         verbose=verbose)
            else:  # A video
                video = ExtractLandmarks(source_type='video', file=content, save_repertory='./test_datasets',
                                         verbose=verbose)

            video.read_file(process_video)
            video.save_stream('faciallandmarks', normal)

            self.dataframe = pd.read_csv('./test_datasets/stream/faciallandmarks.csv', delimiter=',')
        else:
            self.dataframe = pd.read_csv(content, delimiter=',')

        self.predict_data_x = self.dataframe.drop("nframe", 1)
        self.predict_data_x = self.dataframe.drop("Target", 1)
        self.predict_data_y = self.dataframe['Target']
        self.predict_data_x = self.normalize(self.predict_data_x)

        predictions = self.tree.predict(self.predict_data_x)
        print("PREDICTION : ")
        print(predictions)

        confusion, accuracy = self.show_prediction(predictions)
        print("Result for accuracy : " + str(accuracy))

    def load_clf(self, file):
        with open(file, 'rb') as clf_file:
            self.tree = pickle.load(clf_file)

    @staticmethod
    def normalize(df):
        return MinMaxScaler(feature_range=(0, 1)).fit_transform(df)

    def show_prediction(self, predictions):
        confusion = metrics.confusion_matrix(self.predict_data_y, predictions)
        accuracy = metrics.accuracy_score(self.predict_data_y, predictions, normalize=True)
        print(confusion)
        print(accuracy)
        return confusion, accuracy
