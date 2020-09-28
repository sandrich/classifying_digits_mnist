"""
Algorithm class
"""
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin


class AlgorithmMeta(BaseEstimator, ClassifierMixin):
    """
    The Algorithm parent class which contains all the basic algorithm methods
    """

    def __init__(self):
        self.model = None

    def fit(self, data, targets):
        """ Fits the internal model on the given data
        :param data: the data to fit the model on
        :param targets: the output class of the given data
        :return:
        """

    def predict(self, data_to_predict):
        """
        Uses the internally trained model to predict the output of some data
        :param data_to_predict: the data to predict the output
        :return:
        """

    def load_model(self, filepath):
        """
        Loads the model from disk into the object's `model` attribute
        :param filepath: the path of the model on disk
        :return:
        """
        print("Loading model", filepath)
        self.model = pickle.load(open(filepath, 'rb'))

    def save_model(self, filepath):
        """
        Saves the trained `model` attribute to disk
        :param filepath: the destination filepath to save to disk to.
        :return:
        """
        print("Saving model as", filepath)
        pickle.dump(self.model, open(filepath, 'wb'))

    def print_results(self, cache):
        """
        Prints the results of the classification
        :param cache: the arguments to print out (varies from algorithm to algorithm)
        :return:
        """

    def display_results(self, cache):
        """
        Displays various graphs that are pertinent to the algorithm's score (such as a confusion matrix)
        :param cache: the arguments to display (varies from algorithm to algorithm)
        :return:
        """

    def run_classification(self, train_data, train_labels, test_data, test_labels, model_to_save=None,
                           model_to_load=None):
        """
        A one-hit method to run everything in one go. This method can:
         - fit or load a model from disk
         - test that model on some data
         - save the model to disk
         - print the results
         - display the results

        :param train_data:
        :param train_labels:
        :param test_data:
        :param test_labels:
        :param model_to_save: the location of a model to load
        :param model_to_load: the location where to save the model
        :return:
        """
