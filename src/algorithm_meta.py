"""
Algorithm class
"""
from numpy import sum as npsum


class AlgorithmMeta():
    """
    AlogirthmMeta class
    """

    @staticmethod
    def accuracy(labels, pred):
        """
        Returns accuracy given prediction and labels
        """
        return npsum(pred == labels) / len(labels)

    def train(self, samples, labels):
        """
        Trains the internal model on the given data
        :param samples:
        :param labels:
        :return:
        """

    def predict(self, test_data):
        """
        Uses the internally trained model to predict the output of some data
        :param test_data: the data to predict the output
        :return:
        """

    def load_model(self, filepath):
        """
        Loads its model from the desired filepath
        :param filepath: the filepath of the model to load
        :return:
        """

    def save_model(self, filepath):
        """
        Saves the trained model to the desired filepath
        :param filepath: the filepath
        """

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
         - train or load a model from disk
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
