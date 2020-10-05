"""
Algorithm class
"""
import pickle
import os
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from .visualizer import display_train_test_matrices

class AlgorithmMeta(BaseEstimator, ClassifierMixin):
    """
    The Algorithm parent class which contains all the basic algorithm methods
    """

    def __init__(self):
        self.model = None

    def _check_loaded_model_and_set_conf(self, props: list):
        """
        Warns the user if there are any discrepancies between the configuration and loaded model's parameters.

        If a use enters for example `python mnist.py --load_RF model.rf --n_estimators 200` but `model.rf` has only 100
        estimators, this method will warn the user that the loaded model does not have the same configuration as
        the user inputted. It will also set the object's properties to reflect that of the loaded model.

        :param props: The list of properties to check between the model and the object. This means that they must have
        the same names.
        :return:
        """
        for prop in props:
            if prop in self.__dict__ and \
               self.__dict__[prop] is not None and \
               self.__dict__[prop] != self.model.__dict__[prop]:
                print(f"[WARNING] - the model you loaded has {self.model.__dict__[prop]} as {prop}, "
                      f"but you specified {self.__dict__[prop]}! Continuing with loaded model...")

            elif prop not in self.__dict__ or self.__dict__[prop] is None:
                self.__dict__[prop] = self.model.__dict__[prop]

    def fit(self, data, targets):
        """ Fits the internal model on the given data
        :param data: the data to fit the model on
        :param targets: the output class of the given data
        :return:
        """

        print(f"Starting training {self.__class__.__name__}...")
        time_start = time.time()
        self.model.fit(data, targets)
        self.model.train_time = time.time() - time_start
        print("Done training.")
        return self.model

    def predict(self, data_to_predict):
        """
        Returns prediction of the class y for input

        :param
            classifier: Class instance with the trained random forest
            data_to_predict: Sample data set

        :return
            Array with the predicted class label
        """
        return self.model.predict(data_to_predict)

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
        print('Classification stats:')
        print('-----------------')
        print('Train Accuracy: {0:.3f}'.format(cache['accuracy']['train']))
        print('Test  Accuracy: {0:.3f}'.format(cache['accuracy']['test']))
        print('Training time : {0:.2f}s'.format(self.model.train_time))

    def display_results(self, cache, save_directory):
        """
        Displays various graphs that are pertinent to the algorithm's score (such as a confusion matrix)
        :param save_directory: where to save the plots. If None, the plots will be displayed at runtime
        :param cache: the arguments to display (varies from algorithm to algorithm)
        :return:
        """
        if save_directory is not None and not os.path.exists(save_directory):
            os.mkdir(save_directory)
        test_matrices_out = None if save_directory is None else os.path.join(save_directory, "confusion_matrices.png")
        display_train_test_matrices(cache, save_location=test_matrices_out)

    def eval_train_test_cache(self, train_data, train_labels,
                              test_data, test_labels):
        """
        Generates a cache object containing the test and test data, labels, and accuracies, and a copy of the model.
        :param train_data:
        :param train_labels:
        :param test_data:
        :param test_labels:
        :return:
        """
        train_pred = self.predict(train_data)
        test_pred = self.predict(test_data)

        train_acc = self.score(train_data, train_labels)  # score() inherited from sklearn.base.ClassifierMixin
        test_acc = self.score(test_data, test_labels)

        return {
            'prediction': {
                'train': train_pred,
                'test': test_pred
            }, 'accuracy': {
                'train': train_acc,
                'test': test_acc
            }, 'actual': {
                'train': train_labels,
                'test': test_labels
            }, 'model': self.model}

    def run_classification(self, train_data, train_labels, test_data, test_labels,
                           model_to_save=None, model_to_load=None, save_directory: str = None):
        """
        Trains and tests the classification

        :param
            train_data: Train sample set
            train_labels: Train y
            X: Test data set
            test_labels: Test y
            save_model: filepath to save the trained model
            model_to_load: filepath of saved model to load instead of training
            save_directory: Where to save the output figures. If None, the figures will be displayed at runtime


        :return
            Returns collection with prediction and accuracy
            cache:
                prediction:
                    train: float
                    test: float
                accuracy:
                    train: float
                    test: float
        """
        if model_to_load is not None and os.path.exists(model_to_load):
            self.load_model(model_to_load)
        else:
            if model_to_load is not None:
                print(f"Could not find the model {model_to_load}... training a new model.")
            self.fit(train_data, train_labels)

        if model_to_save is not None:
            self.save_model(model_to_save)

        cache = self.eval_train_test_cache(train_data, train_labels, test_data, test_labels)

        self.print_results(cache)
        self.display_results(cache, save_directory)
        return cache
