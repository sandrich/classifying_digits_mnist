"""
AlgorithmMeta
"""
import pickle
import os
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from pandas import DataFrame, read_csv, concat
from .visualizer import display_train_test_matrices


class AlgorithmMeta(BaseEstimator, ClassifierMixin):
    """
    The Algorithm parent class which contains all the basic algorithm methods. Most of the logic of the algorithm is
    done here. Indeed, other than setting up the algorithm to match given specs (like number of trees or hidden layers)
    the train/test mechanics is the same.
    """

    def __init__(self, report_directory: str = None, test_suite_iter: int = None):
        self.model = None
        self.report_directory = report_directory
        self.test_suite_iter = test_suite_iter

    def _check_loaded_model_and_set_conf(self, props: list):
        """
        Warns the user if there are any discrepancies between the configuration and loaded model's parameters.

        If a use enters for example ``python mnist.py --load_RF model.rf --n_estimators 200`` but ``model.rf``
        has only 100 estimators, this method will warn the user that the loaded model does not have the same
        configuration as the user inputted. It will also set the object's
        properties to reflect that of the loaded model.

        Parameters
        ----------
        props
            The list of properties to check between the model and the object.
            This means that they must have the same names.
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
        """
        Fits the internal model on the given data, and returns it

        Parameters
        ----------
        data : numpy.array
            the data on which you want to fit
        targets : numpy.array
            the target classes of the training data you want to fit

        Returns
        -------
        sklearn.BaseEstimator: The trained model
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

        Parameters
        ----------
        data_to_predict :numpy.array
            Sample data set on which to generate predictions

        Returns
        -------
        numpy.array: Array with the predicted class label
        """
        return self.model.predict(data_to_predict)

    def load_model(self, filepath):
        """
        Loads the model from disk into the object's ``model`` attribute

        Parameters
        ----------
        filepath : str
            the path of the model on disk
        """
        print("Loading model", filepath)
        self.model = pickle.load(open(filepath, 'rb'))

    def save_model(self, filepath):
        """
        Saves the trained `model` attribute to disk

        Parameters
        ----------
        filepath : str
            the destination filepath to save to disk to.
        """
        print("Saving model as", filepath)
        if self.report_directory is not None:
            filepath = os.path.join(self.report_directory, filepath)
        pickle.dump(self.model, open(filepath, 'wb'))

    @staticmethod
    def calc_standard_error(error: float, sample_count: int):
        """
        Calculates the Wilson score interval with 95% confidence, based on the paper by Edwin B. Wilson [#]_.

        .. [#] Edwin B. Wilson (1927) Probable Inference, the Law of Succession, and Statistical Inference,
           Journal of the American Statistical Association

        We interpret the results as the +/- of the error rate of the algorithm. For example,
        an error rate of 0.02 with 50 samples and a confidence of 95% yields 0.0388. So the error rate can be
        read as 0.02 +/- 0.0388.

        Parameters
        ----------
        error : float
            the error rate of the test results
        sample_count : int
            the number of test samples used

        Returns
        -------
        float
            the standard error
        """
        return 1.96 * (error * (1.-error)/sample_count)**0.5  # using 1.96 as a constant for 95% confidence interval

    def print_results(self, cache):
        """
        Prints the results of the classification, and returns them as a pandas DataFrame

        Parameters
        ----------
        cache : dict
            the cache of a ``run_classification()`` function call.

        Returns
        -------
        pandas.DataFrame:
            the classification results as a single-line data frame
        """
        standard_error = AlgorithmMeta.calc_standard_error(1-cache['accuracy']['test'], len(cache['actual']['test']))

        print('Classification stats:')
        print('-----------------')
        print('Train Accuracy: {0:.3f}'.format(cache['accuracy']['train']))
        print('Test  Accuracy: {0:.3f}'.format(cache['accuracy']['test']))
        print('Test Standard Error: {0:.3f}'.format(standard_error))
        print('Training time : {0:.2f}s'.format(self.model.train_time))

        out = DataFrame.from_dict({
            "Algorithm": [self.__class__.__name__],
            "Train accuracy": [cache['accuracy']['train']],
            "Test accuracy": [cache['accuracy']['test']],
            "Standard Error": [standard_error],
            "Training time (s)": [self.model.train_time]
        })
        if self.test_suite_iter is not None:
            out.reindex([self.test_suite_iter])
        return out

    def display_results(self, cache):
        """
        Displays various graphs that are pertinent to the algorithm's score (such as a confusion matrix)

        Parameters
        ----------
        cache : dict
            the arguments to display (varies from algorithm to algorithm)
        """
        if self.report_directory is None:
            test_matrices_out = None
        elif self.test_suite_iter is None:
            test_matrices_out = os.path.join(self.report_directory, "confusion_matrices.png")
        else:
            test_matrices_out = os.path.join(self.report_directory, f"{self.test_suite_iter}_confusion_matrices.png")

        display_train_test_matrices(cache, save_location=test_matrices_out)

    def eval_train_test_cache(self, train_data, train_labels,
                              test_data, test_labels):
        """
        Generates a cache object containing the test and test data, labels, and accuracies, and a copy of the model.

        Parameters
        ----------
        train_data : numpy.array
            the raw training data
        train_labels : numpy.array
            the ground truth of the training data
        test_data : numpy.array
            the data to test against
        test_labels : numpy.array
            the ground truth of the test data

        Returns
        -------
        dict: the actual data, predicted data, accuracy, and model in a dict format

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

    def save_results(self, results: DataFrame):
        """
        Saves the results to disk as a CSV file if the report_directory is not None. If the output report file already
        exists, it will have lines appended to it

        Parameters
        ----------
        results : pandas.DataFrame
            the results table to save to disk.
        """
        if self.report_directory is None or not os.path.exists(self.report_directory):
            return
        path = os.path.join(self.report_directory, "report.csv")
        if os.path.exists(path):
            existing = read_csv(path, index_col="iteration")
            results = concat([existing, results], axis=0,ignore_index=True)
        results.to_csv(path, index=(self.test_suite_iter is not None), index_label="iteration")

    def run_classification(self, train_data, train_labels, test_data, test_labels,
                           model_to_save=None, model_to_load=None):
        """
        Trains and tests the classification

        Parameters
        ----------
        train_data : numpy.array
            the data to train on
        train_labels :numpy.array
            the labels of the train data
        test_data : numpy.array
            the data to use to run predictions
        test labels : numpy.array
            the ground truth of the test data
        model_to_load : str
            filepath of a saved model to load instead of train
        model_to_save : str
            filepath on which to save the trained model

        Returns
        -------
        dict: Returns collection with prediction and accuracy
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

        self.save_results(self.print_results(cache))
        self.display_results(cache)
        return cache
