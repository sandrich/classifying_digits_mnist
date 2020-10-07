"""
Master Class for tests testing algorithms
"""
from io import StringIO
import os
import pickle
import shutil
import sys
from mnist_classifier.algorithm_meta import AlgorithmMeta
from .test_dataset import load_data_and_cleanup
TEST_FOLDER = os.path.join("reports", "test_outputs")
TEST_MODEL_LOCATION = "test.model"
TEST_CACHE_LOCATION = "cache.pkl"

EXP_PRINT_OUTPUT_BASE = "Classification stats:\n"\
                          "-----------------\n"\
                          "Train Accuracy: {0:.3f}\n"\
                          "Test  Accuracy: {1:.3f}\n"\
                          "Test Standard Error: {2:.3f}\n"\
                          "Training time : {3:.2f}s\n"\
                          "-----------------\n"


class AlgorithmTestMaster:
    """Master Test class for algorithm tests (contains overarching methods and tests)"""

    def __init__(self):
        self.test_model: AlgorithmMeta = None
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

    def load_test_datasets(self):
        """Prepare all the datasets"""
        data, labels = load_data_and_cleanup("train")
        self.train_data = data[:500]
        self.train_labels = labels[:500]
        self.test_labels = labels[501:511]
        self.test_data = data[501:511]
        if not os.path.exists("reports"):
            os.mkdir("reports")
        if not os.path.exists(TEST_FOLDER):
            os.mkdir(TEST_FOLDER)

    def tear_down_class(self):
        """Removes any created models, cache files, folders, or images from disk"""
        if os.path.exists(TEST_FOLDER):
            shutil.rmtree(TEST_FOLDER)
        for file in [TEST_MODEL_LOCATION, TEST_CACHE_LOCATION]:
            if os.path.exists(file):
                os.remove(file)

    def set_up_trained_model(self, save_cache=False, save_model=False):
        """Runs a quick training of the model in case it is needed. can also save the model or the output"""
        self.test_model.fit(self.train_data, self.train_labels)
        if save_cache:
            # Save this to an output, will be used for future tests.
            with open(TEST_CACHE_LOCATION, "wb") as pickle_dump:
                pickle.dump(self.test_model.eval_train_test_cache(
                    self.train_data, self.train_labels, self.test_data, self.test_labels
                ), pickle_dump)
                pickle_dump.close()
        if save_model is not None:
            self.test_model.save_model(TEST_MODEL_LOCATION)

    def test_training(self):
        """just testing that our wrapper class works, no need to check the results. Sklearn took care of that"""
        raised = False
        try:
            self.test_model.fit(self.train_data, self.train_labels)
        except Exception as error:
            raised = True
            print(error)
        assert not raised

    def test_predictions(self):
        """just testing that our wrapper class works, no need to check the results. Sklearn took care of that"""
        self.set_up_trained_model()
        raised = False
        try:
            self.test_model.predict(self.test_data)
        except Exception as error:
            print(error)
            raised = True
        assert not raised

    def test_accuracy(self):
        """Test accuracy of predictions works (no need to check the actual value"""
        self.set_up_trained_model()
        raised = False
        try:
            self.test_model.score(self.test_data, self.test_labels)
        except Exception as error:
            print(error)
            raised = True
        assert not raised

    def predict_and_print(self):
        """Runs a prediction, and captures the print results output"""
        self.set_up_trained_model(save_cache=True)
        old_stdout = sys.stdout
        sys.stdout = my_stdout = StringIO()
        self.test_model.print_results(pickle.load(open(TEST_CACHE_LOCATION, "rb")))
        sys.stdout = old_stdout
        return my_stdout.getvalue()

    def test_save_model(self):
        """Test if saving the model works without errors"""
        raised = False
        try:
            self.set_up_trained_model()
            self.test_model.save_model(TEST_MODEL_LOCATION)
        except Exception as error:
            print(error)
            raised = True
        assert not raised

    def test_loaded_model(self):
        """Test if loading the model works without errors and is not none."""
        raised = False
        try:
            self.set_up_trained_model(save_model=True)
            self.test_model.load_model(os.path.join(TEST_FOLDER, TEST_MODEL_LOCATION))
        except Exception as error:
            print(error)
            raised = True
        assert self.test_model is not None and not raised

    def test_run_classification(self):
        """
        Tests that the run_classification() runs without any errors
        (all the individual methods are already tested)
        """
        raised = False
        self.set_up_trained_model()
        try:
            self.test_model.run_classification(
                train_data=self.train_data,
                train_labels=self.train_labels,
                test_data=self.test_data,
                test_labels=self.test_labels)
        except Exception as error:
            print(error)
            raised = True
        assert not raised
