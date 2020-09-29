"""
MLP class tests
"""
import unittest
import pickle
import os
import sys
from io import StringIO
import numpy.testing
from src.mlp import MLP
from tests.test_dataset import load_data_and_cleanup


class MLPTestCase(unittest.TestCase):
    """unittest class to test MLP"""

    @classmethod
    def setUpClass(cls):
        cls.test_model = MLP(hidden_layer_sizes=(28, 28), max_iter=20)
        data, labels = load_data_and_cleanup("train")
        cls.train_data = data[:500]
        cls.train_labels = labels[:500]
        cls.test_data = data[501:511]
        cls.test_labels = labels[501:511]

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test/mlp.model"):
            os.remove("tests/mlp.model")
        if os.path.exists("tests/exp_cache.pkl"):
            os.remove("tests/exp_cache.pkl")

    def set_up_trained_model(self, save_cache=False, save_model=False):
        """Runs a quick training of the model in case it is needed. can also save the model or the output"""
        self.test_model.fit(self.train_data, self.train_labels)
        if save_cache:
            # Save this to an output, will be used for future tests.
            with open("tests/exp_cache.pkl", "wb") as pickle_dump:
                pickle.dump(self.test_model.eval_train_test_cache(
                    self.train_data, self.train_labels, self.test_data, self.test_labels
                ), pickle_dump)
                pickle_dump.close()

        if save_model:
            self.test_model.save_model("tests/mlp.model")

    def test_training(self):
        """Test training with no errors (run first)"""
        raised = False
        try:
            self.test_model.fit(self.train_data, self.train_labels)
        except Exception as error:
            raised = True
            print(error)
        self.assertFalse(raised)

    def test_save_model(self):
        """Test if saving the model works without errors"""
        raised = False
        try:
            self.set_up_trained_model()
            self.test_model.save_model('tests/mlp.model')
        except Exception as error:
            print(error)
            raised = True
        self.assertFalse(raised)

    def test_hidden_layer_empty(self):
        """Test empty hidden layer config"""
        with self.assertRaises(ValueError):
            _ = MLP(hidden_layer_sizes=())

    def test_hidden_layer_negative(self):
        """Test negative neuron count in hidden layers config"""
        with self.assertRaises(ValueError):
            _ = MLP(hidden_layer_sizes=(-20, 50))

    def test_max_iter_negative(self):
        """Test negative iterations value"""
        with self.assertRaises(ValueError):
            _ = MLP(max_iter=-1)

    def test_batch_size_negative(self):
        """Test negative batch size value"""
        with self.assertRaises(ValueError):
            _ = MLP(batch_size=-1)

    def test_batch_size_str(self):
        """Test batch size other than 'auto'"""
        with self.assertRaises(ValueError):
            _ = MLP(batch_size="test")

    def test_no_errors(self):
        """Test if no errors pass"""
        raised = False
        try:
            _ = MLP()  # using all the default values
        except ValueError:
            raised = True
        self.assertFalse(raised)

    def test_predictions(self):
        """test predictions (it's bad, but we don't care)"""
        predictions = self.test_model.predict(self.test_data)
        numpy.testing.assert_almost_equal(predictions, [2, 8, 2, 3, 7, 6, 1, 1, 9, 1])

    def test_accuracy(self):
        """Test accuracy of predictions"""
        self.set_up_trained_model()
        accuracy = self.test_model.score(self.test_data, self.test_labels)
        self.assertEqual(accuracy, 0.5)

    def test_print_results(self):
        """Testing the printed results"""
        self.set_up_trained_model(save_cache=True)
        old_stdout = sys.stdout
        sys.stdout = my_stdout = StringIO()
        self.test_model.print_results(pickle.load(open("tests/exp_cache.pkl", "rb")))
        sys.stdout = old_stdout
        calculated = my_stdout.getvalue()
        self.assertEqual(calculated,
                          "Classification stats:\n"
                          "-----------------\n"
                          "Train Accuracy: 0.802\n"
                          "Test  Accuracy: 0.500\n"
                          "Training time : {0:.2f}s\n"
                          "-----------------\n"
                          "Hidden-layer neuron count: 28, 28\n"
                          "Alpha: 0.0001\n"
                          "Batch size: auto\n"
                          "Max training iterations: 20\n".format(self.test_model.model.train_time))

    def test_loaded_model(self):
        """Test if loading the model works and gives the same predictions as before."""
        self.set_up_trained_model(save_model=True)
        self.test_model.load_model("tests/mlp.model")
        numpy.testing.assert_almost_equal(self.test_model.predict(self.test_data), [2, 8, 2, 3, 7, 6, 1, 1, 9, 1])


if __name__ == '__main__':
    unittest.main()
