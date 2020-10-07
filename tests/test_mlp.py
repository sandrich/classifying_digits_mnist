"""
MLP class tests
"""
import unittest
from mnist_classifier.mlp import MLP
from .algo_test_master import AlgorithmTestMaster, EXP_PRINT_OUTPUT_BASE, TEST_FOLDER


class MLPTestCase(unittest.TestCase, AlgorithmTestMaster):
    """unittest class to test MLP"""

    @classmethod
    def setUpClass(self):
        super().load_test_datasets(self)
        self.test_model = MLP(hidden_layer_sizes=(28, 28), max_iter=20, report_directory=TEST_FOLDER, random_seed=12345)

    @classmethod
    def tearDownClass(self):
        super().tear_down_class(self)

    def test_setup_errors(self):
        """Test all the MLP setup errors"""
        # Test empty hidden layer config
        with self.assertRaises(ValueError):
            _ = MLP(hidden_layer_sizes=())

        # Test negative neuron count in hidden layers config
        with self.assertRaises(ValueError):
            _ = MLP(hidden_layer_sizes=(-20, 50))

        # Test negative iterations value
        with self.assertRaises(ValueError):
            _ = MLP(max_iter=-1)

        # Test negative batch size value
        with self.assertRaises(ValueError):
            _ = MLP(batch_size=-1)

        # Test batch size other than 'auto'
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

    def test_print_results(self):
        """Tests the print output of the algorithm's print_results() function"""
        calculated = super().predict_and_print()
        self.assertEqual(calculated, EXP_PRINT_OUTPUT_BASE.format(.802, .5, 0.31, self.test_model.model.train_time) +
                         "Hidden-layer neuron count: 28, 28\n"
                         "Alpha: 0.0001\n"
                         "Batch size: auto\n"
                         "Max training iterations: 20\n")
