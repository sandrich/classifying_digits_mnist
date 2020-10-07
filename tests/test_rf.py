"""
Testing the Random Forest Algorithm
"""
import unittest
from mnist_classifier.random_forest import RandomForest
from .algo_test_master import AlgorithmTestMaster, EXP_PRINT_OUTPUT_BASE, TEST_FOLDER


class RFTestCase(unittest.TestCase, AlgorithmTestMaster):
    """
    Testing the app
    """

    @classmethod
    def setUpClass(self):
        """Set up superclass"""
        super().load_test_datasets(self)
        self.test_model = RandomForest(n_estimators=1, max_depth=1,
                                       criterion="entropy", report_directory=TEST_FOLDER, random_seed=12345)

    @classmethod
    def tearDownClass(self):
        super().tear_down_class(self)

    def test_setup_errors(self):
        """Tests errors when setting up the model"""
        with self.assertRaises(ValueError):
            _ = RandomForest(n_estimators=0, max_depth=1, criterion='entropy')

        with self.assertRaises(ValueError):
            _ = RandomForest(n_estimators=1, max_depth=0, criterion='entropy')

        with self.assertRaises(ValueError):
            _ = RandomForest(n_estimators=1, max_depth=1, criterion='test')

    def test_no_errors(self):
        """Tests the correct creation of the model without errors"""
        raised = False
        try:
            _ = RandomForest(n_estimators=1, max_depth=1, criterion="entropy")
        except Exception as error:
            print(error)
            raised = True
        self.assertFalse(raised)

    def test_print_results(self):
        """Tests the printout method of the algorithm"""
        calculated = super().predict_and_print()
        self.assertEqual(calculated, EXP_PRINT_OUTPUT_BASE.format(.18, .1, 0.186, self.test_model.model.train_time) +
                         "Max tree max_depth: 1\n"
                         "Number of n_estimators: 1\n"
                         "Impurity method: entropy\n")
