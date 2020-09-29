"""
Testing the app
"""
import unittest
from src.random_forest import RandomForest


class TestApp(unittest.TestCase):
    """
    Testing the app
    """
    def test_train(self):
        """
        Tests around training the model
        """
        with self.assertRaises(ValueError):
            _ = RandomForest(n_estimators=0, max_depth=1, criterion='entropy')

        with self.assertRaises(ValueError):
            _ = RandomForest(n_estimators=1, max_depth=0, criterion='entropy')

        with self.assertRaises(ValueError):
            _ = RandomForest(n_estimators=1, max_depth=1, criterion='test')
