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
            _ = RandomForest(trees=0, depth=1, impurity_method='entropy')

        with self.assertRaises(ValueError):
            _ = RandomForest(trees=1, depth=0, impurity_method='entropy')

        with self.assertRaises(ValueError):
            _ = RandomForest(trees=1, depth=1, impurity_method='test')
