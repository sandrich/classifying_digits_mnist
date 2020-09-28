"""
MLP class tests
"""
import unittest
from src.mlp import MLP


class MyTestCase(unittest.TestCase):
    """unittest class to test MLP"""

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


if __name__ == '__main__':
    unittest.main()
