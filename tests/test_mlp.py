import unittest
from src.mlp import MLP


class MyTestCase(unittest.TestCase):

    def test_hidden_layer_empty(self):
        with self.assertRaises(ValueError):
            _ = MLP(hidden_layer_sizes=())

    def test_hidden_layer_negative(self):
        with self.assertRaises(ValueError):
            _ = MLP(hidden_layer_sizes=(-20, 50))

    def test_max_iter_negative(self):
        with self.assertRaises(ValueError):
            _ = MLP(max_iter=-1)

    def test_batch_size_negative(self):
        with self.assertRaises(ValueError):
            _ = MLP(batch_size=-1)

    def test_batch_size_str(self):
        with self.assertRaises(ValueError):
            _ = MLP(batch_size="test")

    def test_no_errors(self):
        _ = MLP()  # using all the default values
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
