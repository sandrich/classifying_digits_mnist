import unittest
from src import app

class TestApp(unittest.TestCase):
    
    def test_train(self):
        with self.assertRaises(ValueError):
            _ = app.train([[1]], [1], 0, 1, 'entropy')
        
        with self.assertRaises(ValueError):
            _ = app.train([[1]], [1], 1, 0, 'entropy')

        with self.assertRaises(ValueError):
            _ = app.train([[1]], [1], 1, 1, 'test')
