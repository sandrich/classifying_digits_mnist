import unittest
from src import app

class TestApp(unittest.TestCase):
    
    def test_train(self):
        self.assertRaises(app.train([], [], 0, 1, 'entropy'), ValueError)
        self.assertRaises(app.train([], [], 1, 0, 'entropy'), ValueError)
        self.assertRaises(app.train([], [], 1, 1, 'test'), ValueError)
