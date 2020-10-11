"""
Testing entrypoint
"""
import os
import sys
import unittest
from shutil import rmtree
from unittest.mock import patch
from mnist_classifier import predict
sys.path.append('..')

class AttrDict(dict):
    """
    Mocks a callable dict
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

class TestEntrypoint(unittest.TestCase):
    """
    Tests around the entrypoint
    """
    def setUp(self):
        """
        Removes report folder
        """
        rmtree("reports", True)

    def test_pa_and_run_exit(self):
        """
        Tests if program exists with exit code 1
        """
        with self.assertRaises(SystemExit) as exp:
            args = AttrDict()
            args.update({'save':1, 'load': 1})
            predict.process_args_and_run(args)

            self.assertEqual(exp.exception.code, 1)

    def test_load_report_directory(self):
        """
        Tests if report directory is being set
        """
        args = AttrDict()
        args.update({'report_directory':'foo'})
        self.assertEqual(predict.load_report_directory(args),
                         {'report_directory': os.path.join('reports','foo')})

    def test_hidden_layer_value(self):
        """
        Checks that hidden layer value is an integer
        """
        with self.assertRaises(SystemExit) as exp:
            args = AttrDict()
            args.update({'save': True, 'load': None, 'alpha':1, 'hidden_layers': "1 1.1", 'max_iter': 1, 'verbose': True, 'report_directory': 'test'})
            predict.process_args_and_run(args)
            self.assertEqual(exp.exception.code, 1)

    @patch('mnist_classifier.predict.RandomForest.run_classification')
    def test_rf_being_called(self, mock):
        """
        Checks if rf classifier is being called
        """
        args = AttrDict()
        args.update({'save': True, \
            'report_directory': 'test',\
            'load': None, \
            'depth': 1,\
            'impurity_method': 'gini',\
            'trees': 1})
        predict.process_args_and_run(args)
        mock.assert_called_once()

    @patch('mnist_classifier.predict.MLP.run_classification')
    def test_mlp_being_called(self, mock):
        """
        Checks if mlp classifier is being called
        """
        args = AttrDict()
        args.update({'save': True, \
            'load': None, \
            'hidden_layers': "1", \
            'verbose': True, \
            'report_directory': 'test', \
            'batch_size': 1, \
            'max_iter': 1, \
            'random_seed': 1,\
            'alpha': 1})
        predict.process_args_and_run(args)
        mock.assert_called_once()

    def test_parse_args(self):
        """
        Tests if arguments are properly parsed
        """
        parser = predict.parse_args(['-rs', '1', '-rd', 'test', '-s', 'file', '-l', 'file',\
            '-ts', 'rf'])
        self.assertEqual(parser.random_seed, 1)
        self.assertEqual(parser.report_directory, 'test')
        self.assertEqual(parser.save, 'file')
        self.assertEqual(parser.load, 'file')
        self.assertEqual(parser.test_suite, 'rf')

        # Test rf sub commands
        parser = predict.parse_args(['rf', '-t', '1', '-d', '1', '-i', 'gini'])
        self.assertEqual(parser.trees, 1)
        self.assertEqual(parser.depth, 1)
        self.assertEqual(parser.impurity_method, 'gini')

        # Test mlp sub commands
        parser = predict.parse_args(['mlp', '-a', '1', '-b', '1', '-i', '1', '-v'])
        self.assertEqual(parser.alpha, 1)
        self.assertEqual(parser.batch_size, '1')
        self.assertEqual(parser.max_iter, 1)
        self.assertTrue(parser.verbose)
