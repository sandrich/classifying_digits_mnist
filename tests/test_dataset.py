import sys
import unittest
import numpy as np
import os
sys.path.append('..')
from src import dataset


def load_data_and_cleanup(which: str = "train"):
    if which == "train":
        data, labels = dataset.load_train_data()
    elif which == "test":
        data, labels = dataset.load_test_data()
    else:
        raise AttributeError("enter either 'train' or 'test'")
    # cleanup any downloaded files
    os.unlink(f"{which}_data.npy")
    os.unlink(f"{which}_labels.npy")
    os.unlink(f"{which}_dims.npy")

    return data, labels


def load_expected(which: str):
    """depending on where you run the script from, the expected files are either in the local folder or in tests/"""
    try:
        return np.load(f"tests/{which}")
    except FileNotFoundError:
        return np.load(which)


class TestDataset(unittest.TestCase):

    def test_loadTrainData(self):
        data, _ = load_data_and_cleanup('train')
        expected = load_expected('exp_train_data.npy')
        expected = expected.astype(int)
        np.testing.assert_almost_equal(data[0], expected)

    def test_loadTrainLabels(self):
        _, labels = load_data_and_cleanup('train')

        expected = load_expected('exp_train_labels.npy')
        expected = expected.astype(int)
        np.testing.assert_almost_equal(labels[0], expected)

    def test_loadTestData(self):
        data, _ = load_data_and_cleanup('test')

        expected = load_expected('exp_test_data.npy')
        expected = expected.astype(int)
        np.testing.assert_almost_equal(data[0], expected)

    def test_loadTestLabels(self):
        _, labels = load_data_and_cleanup('test')
        expected = load_expected('exp_test_labels.npy')
        expected = expected.astype(int)
        np.testing.assert_almost_equal(labels[0], expected)
