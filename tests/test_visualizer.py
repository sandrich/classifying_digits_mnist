"""
Visualizer tests

These tests don't actually test the content of the matplotlib plot, but they just check that the plots are created,
saved in the correct spot, and that they have the right subplots, titles, and axes.
"""
import unittest
import os
from mnist_classifier.random_forest import RandomForest
from mnist_classifier.mlp import MLP
from mnist_classifier.visualizer import display_mlp_coefficients, \
    display_loss_curve, \
    display_train_test_matrices, \
    display_rf_feature_importance
from .test_dataset import load_data_and_cleanup
TEST_IMAGE_LOCATION = "tests/test_image.png"


class VisualizerTestCase(unittest.TestCase):
    """
    Testing all the visual components of the app
    """
    @classmethod
    def setUpClass(self):
        data, labels = load_data_and_cleanup("train")

        train_data = data[:500]
        train_labels = labels[:500]
        test_labels = labels[501:511]
        test_data = data[501:511]

        rf_model = RandomForest(n_estimators=1, max_depth=1, criterion="entropy")
        self.mlp_model = MLP(hidden_layer_sizes=(10,), max_iter=10)

        rf_model.fit(train_data, train_labels)
        self.rf_cache = rf_model.eval_train_test_cache(train_data, train_labels, test_data, test_labels)

        self.mlp_model.fit(train_data, train_labels)
        self.mlp_cache = self.mlp_model.eval_train_test_cache(train_data, train_labels, test_data, test_labels)

    def tearDown(self):
        if os.path.exists(TEST_IMAGE_LOCATION):
            os.remove(TEST_IMAGE_LOCATION)

    def test_confusion_matrices(self):
        """ Test the two confusion matrices, checks that the subplots are created and that the output file exists"""
        fig = display_train_test_matrices(self.rf_cache, save_location=TEST_IMAGE_LOCATION)
        axes = fig.get_axes()
        # Tests train subplot
        with self.subTest():
            self.assertEqual(axes[0].title.get_text(), 'Train Confusion Matrix')
        with self.subTest():
            self.assertEqual(axes[0].get_xlabel(), 'Predicted label')
        with self.subTest():
            self.assertEqual(axes[0].get_ylabel(), 'True label')

        # Test test subplot (axes[1] is the colorbar)
        with self.subTest():
            self.assertEqual(axes[2].title.get_text(), 'Test Confusion Matrix')
        with self.subTest():
            self.assertEqual(axes[2].get_xlabel(), 'Predicted label')
        with self.subTest():
            self.assertEqual(axes[2].get_ylabel(), 'True label')

        # test file creation
        with self.subTest():
            self.assertTrue(os.path.exists(TEST_IMAGE_LOCATION))

    def test_feature_importance(self):
        """Tests if the RF Feature importance is created and saved in the right spot"""
        fig = display_rf_feature_importance(self.rf_cache, save_location=TEST_IMAGE_LOCATION)
        with self.subTest():
            self.assertEqual(fig.get_axes()[0].get_title(), "Pixel importance in random forest classification")
        with self.subTest():
            self.assertTrue(os.path.exists(TEST_IMAGE_LOCATION))

    def test_loss_curve(self):
        """Tests that the plot has been created with the correct axes, and exists on disk"""
        fig = display_loss_curve(self.mlp_model.model.loss_curve_, save_location=TEST_IMAGE_LOCATION)
        axes = fig.get_axes()[0]
        with self.subTest():
            self.assertEqual(axes.get_xlabel(), "Iteration")
        with self.subTest():
            self.assertEqual(axes.get_ylabel(), "Loss")
        with self.subTest():
            self.assertTrue(os.path.exists(TEST_IMAGE_LOCATION))

    def test_mlp_coefficients(self):
        """Checks that the plot is created and it has as many subplots as neurons in its first hidden layer"""
        fig = display_mlp_coefficients(self.mlp_model.model.coefs_, rows=4, cols=4, save_location=TEST_IMAGE_LOCATION)
        axes = fig.get_axes()
        with self.subTest():
            self.assertEqual(len(axes), self.mlp_model.hidden_layers[0])
        with self.subTest():
            self.assertTrue(os.path.exists(TEST_IMAGE_LOCATION))
