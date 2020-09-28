"""
RandomForest model
"""
import os
from sklearn.ensemble import RandomForestClassifier
from . import visualizer
from .algorithm_meta import AlgorithmMeta


class RandomForest(AlgorithmMeta):
    """
    RandomForest implementation
    """

    def __init__(self, trees, depth, impurity_method):
        """
        Initiates the model with the correct configurations.
        :param trees:
        :param depth:
        :param impurity_method:
        """
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=trees, max_depth=depth, criterion=impurity_method)
        self.trees = trees
        self.depth = depth
        self.impurity_method = impurity_method

        if self.trees < 1:
            raise ValueError('Number of trees have to be greater than 0')

        if self.depth < 1:
            raise ValueError('Depth of a tree has to be greater than 0')

        if self.impurity_method not in ['entropy', 'gini']:
            raise ValueError('Impurity method supported: entropy, gini')

    def fit(self, data, targets):
        """ Fits the internal model on the given data
        :param data: the data to fit the model on
        :param targets: the output class of the given data
        :return:
        """

        print("Starting training...")
        self.model.fit(data, targets)
        print("Done training.")
        return self.model

    def predict(self, data_to_predict):
        """
        Returns prediction of the class y for input

        :param
            classifier: Class instance with the trained random forest
            data_to_predict: Sample data set

        :return
            Array with the predicted class label
        """
        print("Predicting...")
        return self.model.predict(data_to_predict)

    def load_model(self, filepath):
        super().load_model(filepath)
        if self.model.n_estimators != self.trees:
            print(f"[WARNING] - the model you loaded has {self.model.n_estimators} trees, "
                  f"but you specified {self.trees}! Continuing with loaded model...")

        if self.model.max_depth != self.depth:
            print(f"[WARNING] - the model you loaded has a max depth of {self.model.depth}, "
                  f"but you specified {self.depth}! Continuing with loaded model...")

        if self.model.criterion != self.impurity_method:
            print(f"[WARNING] - the model you loaded has an impurity criterion of {self.model.criterion}, "
                  f"but you specified {self.impurity_method}! Continuing with loaded model...")

    def run_classification(self, train_data, train_labels,
                           test_data, test_labels,
                           model_to_save=None, model_to_load=None):
        """
        Trains and tests the classification

        :param
            train_data: Train sample set
            train_labels: Train y
            X: Test data set
            test_labels: Test y
            save_model: filepath to save the trained model
            model_to_load: filepath of saved model to load instead of training


        :return
            Returns collection with prediction and accuracy
            cache:
                prediction:
                    fit: float
                    test: float
                accuracy:
                    fit: float
                    test: float
        """
        if model_to_load is not None and os.path.exists(model_to_load):
            self.load_model(model_to_load)
        else:
            if model_to_load is not None:
                print(f"Could not find the model {model_to_load}... training a new model.")
            self.fit(train_data, train_labels)

        if model_to_save is not None:
            self.save_model(model_to_save)

        train_pred = self.predict(train_data)
        test_pred = self.predict(test_data)

        train_acc = self.score(train_data, train_labels)  # score() inherited from sklearn.base.ClassifierMixin
        test_acc = self.score(test_data, test_labels)

        cache = {
            'prediction': {
                'train': train_pred,
                'test': test_pred
            }, 'accuracy': {
                'train': train_acc,
                'test': test_acc
            }, 'actual': {
                'train': train_labels,
                'test': test_labels
            }, 'model': self.model}

        self.print_results(cache)
        self.display_results(cache)
        return cache

    def print_results(self, cache):
        print('Classification stats:')
        print('-----------------')
        print('Max tree depth: {}'.format(self.depth))
        print('Number of trees: {}'.format(self.trees))
        print('Impurity method: {}'.format(self.impurity_method))
        print('-----------------')
        print('Train Accuracy: {0:.3f}'.format(cache['accuracy']['train']))
        print('Test  Accuracy: {0:.3f}'.format(cache['accuracy']['test']))

    def display_results(self, cache):
        visualizer.display_train_test_matrices(cache)
        visualizer.display_rf_feature_importance(cache)
