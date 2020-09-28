"""
RandomForest model
"""
from sklearn.ensemble import RandomForestClassifier
from . import visualizer
from .algorithm_meta import AlgorithmMeta


class RandomForest(AlgorithmMeta):
    """
    RandomForest implementation
    """

    def __init__(self, n_estimators, max_depth, criterion):
        """
        Initiates the model with the correct configurations.
        :param n_estimators:
        :param max_depth:
        :param criterion:
        """
        super().__init__()
        self.trees = n_estimators
        self.depth = max_depth
        self.impurity_method = criterion

        if self.trees < 1:
            raise ValueError('Number of n_estimators have to be greater than 0')

        if self.depth < 1:
            raise ValueError('Depth of a tree has to be greater than 0')

        if self.impurity_method not in ['entropy', 'gini']:
            raise ValueError('Impurity method supported: entropy, gini')

        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)

    def load_model(self, filepath):
        super().load_model(filepath)
        self._check_loaded_model_and_set_conf(['n_estimators', 'max_depth', 'criterion'])

    def print_results(self, cache):
        super().print_results(cache)
        print('-----------------')
        print('Max tree max_depth: {}'.format(self.depth))
        print('Number of n_estimators: {}'.format(self.trees))
        print('Impurity method: {}'.format(self.impurity_method))

    def display_results(self, cache):
        visualizer.display_train_test_matrices(cache)
        visualizer.display_rf_feature_importance(cache)
