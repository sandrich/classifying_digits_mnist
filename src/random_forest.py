"""
RandomForest
"""
import os
from sklearn.ensemble import RandomForestClassifier
from . import visualizer
from .algorithm_meta import AlgorithmMeta


class RandomForest(AlgorithmMeta):
    """
    Random Forest which inherits from the AlgorithmMeta class
    """

    def __init__(self, n_estimators, max_depth, criterion, random_seed: int = None, report_directory: str = None):
        """
        Initiates the model with the correct configurations.

        Parameters
        ----------
        n_estimators : int
            the number of estimators to use in the random forest

        max_depth : int
            the maximum depth of each tree in the forest

        criterion : str
            the split criterion for the trees in the forest. Can either be `gini` or`entropy`

        random_seed : int
            a random seed to make reproducible results. if None, no random seed is used.

        report_directory : str
            Where to save the output images
        """
        super().__init__(report_directory)
        self.trees = n_estimators
        self.depth = max_depth
        self.impurity_method = criterion
        self.report_directory = report_directory

        if self.trees < 1:
            raise ValueError('Number of n_estimators have to be greater than 0')

        if self.depth < 1:
            raise ValueError('Depth of a tree has to be greater than 0')

        if self.impurity_method not in ['entropy', 'gini']:
            raise ValueError('Impurity method supported: entropy, gini')

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_seed)

    def load_model(self, filepath):
        super().load_model(filepath)
        self._check_loaded_model_and_set_conf(['n_estimators', 'max_depth', 'criterion'])

    def print_results(self, cache):
        results = super().print_results(cache)
        print('-----------------')
        print('Max tree max_depth: {}'.format(self.depth))
        print('Number of n_estimators: {}'.format(self.trees))
        print('Impurity method: {}'.format(self.impurity_method))

        results['Max tree depth'] = [self.depth]
        results['Number of trees'] = [self.trees]
        results['Impurity method'] = [self.impurity_method]

        return results

    def display_results(self, cache, save_directory: str = None):
        super().display_results(cache)
        rf_feature_importance_out = None if self.report_directory is None else \
            os.path.join(self.report_directory, "RF_feature_importance.png")
        visualizer.display_rf_feature_importance(cache, save_location=rf_feature_importance_out)
