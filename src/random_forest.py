from sklearn.ensemble import RandomForestClassifier
from . import visualizer
from .algorithm_meta import AlgorithmMeta
import os
import pickle


class RandomForest(AlgorithmMeta):

    def __init__(cls, trees, depth, impurity_method):
        """
        Initiates the model with the correct configurations.
        :param trees:
        :param depth:
        :param impurity_method:
        """
        super().__init__()
        cls.model = RandomForestClassifier(n_estimators=trees, max_depth=depth, criterion=impurity_method)
        cls.trees = trees
        cls.depth = depth
        cls.impurity_method = impurity_method

        if cls.trees < 1:
            raise ValueError('Number of trees have to be greater than 0')

        if cls.depth < 1:
            raise ValueError('Depth of a tree has to be greater than 0')

        if cls.impurity_method not in ['entropy', 'gini']:
            raise ValueError('Impurity method supported: entropy, gini')

    def train(self, samples, labels):
        """
        This function trains the dataset using RandomForest algorithm

        :param
            samples: Dataset with samples
            labels: Labels matching the dataset
            trees: Number of trees
            depth: Maximum tree depth
            impurity_method: Impurity node method used (Entropy, Gini)
            save_model: whether to save the model to disk or not

        :return
            Returns train instance for further processing
        """

        print("Starting training...")
        self.model.fit(samples, labels)
        print("Done training.")

    def load_model(self, fp):
        print("Loading model", fp)
        self.model = pickle.load(open(fp, 'rb'))
        if self.model.n_estimators != self.trees:
            print(f"[WARNING] - the model you loaded has {self.model.n_estimators} trees, "
                  f"but you specified {self.trees}! Continuing with loaded model...")
        if self.model.max_depth != self.depth:
            print(f"[WARNING] - the model you loaded has a max depth of {self.model.depth}, "
                  f"but you specified {self.depth}! Continuing with loaded model...")
        if self.model.criterion != self.impurity_method:
            print(f"[WARNING] - the model you loaded has an impurity criterion of {self.model.criterion}, "
                  f"but you specified {self.impurity_method}! Continuing with loaded model...")

    def save_model(self, fp):
        print("Saving model as", fp)
        pickle.dump(self.model, open(fp, 'wb'))

    def predict(self, samples):
        """
        Returns prediction of the class labels for input

        :param
            classifier: Class instance with the trained random forest
            samples: Sample data set

        :return
            Array with the predicted class label
        """
        print("Predicting...")
        return self.model.predict(samples)

    def run_classification(self, train_data, train_labels, test_data, test_labels, model_to_save=None, model_to_load=None):
        """
        Trains and tests the classification

        :param
            train_data: Train sample set
            train_labels: Train labels
            test_data: Test data set
            test_labels: Test labels
            save_model: filepath to save the trained model
            model_to_load: filepath of saved model to load instead of training


        :return
            Returns collection with prediction and accuracy
            cache:
                prediction:
                    train: float
                    test: float
                accuracy:
                    train: float
                    test: float
        """
        if model_to_load is not None and os.path.exists(model_to_load):
            self.load_model(model_to_load)
        else:
            if model_to_load is not None:
                print(f"Could not find the model {model_to_load}... training a new model.")
            self.train(train_data, train_labels)

        if model_to_save is not None:
            self.save_model(model_to_save)

        train_pred = self.predict(train_data)
        test_pred = self.predict(test_data)

        train_acc = AlgorithmMeta.accuracy(train_labels, train_pred)
        test_acc = AlgorithmMeta.accuracy(test_labels, test_pred)

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
