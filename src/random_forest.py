import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys


def train(samples, labels, trees, depth, impurity_method):
    """
    This function trains the dataset using RandomForest algorithm

    :param
        samples: Dataset with samples
        labels: Labels matching the dataset
        trees: Number of trees
        depth: Maximum tree depth
        impurity_method: Impurity node method used (Entropy, Gini)


    :return
        Returns train instance for further processing
    """
    if trees < 1:
        raise ValueError('Number of trees have to be greater than 0')

    if depth < 1:
        raise ValueError('Depth of a tree has to be greater than 0')

    if impurity_method not in ['entropy', 'gini']:
        raise ValueError('Impurity method supported: entropy, gini')

    retval = RandomForestClassifier(n_estimators=trees,
                                    max_depth=depth,
                                    criterion=impurity_method)

    print("Starting training...")
    retval.fit(samples, labels)
    print("Done training.")

    return retval


def prediction(classifier, samples):
    """
    Returns prediction of the class labels for input

    :param
        classifier: Class instance with the trained random forest
        samples: Sample data set

    :return
        Array with the predicted class label
    """
    print("Predicting...")
    return classifier.predict(samples)


def accuracy(labels, pred, show=False):
    """
    Returns accuracy given prediction and labels
    """

    return np.sum(pred == labels) / len(labels)


def run_classification(train_data, train_labels, test_data, test_labels, trees, depth, impurity_method):
    """
    Trains and tests the classification

    :param
        train_data: Train sample set
        train_labels: Train labels
        test_data: Test data set
        test_labels: Test labels
        trees: Number of trees
        depth: Maximum tree depth
        impurity_method: Impurity node method used (Entropy, Gini)

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
    classifier = train(train_data, train_labels, trees, depth, impurity_method)

    train_pred = prediction(classifier, train_data)
    test_pred = prediction(classifier, test_data)

    train_acc = accuracy(train_labels, train_pred)
    test_acc = accuracy(test_labels, test_pred)

    cache = {'prediction': {'train': train_pred, 'test': test_pred}, 'accuracy': {'train': train_acc, 'test': test_acc}}

    return cache


def print_results(args, cache):
    print('Classification stats:')
    print('-----------------')
    print('Max tree depth: {}'.format(args.depth))
    print('Number of trees: {}'.format(args.trees))
    print('Impurity method: {}'.format(args.impurity_method))
    print('-----------------')
    print('Train Accuracy: {0:.3f}'.format(cache['accuracy']['train']))
    print('Train Accuracy: {0:.3f}'.format(cache['accuracy']['test']))

