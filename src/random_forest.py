import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from . import visualizer
import os
import pickle


def train(samples, labels, trees, depth, impurity_method):
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


def load_model(fp):
    print("Loading model", fp)
    return pickle.load(open(fp, 'rb'))


def save_model(classifier, fp):
    print("Saving model as", fp)
    pickle.dump(classifier, open(fp, 'wb'))


def predict(classifier, samples):
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


def run_classification(train_data, train_labels, test_data, test_labels, trees, depth, impurity_method, model_to_save,
                       model_to_load=None):
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
        classifier = load_model(model_to_load)
    else:
        if model_to_load is not None:
            print(f"Couldn't find the model {model_to_load}... training a new model.")
        classifier = train(train_data, train_labels, trees, depth, impurity_method)

    if model_to_save is not None:
        save_model(classifier, model_to_save)

    train_pred = predict(classifier, train_data)
    test_pred = predict(classifier, test_data)

    train_acc = accuracy(train_labels, train_pred)
    test_acc = accuracy(test_labels, test_pred)

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
        }}

    return cache


def print_results(args, cache):
    print('Classification stats:')
    print('-----------------')
    print('Max tree depth: {}'.format(args.depth))
    print('Number of trees: {}'.format(args.trees))
    print('Impurity method: {}'.format(args.impurity_method))
    print('-----------------')
    print('Train Accuracy: {0:.3f}'.format(cache['accuracy']['train']))
    print('Test  Accuracy: {0:.3f}'.format(cache['accuracy']['test']))


def display_results(cache):
    #visualizer.display_confusion_matrix(cache['actual']['test'], cache['prediction']['test'])
    visualizer.display_train_test_matrices(cache)