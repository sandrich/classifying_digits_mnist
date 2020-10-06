"""
Entry point script
"""
import argparse
import sys
from mnist_classifier import dataset
from mnist_classifier.random_forest import  RandomForest
from mnist_classifier.mlp import MLP


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Run MNIST classifier')
    subparsers = parser.add_subparsers(title="algorithms", help="Individual algorithms' options")

    # Random Forest arguments
    rf_parser = subparsers.add_parser("rf", help="Use a Random Forest Classifier to train/test the data")
    rf_parser.add_argument('--trees', '-t', type=int, help='Number of n_estimators', default=20)
    rf_parser.add_argument('--depth', '-d', type=int, help='Maximum tree max_depth', default=9)
    rf_parser.add_argument('--impurity_method', '-i', help='Impurity method', default='entropy',
                           choices=['entropy', 'gini'])
    rf_parser.add_argument('--save', '-s', help="Indicate the filename to save the model to disk", default=None)
    rf_parser.add_argument('--load', '-l', help="Indicate the filename of a saved model to load", default=None)

    # MLP arguments
    mlp_parser = subparsers.add_parser('mlp', help="Use a MLP Classifier to train/test the data")
    mlp_parser.add_argument('--hidden_layers', '-hl', nargs="+", help="The number of neurons in each hidden layer. " +
                                                                     "Separate the hidden layers with spaces.",
                            default=['100'])
    mlp_parser.add_argument('--alpha', '-a', type=float, help="the alpha bias of the MLP", default=0.0001)
    mlp_parser.add_argument('--batch_size', '-b', help="The batch size of training. you can specify a number or 'auto'",
                            default='auto')
    mlp_parser.add_argument('--max_iter', '-i', type=int,
                            help="The number of maximum training iterations to perform", default=200)
    mlp_parser.add_argument('--verbose', '-v',
                            help="Show training iterations with losses while training", action="store_true")
    mlp_parser.add_argument('--save', '-s', help="Indicate the filename to save the model to disk", default=None)
    mlp_parser.add_argument('--load', '-l', help="Indicate the filename of a saved model to load", default=None)

    args = parser.parse_args()

    if args.save_RF is not None and args.load_RF is not None:
        print("Can't load and save a model at the same time... please choose just one of the two options")
        sys.exit(1)
    if args.save_MLP is not None and args.load_MLP is not None:
        print("Can't load and save a model at the same time... please choose just one of the two options")
        sys.exit(1)

    hidden_layers = []
    for layer in args.hidden_layers:
        try:
            hidden_layers.append(int(layer))
        except ValueError:
            print(f"Can't set a hidden layer value of {layer}. Please enter an integer value")
            sys.exit(1)
    args.hidden_layers = tuple(hidden_layers)

    # Load dataset
    train_data, train_labels = dataset.load_train_data()
    test_data, test_labels = dataset.load_test_data()

    # Random Forest
    if args.rf:
        rf_classifier = RandomForest(n_estimators=args.trees, max_depth=args.depth, criterion=args.impurity_method)
        rf_classifier.run_classification(train_data, train_labels, test_data, test_labels,
                                         args.save, args.load)

    if args.mlp:
        mlp_classifier = MLP(hidden_layer_sizes=args.hidden_layers, alpha=args.alpha,
                             max_iter=args.max_iter, batch_size=args.batch_size)
        mlp_classifier.run_classification(train_data, train_labels, test_data, test_labels,
                                          args.save, args.load)


if __name__ == "__main__":
    main()
