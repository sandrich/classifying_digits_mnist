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

    # Random Forest arguments
    parser.add_argument('--rf', action="store_true", help="Use this flag to use the Random Forest Classifier.")
    parser.add_argument('--trees', type=int, help='Number of n_estimators', default=20)
    parser.add_argument('--depth', type=int, help='Maximum tree max_depth', default=9)
    parser.add_argument('--impurity_method', help='Impurity method', default='entropy', choices=['entropy', 'gini'])
    parser.add_argument('--save_RF', help="Saves the trained Random Forest model to disk", default=None)
    parser.add_argument('--load_RF', help="Loads a trained Random Forest model from disk", default=None)

    # MLP arguments
    parser.add_argument('--mlp', action="store_true", help="Use this flag to use the MLP Classifier")
    parser.add_argument('--hidden_layers', nargs="+", help="The number of neurons in each hidden layer. " +
                                                           "Separate the hidden layers with spaces.", default=['100'])
    parser.add_argument('--alpha', type=float, help="the alpha bias of the MLP", default=0.0001)
    parser.add_argument('--batch_size', help="The batch size of training. you can specify a number or 'auto'",
                        default='auto')
    parser.add_argument('--max_iter', type=int,
                        help="The number of maximum training iterations to perform", default=200)
    parser.add_argument('--verbose_MLP',
                        help="Show training iterations with losses while training", action="store_true")
    parser.add_argument('--save_MLP', help="Saves the trained MLP model to disk", default=None)
    parser.add_argument('--load_MLP', help="Loads a trained MLP model from disk", default=None)

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
                                         args.save_RF, args.load_RF)

    if args.mlp:
        mlp_classifier = MLP(hidden_layer_sizes=args.hidden_layers, alpha=args.alpha,
                             max_iter=args.max_iter, batch_size=args.batch_size)
        mlp_classifier.run_classification(train_data, train_labels, test_data, test_labels,
                                          args.save_MLP, args.load_MLP)


if __name__ == "__main__":
    main()
