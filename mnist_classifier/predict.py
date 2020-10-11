"""
Entry point script
"""
import argparse
import sys
from mnist_classifier import dataset
from mnist_classifier import report_manager
from mnist_classifier.random_forest import RandomForest
from mnist_classifier.mlp import MLP

def parse_args(args):
    """
    Parses the arguments
    """
    parser = argparse.ArgumentParser(description='Run MNIST classifier',
                                     epilog="Visit "
                                            "https://sandrich.github.io/classifying_digits_mnist/usage/params.html "
                                            "for a comprehensive explanation of these arguments.")
    subparsers = parser.add_subparsers(title="algorithms", help="Individual algorithms' options")

    # Global arguments
    parser.add_argument("--random_seed", "-rs", type=int,
                        help="Defines a random seed to reproduce arguments", default=None)
    parser.add_argument("--report_directory", '-rd', type=str,
                        help="An output directory where to save the report. "
                             "If the folder does not exist, it will be created", default=None)
    parser.add_argument('--save', '-s', help="Indicate the filename to save the model to disk", default=None)
    parser.add_argument('--load', '-l', help="Indicate the filename of a saved model to load", default=None)
    parser.add_argument('--test_suite', '-ts', type=str,
                        help="Indicate the location of a test suite configuration file", default=None)

    # Random Forest arguments
    rf_parser = subparsers.add_parser("rf", help="Use a Random Forest Classifier to train/test the data")
    rf_parser.add_argument('--trees', '-t', type=int, help='Number of n_estimators', default=20)
    rf_parser.add_argument('--depth', '-d', type=int, help='Maximum tree max_depth', default=9)
    rf_parser.add_argument('--impurity_method', '-i', help='Impurity method', default='entropy',
                           choices=['entropy', 'gini'])

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

    return parser.parse_args(args) if len(args) > 0 else parser.parse_args()


def main(*arguments):
    """
    Main function. Parses the arguments and executes the appropriate algorithm(s).

    Parameters
    *arguments : list
        potential arguments to be passed programmatically.
    """

    args = parse_args(arguments)

    if args.test_suite is not None:
        test_suite = report_manager.load_test_suite_conf(args.test_suite)
        for i, test in enumerate(test_suite):
            args = parse_args(test)
            process_args_and_run(args, test_suite_iter=i)
    else:
        process_args_and_run(args)

def load_report_directory(args):
    """
    Loads report directory if exists
    """
    if args.report_directory is None:
        return args
    args.report_directory = report_manager.prepare_report_dest(args.report_directory)
    return args

def process_args_and_run(args, test_suite_iter: int = None):
    """
    Takes the arguments from an argument parser, or a test_suite and runs the appropriate test.

    Parameters
    ----------
    args : argparse.Namespace
        the parsed arguments
    test_suite_iter : int
        if we are in a test suite, this is the test # of the suite, None if we are not in a suite.
    """
    # load global arguments
    if args.save is not None and args.load is not None:
        print("Can't load and save a model at the same time..."
              "please choose just one of the two options")
        sys.exit(1)

    if test_suite_iter is None:
        load_report_directory(args)

    #check if no parameters are given
    if all([argument not in args for argument in ['depth',
                                                  'impurity_method',
                                                  'trees',
                                                  'alpha',
                                                  'hidden_layers',
                                                  'max_iter',
                                                  'verbose']]):
        print("Please select at least one algorithm or test-suite to run this program")
        sys.exit()

    # Load dataset
    train_data, train_labels = dataset.load_train_data()
    test_data, test_labels = dataset.load_test_data()
    # Random Forest
    if all([param in args for param in ['depth', 'impurity_method', 'trees']]):
        rf_classifier = RandomForest(n_estimators=args.trees, max_depth=args.depth, criterion=args.impurity_method,
                                     report_directory=args.report_directory, test_suite_iter=test_suite_iter)
        rf_classifier.run_classification(train_data, train_labels, test_data, test_labels,
                                         args.save, args.load)
    if all([param in args for param in ['alpha', 'hidden_layers', 'max_iter', 'verbose']]):
        hidden_layers = []
        # case when reading from a test suite where args.hidden_layers is just ['x y z'] and not ['x', 'y', 'z']
        if len(args.hidden_layers) == 1:
            args.hidden_layers = args.hidden_layers[0].split(" ")
        for layer in args.hidden_layers:
            try:
                hidden_layers.append(int(layer))
            except ValueError:
                print(f"Can't set a hidden layer value of {layer}. Please enter an integer value")
                sys.exit(1)
        args.hidden_layers = tuple(hidden_layers)

        mlp_classifier = MLP(hidden_layer_sizes=args.hidden_layers,
                             alpha=args.alpha,
                             max_iter=args.max_iter,
                             batch_size=args.batch_size,
                             verbose=args.verbose,
                             random_seed=args.random_seed,
                             report_directory=args.report_directory,
                             test_suite_iter=test_suite_iter)

        mlp_classifier.run_classification(train_data, train_labels, test_data, test_labels,
                                          args.save, args.load)


if __name__ == "__main__":
    main()
