"""
Entry point script
"""
import argparse
import sys
from src import dataset
from src.random_forest import  RandomForest



def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Run MNIST classifier')
    parser.add_argument('--trees', type=int, help='Number of trees', default=20)
    parser.add_argument('--depth', type=int, help='Maximum tree depth', default=9)
    parser.add_argument('--impurity_method', help='Impurity method', default='entropy', choices=['entropy', 'gini'])
    parser.add_argument('--save_RF', help="Saves the trained Random Forest model to disk", default=None)
    parser.add_argument('--load_RF', help="Loads a trained Random Forest model from disk", default=None)

    args = parser.parse_args()
    if args.save_RF is not None and args.load_RF is not None:
        print("Can't load and save a model at the same time... please choose just one of the two options")
        sys.exit(1)

    # Load dataset
    train_data, train_labels = dataset.load_train_data()
    test_data, test_labels = dataset.load_test_data()

    # Run classifier
    rf_classifier = RandomForest(trees=args.trees, depth=args.depth, impurity_method=args.impurity_method)
    rf_classifier.run_classification(train_data, train_labels, test_data, test_labels,
                                             args.save_RF, args.load_RF)


if __name__ == "__main__":
    main()
