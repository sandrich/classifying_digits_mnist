from src import dataset, random_forest
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run MNIST classifier')
    parser.add_argument('--trees', type=int, help='Number of trees', default=20)
    parser.add_argument('--depth', type=int, help='Maximum tree depth', default=9)
    parser.add_argument('--impurity_method', help='Impurity method', default='entropy', choices=['entropy', 'gini'])
    parser.add_argument('--save_RF', help="Saves the trained Random Forest model to disk", default=None)
    parser.add_argument('--load_RF', help="Loads a trained Random Forest model from disk", default=None)

    args = parser.parse_args()
    if args.save_RF is not None and args.load_RF is not None:
        print("Can't load and save a model at the same time... please choose just one of the two options")
        exit(0)
    # Load dataset
    train_data, train_labels = dataset.load_train_data()
    test_data, test_labels = dataset.load_test_data()

    # Run classifier
    cache = random_forest.run_classification(train_data, train_labels, test_data, test_labels, args.trees, args.depth,
                               args.impurity_method, args.save_RF, args.load_RF)

    random_forest.print_results(args, cache)

    random_forest.display_results(cache)


if __name__ == "__main__":
    main()
