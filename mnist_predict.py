from src import dataset, random_forest
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run MNIST classifier')
    parser.add_argument('--trees', type=int, help='Number of trees', default=20)
    parser.add_argument('--depth', type=int, help='Maximum tree depth', default=9)
    parser.add_argument('--impurity_method', help='Impurity method', default='entropy', choices=['entropy', 'gini'])

    args = parser.parse_args()

    # Load dataset
    train_data, train_labels = dataset.load_train_data()
    test_data, test_labels = dataset.load_test_data()

    # Run classifier
    cache = random_forest.run_classification(train_data, train_labels, test_data, test_labels, args.trees, args.depth,
                               args.impurity_method)

    random_forest.print_results(args, cache)


if __name__ == "__main__":
    main()
