import dataset
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run MNIST classifier')
    parser.add_argument('--trees', type=int, help='Number of trees', default=20)
    parser.add_argument('--depth', type=int, help='Maximum tree depth', default=9)
    parser.add_argument('--impurity_method', help='Impurity method', default='entropy', choices=['entropy', 'gini'])

    args = parser.parse_args()
    print(args)
    print(dataset.load_train_data())

if __name__ == "__main__":
    main()
