import dataset
import argparse
from sklearn.ensemble import RandomForestClassifier

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
    retval = RandomForestClassifier(n_estimators=trees, 
        max_depth=depth,
        criterion=impurity_method)
    
    retval.fit(samples, labels)

    return retval


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
