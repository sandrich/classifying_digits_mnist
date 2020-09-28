"""
Visualizer
"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def display_confusion_matrix(actual_y, predicted_y, show=True, axes=None):
    """
    :param actual_y: the actual labels of the classified objects
    :param predicted_y: the predicted labels of the classified objects
    :param show: whether or not to display the plot (useful for combining multiple plots)
    :param axes: the matplotlib axes to plot on
    :return:
    """
    cmd = ConfusionMatrixDisplay(confusion_matrix(actual_y, predicted_y))
    cmd.plot(cmap="Blues", ax=axes)

    if show:
        plt.show()


def display_train_test_matrices(cache):
    """
    Displays the train and test confusion matrices
    :param cache: the cache dict returned by the classifier.
    Must at least include ['actual', 'predtion'] objects, each with ['train', 'test'] arrays
    :return:
    """
    plt.figure(figsize=(13, 7))
    ax1 = plt.subplot(121)
    plt.title("Train Confusion Matrix")
    display_confusion_matrix(cache['actual']['train'], cache['prediction']['train'], show=False, axes=ax1)
    ax2 = plt.subplot(122)
    plt.title("Test Confusion Matrix")
    display_confusion_matrix(cache['actual']['test'], cache['prediction']['test'], show=False, axes=ax2)
    plt.show()


def display_rf_feature_importance(cache):
    """
    Displays which pixels have the most influence in the model's decision.
    This is based on sklearn,ensemble.RandomForestClassifier's feature_importances_ array
    :param cache: the classifier cache (which contains the model itself)
    :return:
    """
    plt.title("Pixel importance in random forest Classification")
    plt.imshow(cache['model'].feature_importances_.reshape((28,28)), extent=[0,28,28,0])
    plt.colorbar()
    plt.show()


def display_loss_curve(losses):
    """
    Plots and displays the loss curve (usually for Neural Network models)
    :param losses: 1D numpy array
    :return:
    """
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


def display_mlp_coefficients(coefficients, rows=4, cols=4):
    """
    Shows the first layer's coefficients of the input layer

    The first rows*cols neurons' coefficients are displayed. if rows*cols is greater than the number of neurons, all the
    neurons are displayed
    :param coefficients: 2D array, the network's coefficients
    :param rows: int, the number of rows to display
    :param cols: int, the number of columns to display
    :return:
    """
    coefficients = coefficients[0]
    coefficients -= min(coefficients.ravel())
    coefficients /= max(coefficients.ravel())

    for pos in range(rows*cols):
        if coefficients.shape[1] <= pos:
            continue
        plt.subplot(rows, cols, pos+1)
        plt.imshow(coefficients[:, pos].reshape(28, 28), cmap="gray")
        plt.xticks(())
        plt.yticks(())
    plt.show()
