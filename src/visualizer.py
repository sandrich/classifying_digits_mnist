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
    cm = ConfusionMatrixDisplay(confusion_matrix(actual_y, predicted_y))
    cm.plot(cmap="Blues", ax=axes)

    if show:
        plt.show()


def display_train_test_matrices(cache):
    """
    Displays the train and test confusion matrices
    :param cache: the cache dict returned by the classifier. Must at least include ['actual', 'predtion'] objects,
                    each with ['train', 'test'] arrays
    :return:
    """
    fig = plt.figure(figsize=(13,7))
    ax1 = plt.subplot(121)
    plt.title("Train Confusion Matrix")
    display_confusion_matrix(cache['actual']['train'], cache['prediction']['train'], show=False, axes=ax1)
    ax2 = plt.subplot(122)
    plt.title("Test Confusion Matrix")
    display_confusion_matrix(cache['actual']['test'], cache['prediction']['test'], show=False, axes=ax2)
    plt.show()
