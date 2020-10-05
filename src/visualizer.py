"""
Visualizer
"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def __display_confusion_matrix(actual_y, predicted_y, axes=None):
    """
    :param actual_y: the actual labels of the classified objects
    :param predicted_y: the predicted labels of the classified object
    :param axes: the matplotlib axes to plot on
    :return:
    """
    cmd = ConfusionMatrixDisplay(confusion_matrix(actual_y, predicted_y))
    cmd.plot(cmap="Blues", ax=axes)


def display_train_test_matrices(cache, save_location: str = None):
    """
    Displays the train and test confusion matrices

    Parameters
    ----------
    save_location : str
    the location to save the figure on disk. If None, the plot is displayed on runtime and not saved.
    cache : dict
     the cache dict returned by the classifier.
     Must at least include ['actual', 'prediction'] objects, each with ['train', 'test'] arrays

    Returns
    -------
    matplotlib.pyplot.figure :
        the figure
    """
    fig = plt.figure(figsize=(13, 7))
    ax1 = plt.subplot(121)
    plt.title("Train Confusion Matrix")
    __display_confusion_matrix(cache['actual']['train'], cache['prediction']['train'], axes=ax1)
    ax2 = plt.subplot(122)
    plt.title("Test Confusion Matrix")
    __display_confusion_matrix(cache['actual']['test'], cache['prediction']['test'], axes=ax2)
    if save_location is None:
        plt.show()
    else:
        plt.savefig(fname=save_location)
    return fig


def display_rf_feature_importance(cache, save_location: str = None):
    """
    Displays which pixels have the most influence in the model's decision.
    This is based on sklearn,ensemble.RandomForestClassifier's feature_importance array

    Parameters
    ----------
    save_location : str
        the location to save the figure on disk. If None, the plot is displayed on runtime and not saved.
    cache : dict
        the cache dict returned by the classifier.
        Must at least include ['actual', 'prediction'] objects, each with ['train', 'test'] arrays

    Returns
    -------
    matplotlib.pyplot.figure :
        the figure
    """
    fig = plt.figure()
    plt.title("Pixel importance in random forest classification")
    plt.imshow(cache['model'].feature_importances_.reshape((28,28)), extent=[0,28,28,0])
    plt.colorbar()
    if save_location is None:
        plt.show()
    else:
        plt.savefig(fname=save_location)
    return fig


def display_loss_curve(losses, save_location: str=None):
    """
    Plots and displays the loss curve (usually for Neural Network models)

    Parameters
    ----------
    save_location : str
        the location to save the figure on disk. If None, the plot is displayed on runtime and not saved.
    losses : numpy.array
        the losses array of the MLP classifier's training.

    Returns
    -------
    matplotlib.pyplot.figure :
        the figure
    """
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    if save_location is None:
        plt.show()
    else:
        plt.savefig(fname=save_location)
    return fig


def display_mlp_coefficients(coefficients, rows=4, cols=4, save_location: str=None):
    """
    Shows the first layer's coefficients of the input layer

    The first rows*cols neurons' coefficients are displayed. if rows*cols is greater than the number of neurons, all the
    neurons are displayed. If there are more neurons' worth of coefficients to display than rows*cols, only the first
    ones are displayed.

    Parameters
    ----------
    coefficients numpy.array
        2D numpy array containing the input coefficients (or weights) of the MLP's hidden layers. Only the first
        layer's coefficients are displayed
    rows : int
        the number of rows to display in the figure
    cols : int
        the number of columns to display in the figure
    save_location : str
        the location to save the figure on disk. If None, the plot is displayed on runtime and not saved.

    Returns
    -------
    matplotlib.pyplot.figure :
        the figure
    """
    coefficients = coefficients[0]
    coefficients -= min(coefficients.ravel())
    coefficients /= max(coefficients.ravel())
    fig = plt.figure()
    for pos in range(rows*cols):
        if coefficients.shape[1] <= pos:
            continue
        plt.subplot(rows, cols, pos+1)
        plt.imshow(coefficients[:, pos].reshape(28, 28), cmap="gray")
        plt.xticks(())
        plt.yticks(())
    if save_location is None:
        plt.show()
    else:
        plt.savefig(fname=save_location)
    return fig
