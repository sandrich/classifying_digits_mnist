"""
MLP Classifier class
"""
import os
from sklearn.neural_network import MLPClassifier
from .algorithm_meta import AlgorithmMeta
from . import visualizer


class MLP(AlgorithmMeta):
    """
    A basic MLP classifier
    """

    def __init__(self, hidden_layer_sizes: tuple = (10, 10, 10), alpha: float = 0.0001,
                 batch_size='auto', max_iter: int = 200):
        """
        Initializes the MLP model
        :param hidden_layer_sizes: the number of neurons in each hidden layer.
        For example, (100,200,100) will have three hidden layers. Layer 0 will have 100 neurons, layer 1 will have 200,
        and layer 2 will have 100. These hidden layers are the ones between the input and output layers.
        :param alpha: regularization term parameter
        :param batch_size: the size of minibatches for stochastic optimizers.
        :param max_iter: maximum number of training iterations to perform while training.
        """
        super().__init__()

        if len(hidden_layer_sizes) == 0:
            raise ValueError("You need at least one hidden layer in the MLP!")
        if sum(n <= 0 for n in hidden_layer_sizes):
            raise ValueError("Each layer must have a positive number of neurons!")
        if max_iter <= 0:
            raise ValueError("You need at least one training iteration!")
        if batch_size != "auto" and not isinstance(batch_size, int):
            raise ValueError("batch size must either be an integer value or 'auto'!")
        if isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("batch size must be positive!")

        self.hidden_layers = hidden_layer_sizes
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            batch_size=batch_size,
            max_iter=max_iter,
            verbose=True,
            random_state=12345)  # static random seed

    def load_model(self, filepath):
        super().load_model(filepath)
        self._check_loaded_model_and_set_conf(['hidden_layer_sizes', 'alpha', 'batch_size', 'max_iter'])

    def print_results(self, cache):
        super().print_results(cache)
        print('-----------------')
        print('Hidden-layer neuron count: {}'.format(", ".join([str(n) for n in self.hidden_layers])))
        print('Alpha: {}'.format(self.alpha))
        print('Batch size: {}'.format(self.batch_size))
        print('Max training iterations: {}'.format(self.max_iter))

    def display_results(self, cache, save_directory: str = None):
        super().display_results(cache, save_directory)
        loss_curve_out = None if save_directory is None else os.path.join(save_directory, "loss_curve.png")
        visualizer.display_loss_curve(self.model.loss_curve_, save_location=loss_curve_out)
        mlp_coefficients = None if save_directory is None else os.path.join(save_directory, "MLP_coefficients.png")
        visualizer.display_mlp_coefficients(self.model.coefs_, save_location=mlp_coefficients)
