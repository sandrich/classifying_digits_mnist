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
                 batch_size='auto', max_iter: int = 200, verbose: bool = False, random_seed: int=None,
                 report_directory: str = None, test_suite_iter: int = None):
        """
        Initializes the MLP model

        Parameters
        ----------
        hidden_layer_sizes : tuple
            the number of neurons in each hidden layer. the length of the tuple determines the number of hidden layers
        alpha : float
            the alpha bias of the network
        batch_size : int
            the number of samples per training batch. Can be a specific integer or "auto" to let the algorithm decide.
        max_iter : int
            the maximum number of training iterations the algorithm should perform.
        verbose : bool
            whether or not to print out the training iterations
        random_seed : int
            a random seed to make results reproducible. if none, no random seed is used
        report_directory : str
            Where to save the output images
        test_suite_iter : int
            The test suite number we're in right now. None if we are not in a test suite.
        """
        super().__init__(report_directory, test_suite_iter=test_suite_iter)

        if len(hidden_layer_sizes) == 0:
            raise ValueError("You need at least one hidden layer in the MLP!")
        if sum(n <= 0 for n in hidden_layer_sizes):
            raise ValueError("Each layer must have a positive number of neurons!")
        if max_iter <= 0:
            raise ValueError("You need at least one training iteration!")
        if batch_size != "auto":
            try:
                batch_size = int(batch_size)
            except ValueError as error:
                raise ValueError("batch size must either be an integer value or 'auto'!") from error
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
            verbose=verbose,
            random_state=random_seed)

    def load_model(self, filepath):
        super().load_model(filepath)
        self._check_loaded_model_and_set_conf(['hidden_layer_sizes', 'alpha', 'batch_size', 'max_iter'])

    def print_results(self, cache):
        results = super().print_results(cache)
        hidden_layers_str = ", ".join([str(n) for n in self.hidden_layers])
        print('-----------------')
        print('Hidden-layer neuron count: {}'.format(hidden_layers_str))
        print('Alpha: {}'.format(self.alpha))
        print('Batch size: {}'.format(self.batch_size))
        print('Max training iterations: {}'.format(self.max_iter))

        results['Hidden layer Neurons'] = [hidden_layers_str]
        results['Alpha'] = [self.alpha]
        results['Batch size'] = [self.batch_size]
        results['Max training Iterations'] = [self.max_iter]
        results['Actual training Iterations'] = [self.model.n_iter_]
        return results

    def display_results(self, cache):
        super().display_results(cache)
        if self.report_directory is None:
            loss_curve_out = None
            mlp_coefficients = None
        elif self.test_suite_iter is None:
            loss_curve_out = os.path.join(self.report_directory, "loss_curve.png")
            mlp_coefficients = os.path.join(self.report_directory, "MLP_coefficients.png")
        else:
            loss_curve_out = os.path.join(self.report_directory, f"{self.test_suite_iter}_loss_curve.png")
            mlp_coefficients = os.path.join(self.report_directory, f"{self.test_suite_iter}_MLP_coefficients.png")

        visualizer.display_loss_curve(self.model.loss_curve_, save_location=loss_curve_out)
        visualizer.display_mlp_coefficients(self.model.coefs_, save_location=mlp_coefficients)
