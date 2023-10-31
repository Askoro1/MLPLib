import numpy as np
import scipy
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        square_error_matrix = np.square(input - target)
        output = square_error_matrix.sum() / (square_error_matrix.shape[0] * square_error_matrix.shape[1])
        return output

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        grad_input = 2. / (input.shape[0] * input.shape[1]) * (input - target)
        return grad_input


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        log_proba = self.log_softmax(input)
        target_matrix = np.repeat(target.reshape((-1, 1)), input.shape[1], axis=1)
        class_matrix = np.repeat(np.arange(input.shape[1]).reshape((1, -1)), input.shape[0], axis=0)
        class_equality_mask = np.where((target_matrix == class_matrix), 1, 0)
        output = -1. * (class_equality_mask * log_proba).sum() / input.shape[0]
        return output

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        softmax_f = scipy.special.softmax(input, axis=1)
        target_matrix = np.repeat(target.reshape((-1, 1)), input.shape[1], axis=1)
        class_matrix = np.repeat(np.arange(input.shape[1]).reshape((1, -1)), input.shape[0], axis=0)
        input_grad = np.where(target_matrix == class_matrix, softmax_f - 1., softmax_f) / input.shape[0]
        return input_grad
