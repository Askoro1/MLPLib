import numpy as np
import scipy
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        output = np.where(input > 0., input, 0.)
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # d(ReLU(x)) / dx
        d_ReLU_dx = np.where(input > 0., 1., 0.)
        # dl / dx
        grad_dl_dx = grad_output * d_ReLU_dx
        return grad_dl_dx


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        output = scipy.special.expit(input)
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # sigmoid(x)
        sigmoid_x = self.compute_output(input)
        # d(sigmoid(x)) / dx
        d_sigmoid_dx = sigmoid_x * (1. - sigmoid_x)
        # dl / dx
        grad_dl_dx = grad_output * d_sigmoid_dx
        return grad_dl_dx


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        output = scipy.special.softmax(input, axis=1)
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # softmax(x)
        softmax_x = self.compute_output(input)
        # dl / dx
        grad_dl_dx = (grad_output - (grad_output * softmax_x).sum(axis=1, keepdims=True)) * softmax_x
        return grad_dl_dx


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        output = scipy.special.log_softmax(input, axis=1)
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # softmax(x)
        softmax_x = scipy.special.softmax(input, axis=1)
        # dl / dx
        grad_dl_dx = grad_output - (grad_output.sum(axis=1, keepdims=True) * softmax_x)
        return grad_dl_dx
