import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        output = input @ self.weight.T
        if self.bias is not None:
            bias_matrix = np.repeat(self.bias.reshape((1, -1)), input.shape[0], axis=0)
            output += bias_matrix
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        grad_df_dx = self.weight
        grad_dl_dx = grad_output @ grad_df_dx
        return grad_dl_dx

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += (grad_output.T @ input)
        if self.bias is not None:
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            # train mode
            self.mean = input.sum(axis=0) / input.shape[0]
            mean_matrix = np.repeat(self.mean.reshape((1, -1)), input.shape[0], axis=0)
            self.input_mean = input - mean_matrix
            self.var = (np.square(self.input_mean)).sum(axis=0) / input.shape[0]
            self.sqrt_var = np.sqrt(self.var + self.eps * np.ones_like(self.var))
            self.inv_sqrt_var = 1. / self.sqrt_var
            inv_sqrt_var_matrix = np.repeat(self.inv_sqrt_var.reshape((1, -1)), input.shape[0], axis=0)
            self.norm_input = self.input_mean * inv_sqrt_var_matrix
            if self.affine:
                weight_matrix = np.repeat(self.weight.reshape((1, -1)), input.shape[0], axis=0)
                bias_matrix = np.repeat(self.bias.reshape((1, -1)), input.shape[0], axis=0)
                output = self.norm_input * weight_matrix + bias_matrix
            else:
                output = self.norm_input
            # updating running statistics
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * self.mean
            mod = input.shape[0] / (input.shape[0] - 1.)
            self.running_var = (1. - self.momentum) * self.running_var + self.momentum * mod * self.var
        else:
            # eval mode
            self.mean = self.running_mean
            mean_matrix = np.repeat(self.mean.reshape((1, -1)), input.shape[0], axis=0)
            self.input_mean = input - mean_matrix
            self.var = self.running_var
            self.sqrt_var = np.sqrt(self.var + self.eps * np.ones_like(self.var))
            self.inv_sqrt_var = 1. / self.sqrt_var
            inv_sqrt_var_matrix = np.repeat(self.inv_sqrt_var.reshape((1, -1)), input.shape[0], axis=0)
            self.norm_input = self.input_mean * inv_sqrt_var_matrix
            if self.affine:
                weight_matrix = np.repeat(self.weight.reshape((1, -1)), input.shape[0], axis=0)
                bias_matrix = np.repeat(self.bias.reshape((1, -1)), input.shape[0], axis=0)
                output = self.norm_input * weight_matrix + bias_matrix
            else:
                output = self.norm_input
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            # train mode

            # dl / d(norm_input)
            if self.affine:
                weight_matrix = np.repeat(self.weight.reshape((1, -1)), input.shape[0], axis=0)
                grad_dl_d_norm_input = grad_output * weight_matrix
            else:
                grad_dl_d_norm_input = grad_output
            # dl / d(inv_sqrt_var)
            grad_dl_d_inv_sqrt_var = (grad_dl_d_norm_input * self.input_mean).sum(axis=0)
            # dl / d(sqrt_var)
            d_inv_sqrt_var_d_sqrt_var = -1. * np.square(self.inv_sqrt_var)
            grad_dl_d_sqrt_var = grad_dl_d_inv_sqrt_var * d_inv_sqrt_var_d_sqrt_var
            # dl / d(var)
            d_sqrt_var_d_var = self.inv_sqrt_var / 2.
            grad_dl_d_var = grad_dl_d_sqrt_var * d_sqrt_var_d_var
            # dl / d(input_mean)
            d_norm_input_d_input_mean = np.repeat(self.inv_sqrt_var.reshape((1, -1)), input.shape[0], axis=0)
            grad_dl_d_var_matrix = np.repeat(grad_dl_d_var.reshape((1, -1)), input.shape[0], axis=0)
            d_var_d_input_mean = (2. / input.shape[0]) * self.input_mean
            grad_dl_d_input_mean = (grad_dl_d_norm_input * d_norm_input_d_input_mean +
                                    grad_dl_d_var_matrix * d_var_d_input_mean)
            # dl / d(mean)
            grad_dl_d_mean = (-1. * grad_dl_d_input_mean).sum(axis=0)
            # dl / dx
            grad_dl_d_mean_matrix = np.repeat(grad_dl_d_mean.reshape((1, -1)), input.shape[0], axis=0)
            grad_dl_dx = grad_dl_d_input_mean + grad_dl_d_mean_matrix / input.shape[0]
        else:
            # eval mode

            # dl / d(norm_input)
            if self.affine:
                weight_matrix = np.repeat(self.weight.reshape((1, -1)), input.shape[0], axis=0)
                grad_dl_d_norm_input = grad_output * weight_matrix
            else:
                grad_dl_d_norm_input = grad_output
            # dl / dx
            d_norm_input_dx = np.repeat(self.inv_sqrt_var.reshape((1, -1)), input.shape[0], axis=0)
            grad_dl_dx = grad_dl_d_norm_input * d_norm_input_dx
        return grad_dl_dx

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        if self.affine:
            self.grad_weight += (grad_output * self.norm_input).sum(axis=0)
            self.grad_bias += grad_output.sum(axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            # train mode
            # rng = np.random.default_rng()
            # self.mask = rng.binomial(1, 1. - self.p, input.shape)
            self.mask = np.random.binomial(1, 1. - self.p, input.shape)
            output = (1. / (1. - self.p)) * self.mask * input
        else:
            # eval mode
            output = input
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            # train mode
            grad_dl_dx = grad_output * (1. / (1. - self.p)) * self.mask
        else:
            # eval mode
            grad_dl_dx = grad_output
        return grad_dl_dx

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        output = None
        for module_it in range(len(self.modules)):
            output = self.modules[module_it].compute_output(input)
            input = output
        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        grad_input = None
        # computing each module's input
        module_inputs = []
        module_input = input
        for module_it in range(len(self.modules) - 1):
            module_inputs.append(module_input)
            module_output = self.modules[module_it].compute_output(module_input)
            module_input = module_output
        module_inputs.append(module_input)

        # computing gradients
        for module_it in range(len(self.modules) - 1, -1, -1):
            module_input = module_inputs[module_it]
            grad_input = self.modules[module_it].backward(module_input, grad_output)
            grad_output = grad_input
        return grad_input

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
