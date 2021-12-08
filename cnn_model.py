import pdb
import time
from tqdm.notebook import tqdm
import numpy as np

DATA_TYPE = np.float32
EPSILON = 1e-12


def calculate_fan_in_and_fan_out(shape):
    """
    :param shape: Tuple of shape, e.g. (120,84) for the weight in a FC layer or (5,5,3,6) for the filter in a conv layer
    :return: fan_in, fan_out, representing the number of input parameter and output parameter
    """
    if len(shape) < 2:
        raise ValueError(
            "Unable to calculate fan_in and fan_out with dimension less than 2")
    elif len(shape) == 2:  # Weight of a FC Layer
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:  # filter of a convolutional layer
        fan_in = np.prod(shape[:-1])
        fan_out = shape[-1] * np.prod(shape[:-2])
    else:
        raise ValueError(
            f"Shape {shape} not supported in calculate_fan_in_and_fan_out")
    return fan_in, fan_out


def xavier(shape, seed=None):
    n_in, n_out = calculate_fan_in_and_fan_out(shape)
    if seed is not None:
        # set seed to fixed number (e.g. layer idx) for predictable results
        np.random.seed(seed)
    xavarian_matrix = [[np.random.uniform(-np.sqrt(6/(n_in+n_out)),
                                          np.sqrt(6/(n_in+n_out))) for j in range(n_out)] for i in range(n_in)]
    return np.array(xavarian_matrix, dtype=np.float32)


# InputValue: These are input values. They are leaves in the computational graph.
#              Hence we never compute the gradient wrt them.
class InputValue:
    def __init__(self, value=None):
        self.value = DATA_TYPE(value).copy()
        self.grad = None

    def set(self, value):
        self.value = DATA_TYPE(value).copy()


# Parameters: Class for weight and biases, the trainable parameters whose values need to be updated
class Param:
    def __init__(self, value):
        self.value = DATA_TYPE(value).copy()
        self.grad = DATA_TYPE(0)


class Add:  # Add with broadcasting
    '''
    Class name: Add
    Class usage: add two matrices a, b with broadcasting supported by numpy "+" operation.
    Class function:
        forward: calculate a + b with possible broadcasting
        backward: calculate derivative w.r.t to a and b
    '''

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value + self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad

        if self.b.grad is not None:
            self.b.grad = self.b.grad + self.grad


class Mul:  # Multiply with broadcasting
    '''
    Class Name: Mul
    Class Usage: elementwise multiplication with two matrix
    Class Functions:
        forward: compute the result a*b
        backward: compute the derivative w.r.t a and b
    '''

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value * self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad * self.b.value

        if self.b.grad is not None:
            self.b.grad = self.b.grad + self.grad * self.a.value


class VDot:  # Matrix multiply (fully-connected layer)
    '''
    Class Name: VDot
    Class Usage: matrix multiplication where a is a vector and b is a matrix
        b is expected to be a parameter and there is a convention that parameters come last.
        Typical usage is a is a feature vector with shape (f_dim, ), b a parameter with shape (f_dim, f_dim2).
    Class Functions:
        forward: compute the vector matrix multplication result
        backward: compute the derivative w.r.t a and b, where derivative of a and b are both matrices
    '''

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = []
        for i in range(len(self.b.value[0])):
            summation = 0
            for j in range(len(self.b.value)):
                summation += self.a.value[j] * self.b.value[j][i]
            self.value.append(summation)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = []
            for i in range(len(self.b.value)):
                summation = 0
                for j in range(len(self.b.value[i])):
                    summation += self.b.value[i][j] * self.grad[j]
                self.a.grad.append(summation)
        self.a.grad = np.array(self.a.grad, dtype=np.float32)

        if self.b.grad is not None:
            self.b.grad = []
            for i in range(len(self.b.value)):
                grad_row = []
                for j in range(len(self.b.value[i])):
                    grad_row.append(self.grad[j] * self.a.value[i])
                self.b.grad.append(grad_row)
        self.b.grad = np.array(self.b.grad, dtype=np.float32)


class Sigmoid:
    '''
    Class Name: Sigmoid
    Class Usage: compute the elementwise sigmoid activation. Input is vector or matrix.
        In case of vector, [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = 1/(1 + exp(-a_{i}))
    Class Functions:
        forward: compute activation b_{i} for all i.
        backward: compute the derivative w.r.t input vector/matrix a
    '''

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        if len(self.a.value.shape) == 1:  # if vector
            self.value = [1/(1+np.exp(-self.a.value[i]))
                          for i in range(len(self.a.value))]
        else:  # if matrix
            self.value = []
            for i in range(len(self.a.value)):
                subvalue = []
                for j in range(len(self.a.value[i])):
                    subvalue.append(1/(1+np.exp(-self.a.value[i][j])))
                self.value.append(subvalue)
        self.value = np.array(self.value, dtype=np.float32)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = []
            for i in range(len(self.a.value)):
                self.a.grad.append(
                    self.grad[i]*self.value[i]*(1-self.value[i]))
        self.a.grad = np.array(self.a.grad, dtype=np.float32)


class RELU:
    '''
    Class Name: RELU
    Class Usage: compute the elementwise RELU activation. Input is vector or matrix. In case of vector,
        [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = max(0, a_{i})
    Class Functions:
        forward: compute activation b_{i} for all i.
        backward: compute the derivative w.r.t input vector/matrix a
    '''

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.maximum(self.a.value, np.zeros_like(self.a.value))

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = np.maximum(self.grad, np.zeros_like(self.a.value))


class SoftMax:
    '''
    Class Name: SoftMax
    Class Usage: compute the softmax activation for each element in the matrix, normalization by each all elements
        in each batch (row). Specifically, input is matrix [a_{00}, a_{01}, ..., a_{0n}, ..., a_{b0}, a_{b1}, ..., a_{bn}],
        output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(a_{bi})/(exp(a_{b0}) + ... + exp(a_{bn}))
    Class Functions:
        forward: compute probability p_{bi} for all b, i.
        backward: compute the derivative w.r.t input matrix a
    '''

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        summation = 0.0
        for i in range(len(self.a.value)):
            summation += np.exp(self.a.value[i])
        self.value = [np.exp(self.a.value[i]) /
                      summation for i in range(len(self.a.value))]
        self.value = np.array(self.value, dtype=np.float32)

    def backward(self):
        if self.a.grad is not None:
            yhat = self.value
            summations = []
            for i in range(len(yhat)):
                summation = 0
                for j in range(len(yhat)):
                    summation += self.grad[j]*yhat[j]*yhat[i]
                summations.append(summation)
            dytilde = [self.grad[i]*yhat[i]-summations[i]
                       for i in range(len(yhat))]
            self.a.grad = np.array(dytilde, dtype=np.float32)
            print(self.a.grad)


class Log:  # Elementwise Log
    '''
    Class Name: Log
    Class Usage: compute the elementwise log(a) given a.
    Class Functions:
        forward: compute log(a)
        backward: compute the derivative w.r.t input vector a
    '''

    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.array([np.log(self.a.value[i])
                               for i in range(len(self.a.value))], dtype=np.float32)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = []
            for i in range(len(self.a.value)):
                self.a.grad.append(self.grad[i] / self.a.value[i])
        self.a.grad = np.array(self.a.grad, dtype=np.float32)


class Aref:
    '''
    Class Name: Aref
    Class Usage: get some specific entry in a matrix. a is the matrix with shape (batch_size, N) and idx is vector containing
        the entry index and a is differentiable.
    Class Functions:
        forward: compute a[batch_size, idx]
        backward: compute the derivative w.r.t input matrix a
    '''

    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None if a.grad is None else DATA_TYPE(0)

    def forward(self):
        xflat = self.a.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        self.pick = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.a.grad is not None:
            grad = np.zeros_like(self.a.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.a.grad = self.a.grad + grad


class Accuracy:
    '''
    Class Name: Accuracy
    Class Usage: check the predicted label is correct or not. a is the probability vector where each probability is
                for each class. idx is ground truth label.
    Class Functions:
        forward: find the label that has maximum probability and compare it with the ground truth label.
        backward: None
    '''

    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None
        self.value = None

    def forward(self):
        self.value = np.mean(
            np.argmax(self.a.value, axis=-1) == self.idx.value)

    def backward(self):
        pass


class Conv:
    '''
    Class Name: Conv
    Class Usage: convolutional layer that performs elementwise multiplication within the rolling window
                and output the sum of products to the corresponding cell.
    Class Functions:
        forward: Calculate the output of convolutional layer
        backward: Calculate the derivative w.r.t. the input tensor and kernel
    '''

    def __init__(self, input_tensor, kernel, stride=1, padding=0):
        """
        A “Kernel” refers to a 2D array of weights. The term “filter” is for 3D structures of multiple kernels stacked together.
        For a 2D filter, filter is same as kernel. But for a 3D filter and most convolutions in deep learning, a filter is a collection of kernels

        :param input_tensor: input tensor of size (height, width, in_channels)
        :param kernel: convolving kernel of size (kernel_size, kernel_size, in_channels, out_channels),
                        only square kernels of size (kernel_size, kernel_size) are supported
        :param stride: stride of convolution. Default: 1
        :param padding: zero-padding added to both sides of the input. Default: 0
        """
        self.kernel = kernel
        self.input_tensor = input_tensor
        self.padding = padding
        self.stride = stride
        self.grad = None if kernel.grad is None and input_tensor.grad is None else DATA_TYPE(
            0)
        self.value = None

    def forward(self):
        """
         calculate self.value of size (output_height, output_width, out_channels)
         You can assume stride=1 and padding=0 for simplicity. Support of stride>1 and padding>0 will earn extra credits.
        """
        height, width, in_channels = self.input_tensor.value.shape
        kernel_size = self.kernel.value.shape[0]
        output_channels = self.kernel.value.shape[3]  # 4
        padded_input = np.zeros(
            (height + 2 * self.padding, width + 2 * self.padding, in_channels))
        padded_input[self.padding:(
            self.padding + height), self.padding:(self.padding + width), :] = self.input_tensor.value
        output_height = int((height + 2 * self.padding -
                             kernel_size) / self.stride) + 1
        output_width = int((width + 2 * self.padding -
                            kernel_size) / self.stride) + 1
        self.value = np.zeros((output_height, output_width, output_channels))
        for i in range(output_height):  # 2
            for j in range(output_width):  # 2
                for c in range(output_channels):
                    for g in range(self.kernel.value.shape[2]):
                        for e in range(0, self.kernel.value.shape[0], self.stride):
                            for f in range(0, self.kernel.value.shape[1], self.stride):
                                self.value[i, j, c] += padded_input[e + i,
                                                                    f + j, g] * self.kernel.value[e, f, g, c]

    def backward(self):
        """
         calculate gradient of kernel.grad and input_tensor
         You can assume stride=1 and padding=0 for simplicity. Support of stride>1 and padding>0 will earn extra credits.
        """
        height, width, in_channels = self.input_tensor.value.shape
        kernel_size = self.kernel.value.shape[0]
        output_channels = self.kernel.value.shape[3]
        kernel_grad = np.zeros(self.kernel.value.shape)
        padded_input = np.zeros(
            (height + 2 * self.padding, width + 2 * self.padding, in_channels))
        padded_input[self.padding:(
            self.padding + height), self.padding:(self.padding + width), :] = self.input_tensor.value
        input_grad = np.zeros(padded_input.shape)
        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                for c in range(output_channels):
                    i0 = i * self.stride
                    j0 = j * self.stride
                    for g in range(self.kernel.value.shape[2]):
                        for e in range(0, self.kernel.value.shape[0], self.stride):
                            for f in range(0, self.kernel.value.shape[1], self.stride):
                                kernel_grad[e, f, g, c] += self.grad[i,
                                                                     j, c] * padded_input[e + i, f + j, g]
                                input_grad[e + i, f + j, g] += self.grad[i,
                                                                         j, c] * self.kernel.value[e, f, g, c]
        if self.kernel.grad is not None:
            self.kernel.grad = self.kernel.grad + kernel_grad
        if self.input_tensor.grad is not None:
            self.input_tensor.grad = self.input_tensor.grad + input_grad[self.padding:(self.padding + height),
                                                                         self.padding:(self.padding + width), :]


class MaxPool:
    '''
    Class Name: MaxPool
    Class Usage: Applies a max pooling over an input signal composed of several input planes.
    Class Functions:
        forward: Calculate the output of convolutional layer
        backward: Calculate the derivative w.r.t. the input tensor and kernel
    '''

    def __init__(self, input_tensor, kernel_size=2, stride=None):
        """
        :param input_tensor: input tensor of size (height, width, in_channels)
        :param kernel_size: the size of the window to take a max over. Default: 2
        :param stride: the stride of the window. Default value is kernel_size
        """
        self.input_tensor = input_tensor
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.grad = None if input_tensor.grad is None else DATA_TYPE(0)
        self.value = None
        self.max_index = None

    def forward(self):
        """
        calculate self.value of size (int(height / self.stride), int(width / self.stride), in_channels)
        You can assume stride=kernel_size for simplicity. Support of stride!=kernel_size will earn extra credits.
        """
        height, width, in_channels = self.input_tensor.value.shape
        output_height = int(height / self.stride)
        output_width = int(width / self.stride)
        self.value = np.zeros((output_height, output_width, in_channels))
        self.max_index = np.zeros((height, width, in_channels))
        for c in range(in_channels):
            for i in range(output_height):
                for j in range(output_width):
                    self.value[i, j, c] += max(self.input_tensor.value[self.stride*i, self.stride*j, c], self.input_tensor.value[self.stride*i + self.kernel_size - 1, self.stride*j, c],
                                               self.input_tensor.value[self.stride*i, self.stride*j + self.kernel_size - 1, c], self.input_tensor.value[self.stride*i + self.kernel_size - 1, self.stride*j + self.kernel_size - 1, c])
                    if self.value[i, j, c] == self.input_tensor.value[self.stride*i, self.stride*j, c]:
                        self.max_index[self.stride*i, self.stride*j, c] += 1
                    elif self.value[i, j, c] == self.input_tensor.value[self.stride*i + self.kernel_size - 1, self.stride*j, c]:
                        self.max_index[self.stride*i +
                                       self.kernel_size - 1, self.stride*j, c] += 1
                    elif self.value[i, j, c] == self.input_tensor.value[self.stride*i, self.stride*j + self.kernel_size - 1, c]:
                        self.max_index[self.stride*i, self.stride *
                                       j + self.kernel_size - 1, c] += 1
                    else:
                        self.max_index[self.stride*i + self.kernel_size -
                                       1, self.stride*j + self.kernel_size - 1, c] += 1

    def backward(self):
        """
        calculate the gradient for input_tensor
        You can assume stride=kernel_size for simplicity. Support of stride!=kernel_size will earn extra credits.
        """
        height, width, in_channels = self.input_tensor.value.shape
        input_grad = np.zeros(self.input_tensor.value.shape)
        output_height = int(height / self.stride)
        output_width = int(width / self.stride)
        for c in range(in_channels):
            for i in range(output_height):
                for j in range(output_width):
                    input_grad[self.stride*i, self.stride * j, c] += self.grad[i,
                                                                               j, c] * self.max_index[self.stride*i, self.stride*j, c]
                    input_grad[self.stride*i + self.kernel_size - 1, self.stride * j, c] += self.grad[i,
                                                                                                      j, c] * self.max_index[self.stride*i + self.kernel_size - 1, self.stride*j, c]
                    input_grad[self.stride*i, self.stride * j + self.kernel_size - 1, c] += self.grad[i,
                                                                                                      j, c] * self.max_index[self.stride*i, self.stride*j + self.kernel_size - 1, c]
                    input_grad[self.stride*i + self.kernel_size - 1, self.stride * j + self.kernel_size - 1, c] += self.grad[i,
                                                                                                                             j, c] * self.max_index[self.stride*i + self.kernel_size - 1, self.stride*j + self.kernel_size - 1, c]
        self.input_tensor.grad = self.input_tensor.grad + input_grad


class Flatten:
    '''
    Class name: Flatten
    Class usage: Flatten the input tensor to a 1d vector.
    Class function:
        forward: Flatten the input tensor to a 1d vector.
        backward: calculate derivative w.r.t to input_tensor,
                    which is simply reshaping the output gradient to input_tensor's original shape
    '''

    def __init__(self, input_tensor):
        self.input_tensor = input_tensor
        self.grad = None if input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.ndarray.flatten(self.input_tensor.value)
        print(self.value)

    def backward(self):
        if self.input_tensor.grad is not None:
            self.input_tensor.grad = np.zeros_like(self.input_tensor.value)
            v = 0
            for i in range(self.input_tensor.value.shape[0]):
                for j in range(self.input_tensor.value.shape[1]):
                    for c in range(self.input_tensor.value.shape[2]):
                        self.input_tensor.grad[i, j, c] += self.grad[v]
                        v += 1


class CNN:
    def __init__(self, num_labels=10):
        self.num_labels = num_labels
        # dictionary of trainable parameters
        self.params = {}
        # list of computational graph
        self.components = []
        self.sample_placeholder = InputValue()
        self.label_placeholder = InputValue()
        self.pred_placeholder = None
        self.loss_placeholder = None
        self.accy_placeholder = None

    # helper function for creating a unary operation object and add it to the computational graph
    def nn_unary_op(self, op, a):
        unary_op = op(a)
        print(
            f"Append <{unary_op.__class__.__name__}> to the computational graph")
        self.components.append(unary_op)
        return unary_op

    # helper function for creating a binary operation object and add it to the computational graph
    def nn_binary_op(self, op, a, b):
        binary_op = op(a, b)
        print(
            f"Append <{binary_op.__class__.__name__}> to the computational graph")
        self.components.append(binary_op)
        return binary_op

    def conv_op(self, input_tensor, kernel, stride=1, padding=0):
        conv = Conv(input_tensor, kernel, stride=stride, padding=padding)
        print(f"Append <{conv.__class__.__name__}> to the computational graph")
        self.components.append(conv)
        return conv

    def maxpool_op(self, input_tensor, kernel_size=2, stride=None):
        maxpool = MaxPool(input_tensor, kernel_size=kernel_size, stride=stride)
        print(
            f"Append <{maxpool.__class__.__name__}> to the computational graph")
        self.components.append(maxpool)
        return maxpool

    def set_params_by_dict(self, param_dict: dict):
        """
        :param param_dict: a dict of parameters with parameter names as keys and numpy arrays as values
        """
        # reset params to an empty dict before setting new values
        self.params = {}
        # add Param objects to the dictionary of trainable paramters with names and values
        for name, value in param_dict.items():
            self.params[name] = Param(value)

    def get_param_dict(self):
        """
        :return: param_dict: a dict of parameters with parameter names as keys and numpy arrays as values
        """
        param_dict = {
            "conv1_kernel": None,
            "conv1_bias": None,
            "conv2_kernel": None,
            "conv2_bias": None,
            "fc1_weight": None,
            "fc1_bias": None,
            "fc2_weight": None,
            "fc2_bias": None,
            "fc3_weight": None,
            "fc3_bias": None,
        }
        return param_dict

    def init_params_with_xavier(self):
        param_dict = {
            "conv1_kernel": None,
            "conv1_bias": None,
            "conv2_kernel": None,
            "conv2_bias": None,
            "fc1_weight": None,
            "fc1_bias": None,
            "fc2_weight": None,
            "fc2_bias": None,
            "fc3_weight": None,
            "fc3_bias": None,
        }
        self.set_params_by_dict(param_dict)

    def build_computational_graph(self):
        self.components = []
        input_tensor = self.sample_placeholder
        input_tensor.value = np.zeros((32, 32, 3))
        conv1_kernel = InputValue(np.zeros((5, 5, 3, 6)))
        a = self.conv_op(input_tensor, conv1_kernel, stride=1, padding=2)
        conv1_bias = InputValue(np.zeros((5, 5, 3, 6)))
        conv1 = self.nn_binary_op(Add, conv1_kernel, conv1_bias)
        b = RELU(a)
        c = MaxPool(b, 2)
        d = Conv(c, conv1, 1, 2)
        e = RELU(d)
        f = MaxPool(e, 2)
        g = Flatten(f)
        h = Mul(input_tensor, g)
        i = RELU(h)
        j = Mul(input_tensor, i)
        k = RELU(j)
        l = Mul(input_tensor, k)
        pred = SoftMax(l)
        return pred

    def cross_entropy_loss(self):
        label_prob = self.nn_binary_op(
            Aref, self.pred_placeholder, self.label_placeholder)
        log_prob = self.nn_unary_op(Log, label_prob)
        loss = self.nn_binary_op(Mul, log_prob, InputValue(-1))
        return loss

    def eval(self, X, y):
        if len(self.components) == 0:
            raise ValueError(
                "Computational graph not built yet. Call build_computational_graph first.")
        accuracy = 0.
        objective = 0.
        for k in range(len(y)):
            self.sample_placeholder.set(X[k])
            self.label_placeholder.set(y[k])
            self.forward()
            accuracy += self.accy_placeholder.value
            objective += self.loss_placeholder.value
        accuracy /= len(y)
        objective /= len(y)
        return accuracy, objective

    def fit(self, X, y, alpha, t):
        """
        Use the cross entropy loss.  The stochastic
        gradient descent should go through the examples in order, so
        that your output is deterministic and can be verified.
        :param X: an (m, n)-shaped numpy input matrix  
        :param y: an (m,1)-shaped numpy output
        :param alpha: the learning rate
        :param t: the number of iterations
        :return:
        """
        # create sample and input placeholder
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(
            Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in tqdm(range(X.shape[0])):
                # tqdm adds a progress bar
                for p in self.params.values():
                    p.grad = DATA_TYPE(0)
                for c in self.components:
                    if c.grad is not None:
                        c.grad = DATA_TYPE(0)
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])
                for component in self.components:
                    component.forward()
                for component in self.components:
                    component.backward()

            # evaluate on train set
            avg_acc, avg_loss = self.eval(X, y)
            print("Epoch %d: train loss = %.4f, accy = %.4f, [%.3f secs]" % (
                epoch, avg_loss, avg_acc, time.time()-since))
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            since = time.time()

    def forward(self):
        for c in self.components:
            c.forward()

    def backward(self, loss):
        loss.grad = np.ones_like(loss.value)
        for c in self.components[::-1]:
            c.backward()

    # Optimization functions
    def sgd_update_parameter(self, lr):
        # update the parameter values in self.params
        for p in range(len(list(self.params.keys()))):
            self.params[0][p].value += lr * self.params[0][p].grad
            self.params[1][p].value += lr * self.params[1][p].grad
