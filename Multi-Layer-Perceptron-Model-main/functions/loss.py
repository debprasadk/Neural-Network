
import numpy as np


# loss functions calculate loss for each datapoint
# to get the mean loss over all data points call the mean method


class Loss(object):
    def __init__(self): pass

    @classmethod
    def mean_loss(cls, y1: np.ndarray, y2: np.ndarray) -> float:
        """
        mean_loss = 1/n * sum(error_vector)
        n = no of data points
        :param y1: (Output) Matrix of row vectors containing output data points
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
        :return: mean loss
        """
        a = np.sum(cls.loss(y1=y1, y2=y2)) / len(y1)

        return a


class SquaredError(Loss):
    __name__ = 'squarederror'

    @staticmethod
    def loss(y1: np.ndarray, y2: np.ndarray) -> float:
        """
        mean square error
        1/2 * || y1 - y2 ||^2
        :param y1: (Predicted Output) Matrix of row vectors containing output data points
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
        :return: squared errors for each datapoint as a column vector
        """
        return 0.5 * np.square(np.linalg.norm(y1 - y2, axis=1, keepdims=True))

    @staticmethod
    def gradient(y1: np.ndarray, y2: np.ndarray) -> float:
        """
        gradient = predicted_output_vector - desired_output_vector
        :param y1: (Predicted Output) Matrix of row vectors containing output data points
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
        :return: squared error loss gradients as a column vector
        """
        return y1 - y2


class CrossEntropy(Loss):
    __name__ = 'cross-entropy'
    @staticmethod
    def loss(y1: np.ndarray, y2: np.ndarray) -> float:
        """
        Cross entropy loss (for one datapoint)
        - sum(y2_i * log(y1_i))
        :param y1: (Predicted output) Matrix of row vectors containing output data points
                   (values must be b/w 0 and 1 and sum to 1)
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
                   (values must be b/w 0 and 1 and sum to 1)
        :return: cross entropy losses as a column vector
        """

        # TODO: Add check for values b/w 0 and 1 and sum to 1
        # TODO: log function raises divide by zero error when one of the values in y1 is 0
        return -np.sum(y2 * np.log2(y1), axis=1, keepdims=True)

    @staticmethod
    def gradient(y1: np.ndarray, y2: np.ndarray) -> float:
        """
        gradient = desired_output_vector / predicted_output_vector
        :param y1: (Predicted output) Matrix of row vectors containing output data points
                   (values must be b/w 0 and 1 and sum to 1)
        :param y2: (Desired output) Matrix of row vectors containing desired output data points
                   (values must be b/w 0 and 1 and sum to 1)
        :return: cross entropy loss gradients as a column vector
        """

        return - y2 / y1

class Activation(object):
    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        # apply the activation function
        # returns a matrix of row vectors for each datapoint
        pass

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> float:
        # return the gradients for each datapoint
        # as a matrix of row vectors (gradients)
        # OR
        # return the jacobian for each datapoint
        # as a 3D matrix of 2D matrices (jacobians)
        pass


class Identity(Activation):
    __name__ = 'identity'

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        """
        Apply the identity activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :return: Activated output. Matrix of row vectors containing each activated output
        """
        return x

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gradient of the identity activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :param y: (Activated output) Matrix of row vectors containing each activated output
        :return: Gradient of the identity activation function.
                 Matrix of row vectors containing gradient for each activated output
        """
        return np.ones(x.shape)


class Tanh(Activation):
    __name__ = 'tanh'

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        """
        Apply the tanh activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :return: Activated output. Matrix of row vectors containing each activated output
        """
        return 2 / (1 + np.exp(-2 * x)) - 1

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gradient of the tanh activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :param y: (Activated output) Matrix of row vectors containing each activated output
        :return: Gradient of the tanh activation function.
                 Matrix of row vectors containing gradient for each activated output
        """
        a = (y + 1) / 2
        return 4 * (a - np.square(a))


class Sigmoid(Activation):
    __name__ = 'sigmoid'

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :return: Activated output. Matrix of row vectors containing each activated output
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gradient of the sigmoid activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :param y: (Activated output) Matrix of row vectors containing each activated output
        :return: Gradient of the sigmoid activation function.
                 Matrix of row vectors containing gradient for each activated output
        """
        return y * (1-y)


class ReLu(Activation):
    __name__ = 'relu'

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        """
        Apply the ReLu activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :return: Activated output. Matrix of row vectors containing each activated output
        """
        indices = np.nonzero(x > 0)  # indices where value is > 0

        out = np.zeros(x.shape)
        # print(x)
        # print(type(x))
        out[indices] = x[indices]

        return out

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gradient of the ReLu activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :param y: (Activated output) Matrix of row vectors containing each activated output
        :return: Gradient of the ReLu activation function.
                 Matrix of row vectors containing gradient for each activated output
        """
        indices = np.nonzero(x > 0)  # indices where value is > 0

        out = np.zeros(x.shape)
        out[indices] = 1

        return out


class Softmax(Activation):
    __name__ = 'softmax'

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        """
        Apply the softmax activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :return: Activated output. Matrix of row vectors containing each activated output
        """
        a = np.exp(x)
        return a / np.sum(a, axis=1, keepdims=True)

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Jacobians of the softmax activation function
        :param x: (Input) Matrix of row vectors containing each data point
        :param y: (Activated output) Matrix of row vectors containing each activated output
        :return: Jacobians of the softmax activation function.
                 Matrix of matrices containing jacobian for each activated output
        """
        # returns a 3D matrix of matrices (jacobians)
        # since function is not element wise

        # for one datapoint
        # jacobian = diagonal matrix - outer product
        # jacobian = np.diagflat(y) - np.dot(y, y.T)

        # create matrix of diagonal matrices of datapoints (diagflat along 2nd axis)
        # ref: https://stackoverflow.com/questions/52443302/numpy-diagflat-for-specified-axis
        # first one is a matrix of column vectors of datapoints
        # second one is an identity matrix of size of datapoint
        diag = y[..., np.newaxis] * np.eye(y.shape[1])

        # create matrix of outer product of datapoints (outer product along 2nd axis)
        # ref: https://stackoverflow.com/questions/42378936/numpy-elementwise-outer-product
        # first one is a matrix of column vectors
        # second one is a matrix of row vectors
        # multiply here acts as outer product for each datapoint (along the first axis)
        outer = y[..., np.newaxis] * y[:, np.newaxis]

        # create matrix of jacobians
        jacobians = diag - outer

        return jacobians

    jacobian = gradient