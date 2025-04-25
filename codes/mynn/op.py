from abc import abstractmethod
import numpy as np


class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """

    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False,
                 weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None  # Record the input for backward process.
        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay  # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda  # control the intensity of weight decay

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = np.dot(X, self.params['W']) + self.params['b']
        return output

    def backward(self, grad: np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # print('shape_grad:',grad.shape)
        # print('shape_input:',self.input.shape)
        # print('shape_W:',self.params['W'].shape)
        self.grads['W'] = np.dot(grad.T, self.input).T  # W的梯度
        if self.weight_decay:
            reg = L2Regularization(self.weight_decay_lambda)
            self.grads['W'] += reg.grad(self.params['W'])
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)  # b的梯度
        return np.dot(grad, self.params['W'].T)

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}



class conv2D(Layer):
    """
    The 2D convolutional layer. Implementing forward and backward pass.
    """

    def __init__(self, shape, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.initialize_method = initialize_method
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        self.input = X
        batch_size, in_channels, H, W = X.shape

        # Add padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              mode='constant')
        else:
            X_padded = X

        # Calculate output dimensions
        out_H = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_W = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_H, out_W))

        # Perform convolution using efficient einsum (optimized for performance)
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # Extract the current window
                window = X_padded[:, :, h_start:h_end, w_start:w_end]

                # Use einsum for efficient batch matrix multiplication
                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(window * self.params['W'][k, :, :, :], axis=(1, 2, 3)) + self.params['b'][k]

        return output

    def backward(self, grads):
        """
        Backpropagate the gradients through the convolutional layer.
        grads: [batch_size, out_channels, new_H, new_W]
        """
        X = self.input
        batch_size, in_channels, H, W = X.shape
        _, _, out_H, out_W = grads.shape

        # Initialize gradients
        dW = np.zeros_like(self.params['W'])
        db = np.zeros_like(self.params['b'])
        dX = np.zeros_like(X)

        # Add padding to input if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              mode='constant')
            dX_padded = np.pad(dX, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                               mode='constant')
        else:
            X_padded = X
            dX_padded = dX

        # Compute gradients using optimized vectorized operations
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # Extract the current window
                window = X_padded[:, :, h_start:h_end, w_start:w_end]

                # Compute gradients for weights and bias
                for k in range(self.out_channels):
                    # Gradient for weights
                    dW[k] += np.sum(window * grads[:, k, i, j].reshape(-1, 1, 1, 1), axis=0)

                    # Gradient for bias
                    db[k] += np.sum(grads[:, k, i, j])

                    # Gradient for input
                    dX_padded[:, :, h_start:h_end, w_start:w_end] += self.params['W'][k] * grads[:, k, i, j].reshape(-1, 1, 1, 1)

        # Remove padding from input gradient if needed
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        # Store gradients
        self.grads['W'] = dW / batch_size
        self.grads['b'] = db / batch_size

        # Add weight decay if enabled
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.params['W']

        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}



class Pool2D(Layer):
    def __init__(self, size, stride, mode="max"):
        """
        :param size: The size of the pooling window
        :param stride: The stride of the pooling window
        :param mode: Pooling mode, either "max" or "mean"
        """
        super().__init__()
        self.params = {}
        self.size = size
        self.stride = stride
        self.mode = mode
        self.input = None  # Store the input for backward propagation

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        '''
        pooling layer only use the covered region
        '''
        batch, channels, H, W = X.shape
        self.input = X
        output_height = (H - self.size) // self.stride + 1
        output_width = (W - self.size) // self.stride + 1
        output = np.zeros([batch, channels, output_height, output_width])
        self.arg_max = np.zeros([batch, channels, output_height, output_width], dtype=int)

        for b in range(batch):
            for o in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        height_start = i * self.stride
                        height_end = height_start + self.size
                        width_start = j * self.stride
                        width_end = width_start + self.size

                        window = X[b, o, height_start:height_end, width_start:width_end]

                        if self.mode == "max":
                            output[b, o, i, j] = np.max(window)
                            # Store the index of the maximum value for backpropagation
                            self.arg_max[b, o, i, j] = np.argmax(window)
                        elif self.mode == "mean":
                            output[b, o, i, j] = np.mean(window)

        return output

    def backward(self, grads):
        batch_size, out_channels, new_H, new_W = grads.shape
        grad_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for o in range(out_channels):
                for i in range(new_H):
                    for j in range(new_W):
                        height_start = i * self.stride
                        height_end = height_start + self.size
                        width_start = j * self.stride
                        width_end = width_start + self.size

                        if self.mode == "max":
                            max_idx = self.arg_max[b, o, i, j]
                            grad_input[b, o, height_start + max_idx // self.size, width_start + max_idx % self.size] += grads[b, o, i, j]

                        elif self.mode == "mean":
                            grad_input[b, o, height_start:height_end, width_start:width_end] += grads[b, o, i, j] / (self.size * self.size)

        return grad_input

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.params = {}

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        output = X.reshape(X.shape[0], -1)
        return output

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class ReLU(Layer):
    """
    An activation layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X < 0, 0, X)
        return output

    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """

    def __init__(self, model=None, max_classes=10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.grads = None
        self.predicts = None
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        loss = 0
        self.predicts = predicts
        self.labels = labels
        if self.has_softmax:
            soft_max_predicts = self.softmax(predicts)
            for i in range(predicts.shape[0]):
                label = labels[i]
                loss = loss - np.log(soft_max_predicts[i][label] + 1e-15)
        else:
            for i in range(predicts.shape[0]):
                label = labels[i]
                loss = loss - np.log(predicts[i][label] + 1e-15)
        return loss / predicts.shape[0]

    def backward(self):
        batch_size = self.predicts.shape[0]

        if self.has_softmax:
            probs = self.softmax(self.predicts)
            self.grads = probs.copy()
            self.grads[np.arange(batch_size), self.labels] -= 1
            # 平均梯度
            self.grads /= batch_size
        else:
            # 当没有softmax时，假设输入已经是概率分布
            self.grads = np.zeros_like(self.predicts)
            # 只对正确类别计算梯度
            self.grads[np.arange(batch_size), self.labels] = -1.0 / (
                        self.predicts[np.arange(batch_size), self.labels] + 1e-15)
            # 平均梯度
            self.grads /= batch_size

        if self.model is not None:
            self.model.backward(self.grads)

    def softmax(self, x):
        """
        Compute softmax for each row (each sample)
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """

    def __init__(self, lamda):
        self.lamda = lamda

    def grad(self, W):
        return 2 * self.lamda * W


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.optimizable = False
    def __call__(self, X):
        return self.forward(X)
    def forward(self, x):
        self.mask = np.random.rand(*x.shape) > self.dropout_ratio
        return x * self.mask
    def backward(self, grad):
        return grad * self.mask