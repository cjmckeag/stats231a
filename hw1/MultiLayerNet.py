import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

NN_ARCHITECTURE = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

# number of samples in the data set
n_samples = 1000
# ratio between training and test sets
test_size = 0.1

X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=100)


class MultiLayerNet(object):
    """
    A class to build a multilayer neural network.
    ...

    Attributes
    ----------
    nn_architecture : list
        describes the architecture of the desired network. list of dictionaries, each of the format
        {'input_dim': n, 'output_dim': m, 'activation': "sigmoid" or "relu"}.
    seed : int
        initialize random seed for deterministic output, default is 99.

    Methods
    -------
    sigmoid(Z):
        Provides the sigmoid function to use in hidden layer.
    relu(Z):
        Provides the ReLU function to use in hidden layer.
    sigmoid_backward(dA, Z):
        Provides the gradient of the sigmoid function.
    relu_backward(dA, Z):
        Provides the gradient of the ReLU function.
    get_cost_value(Y_hat, Y):
        Computes cost of predictions for true y-values.
    convert_prob_into_class(probs):
        Converts values in probs > 0.5 to 1, <= 0.5 to 0.
    init_layers():
        Uses nn_architecture attribute to initialize network layers and weights, and store parameters.
    single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
        Performs forward propagation for one layer.
    full_forward_propagation(X, params_values):
        Performs the full forward propagation step, returning predictions.
    single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        Performs backward propagation for one layer.
    full_backward_propagation(Y_hat, Y, memory, params_values):
        Performs the full backward propagation step, returning gradient values.
    get_accuracy_value(Y_hat, Y):
        Uses convert_prob_into_class to compute accuracy of predictions for true y-values.
    update(params_values, grads_values, learning_rate):
        Updates the weights of the hidden layers.
    train(X, Y, epochs, learning_rate, verbose=False, callback=None):
        Culminates entire network building; initializes neural net parameters, iterates by performing
        step forward, calculating metrics, performing step backward, and updating model state.
    """
    def __init__(self, nn_architecture, seed=99):
        """
        Constructs all the necessary attributes for the TwoLayerNet object.

        Parameters
        ----------
        nn_architecture : list
            describes the architecture of the desired network. list of dictionaries, each of the format
            {'input_dim': n, 'output_dim': m, 'activation': "sigmoid" or "relu"}.
        seed : int
            initialize random seed for deterministic output, default is 99.
        """
        self.nn_architecture = nn_architecture
        self.seed = seed

    def __call__(self, X, y, test_size, learning_rate=0.01, verbose=True):
        # train test split on data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Training
        params_values = self.train(np.transpose(X_train),
                                   np.transpose(y_train.reshape((y_train.shape[0], 1))),
                                   epochs=10000,
                                   learning_rate=learning_rate,
                                   verbose=verbose)

        # Prediction
        Y_test_hat, _ = self.full_forward_propagation(np.transpose(X_test), params_values)

        # Accuracy achieved on the test set
        acc_test = self.get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))

        print("Test set accuracy: {:.2f}".format(acc_test))
        return(acc_test)

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def sigmoid_backward(dA, Z):
        sig = 1 / (1 + np.exp(-Z))
        return dA * sig * (1 - sig)

    @staticmethod
    def relu_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    @staticmethod
    def get_cost_value(Y_hat, Y):
        # number of examples
        m = Y_hat.shape[1]
        # calculation of the cost according to the formula
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    # an auxiliary function that converts probability into class
    @staticmethod
    def convert_prob_into_class(probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    def init_layers(self):
        # random seed initiation
        np.random.seed(self.seed)
        # parameters storage initiation
        params_values = {}

        # iteration over network layers
        for idx, layer in enumerate(self.nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1

            # extracting the number of units in layers
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            # initiating the values of the W matrix
            # and vector b for subsequent layers
            params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.1
            params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * 0.1

        return params_values

    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="relu"):
        # calculation of the input value for the activation function
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        # selection of activation function
        if activation is "relu":
            activation_func = self.relu
        elif activation is "sigmoid":
            activation_func = self.sigmoid
        else:
            raise Exception('Non-supported activation function')

        # return of calculated activation A and the intermediate Z matrix
        return activation_func(Z_curr), Z_curr

    def full_forward_propagation(self, X, params_values):
        # creating a temporary memory to store the information needed for a backward step
        memory = {}
        # X vector is the activation for layer 0 
        A_curr = X

        # iteration over network layers
        for idx, layer in enumerate(self.nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1
            # transfer the activation from the previous iteration
            A_prev = A_curr

            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]
            # extraction of W for the current layer
            W_curr = params_values["W" + str(layer_idx)]
            # extraction of b for the current layer
            b_curr = params_values["b" + str(layer_idx)]
            # calculation of activation for the current layer
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            # saving calculated values in the memory
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        # return of prediction vector and a dictionary containing intermediate values
        return A_curr, memory

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        # number of examples
        m = A_prev.shape[1]

        # selection of activation function
        if activation is "relu":
            backward_activation_func = self.relu_backward
        elif activation is "sigmoid":
            backward_activation_func = self.sigmoid_backward
        else:
            raise Exception('Non-supported activation function')

        # calculation of the activation function derivative
        dZ_curr = backward_activation_func(dA_curr, Z_curr)

        # derivative of the matrix W
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        # derivative of the vector b
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        # derivative of the matrix A_prev
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, Y_hat, Y, memory, params_values):
        grads_values = {}

        # a hack ensuring the same shape of the prediction vector and labels vector
        Y = Y.reshape(Y_hat.shape)

        # initiation of gradient descent algorithm
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            # we number network layers from 1
            layer_idx_curr = layer_idx_prev + 1
            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]

            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def update(self, params_values, grads_values, learning_rate):
        # iteration over network layers
        for layer_idx, layer in enumerate(self.nn_architecture, 1):
            params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
            params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
        return params_values

    def train(self, X, Y, epochs, learning_rate, verbose=False, callback=None):
        # initiation of neural net parameters
        params_values = self.init_layers()
        # initiation of lists storing the history
        # of metrics calculated during the learning process
        cost_history = []
        accuracy_history = []

        # performing calculations for subsequent iterations
        for i in range(epochs):
            # step forward
            Y_hat, cache = self.full_forward_propagation(X, params_values)

            # calculating metrics and saving them in history
            cost = self.get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = self.get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)

            # step backward - calculating gradient
            grads_values = self.full_backward_propagation(Y_hat, Y, cache, params_values)
            # updating model state
            params_values = self.update(params_values, grads_values, learning_rate)

            if i % (epochs//10) == 0:
                if verbose:
                    print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
                if callback is not None:
                    callback(i, params_values)

        return params_values