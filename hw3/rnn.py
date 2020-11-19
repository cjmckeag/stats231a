import numpy as np
import math
import matplotlib.pyplot as plt

# create the training and testing data
# create sine-wave-like data
sin_wave = np.array([math.sin(x) for x in np.arange(200)])

X = []
Y = []

seq_len = 50
num_records = len(sin_wave) - seq_len

for i in range(num_records - 50):
    X.append(sin_wave[i:i + seq_len])
    Y.append(sin_wave[i + seq_len])

X = np.array(X)
X = np.expand_dims(X, axis=2)

Y = np.array(Y)
Y = np.expand_dims(Y, axis=1)

X_val = []
Y_val = []

# set aside 50 records as validation data
for i in range(num_records - 50, num_records):
    X_val.append(sin_wave[i:i + seq_len])
    Y_val.append(sin_wave[i + seq_len])

X_val = np.array(X_val)
X_val = np.expand_dims(X_val, axis=2)

Y_val = np.array(Y_val)
Y_val = np.expand_dims(Y_val, axis=1)


# define the activation function, sigmoid
# used in hidden layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# step 2: train the model
# train until convergence, stop if overfit, or predefine number of epochs
def train(X, Y, learning_rate=0.0001, nepoch=25, T=50, hidden_dim=100, output_dim=1, bptt_truncate=5,
          min_clip_value=-10, max_clip_value=10):
    # define the weights of the network
    # U: matrix for weights between input and hidden layers
    # V: matrix for weights between hidden and output layers
    # W: matrix for shared weights in the RNN (hidden) layer
    U = np.random.uniform(0, 1, (hidden_dim, T))
    W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
    V = np.random.uniform(0, 1, (output_dim, hidden_dim))
    # val loss list
    val_losses = []
    # step 2.1: check the loss on training data
    for epoch in range(nepoch):
        # check loss on train
        loss = 0.0

        # do a forward pass to get prediction
        for i in range(Y.shape[0]):
            # get input, output values of each record
            x, y = X[i], Y[i]
            # prev-s is the value of the previous activation of hidden layer
            # initialized as all zeroes
            prev_s = np.zeros((hidden_dim, 1))
            for t in range(T):
                # do a forward pass for every timestep in the sequence
                new_input = np.zeros(x.shape)
                # define a single input for that timestep
                new_input[t] = x[t]
                # multiply input by weights before hidden layers
                mulu = np.dot(U, new_input)
                # multiply prev activation by shared weights of RNN layer
                mulw = np.dot(W, prev_s)
                add = mulw + mulu
                # activation function
                s = sigmoid(add)
                # multiply activated s by weights before output
                mulv = np.dot(V, s)
                prev_s = s

            # calculate squared error to get the loss value
            loss_per_record = (y - mulv) ** 2 / 2
            loss += loss_per_record
        loss = loss / float(y.shape[0])

        # step 2.2: check the loss on validation data
        val_loss = 0.0
        # same algorithm
        for i in range(Y_val.shape[0]):
            x, y = X_val[i], Y_val[i]
            prev_s = np.zeros((hidden_dim, 1))
            for t in range(T):
                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                mulu = np.dot(U, new_input)
                mulw = np.dot(W, prev_s)
                add = mulw + mulu
                s = sigmoid(add)
                mulv = np.dot(V, s)
                prev_s = s

            loss_per_record = (y - mulv) ** 2 / 2
            val_loss += loss_per_record
        val_loss = val_loss / float(y.shape[0])
        val_losses.append(val_loss)

        print('Epoch: ', epoch + 1, ', Loss: ', loss, ', Val Loss: ', val_loss)

        # step 2.3: start actual training
        # step 2.3.1: forward pass
        for i in range(Y.shape[0]):
            # initialization
            x, y = X[i], Y[i]

            layers = []
            prev_s = np.zeros((hidden_dim, 1))
            dU = np.zeros(U.shape)
            dV = np.zeros(V.shape)
            dW = np.zeros(W.shape)

            dU_t = np.zeros(U.shape)
            dW_t = np.zeros(W.shape)

            # forward pass
            for t in range(T):
                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                # multiply the input with the weights between the input and hidden layers
                mulu = np.dot(U, new_input)
                mulw = np.dot(W, prev_s)
                # add with the multiplication of weights in the RNN layer
                # captures knowledge of previous timestep
                add = mulw + mulu
                # pass through sigmoid activation function
                s = sigmoid(add)
                # multiply with weights btwn hidden and output layers
                mulv = np.dot(V, s)
                # save the current layer state and previous timestep state
                layers.append({'s': s, 'prev_s': prev_s})
                prev_s = s

            # derivative of pred
            dmulv = (mulv - y)

            # backward pass
            for t in range(T):
                # calculate the gradients at each layer
                dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
                dsv = np.dot(np.transpose(V), dmulv)

                ds = dsv
                dadd = add * (1 - add) * ds

                dmulw = dadd * np.ones_like(mulw)

                dprev_s = np.dot(np.transpose(W), dmulw)

                # truncated back propagation through time (TBPTT)
                for i in range(t - 1, max(-1, t - bptt_truncate - 1), -1):
                    ds = dsv + dprev_s
                    dadd = add * (1 - add) * ds

                    dmulw = dadd * np.ones_like(mulw)

                    dW_i = np.dot(W, layers[t]['prev_s'])
                    dprev_s = np.dot(np.transpose(W), dmulw)

                    new_input = np.zeros(x.shape)
                    new_input[t] = x[t]
                    dU_i = np.dot(U, new_input)

                    dU_t += dU_i
                    dW_t += dW_i

                dV += dV_t
                dU += dU_t
                dW += dW_t

                # update the weights with the gradients of weights calculated
                # clamp them in a range so that they don't explode
                if dU.max() > max_clip_value:
                    dU[dU > max_clip_value] = max_clip_value
                if dV.max() > max_clip_value:
                    dV[dV > max_clip_value] = max_clip_value
                if dW.max() > max_clip_value:
                    dW[dW > max_clip_value] = max_clip_value

                if dU.min() < min_clip_value:
                    dU[dU < min_clip_value] = min_clip_value
                if dV.min() < min_clip_value:
                    dV[dV < min_clip_value] = min_clip_value
                if dW.min() < min_clip_value:
                    dW[dW < min_clip_value] = min_clip_value

            # update
            U -= learning_rate * dU
            V -= learning_rate * dV
            W -= learning_rate * dW

    return U, V, W, val_losses, hidden_dim, T


# step 3: get predictions
# training set predictions
U, V, W, val_losses, hidden_dim, T = train(X, Y)
preds = []
for i in range(Y.shape[0]):
    x, y = X[i], Y[i]
    prev_s = np.zeros((hidden_dim, 1))
    # Forward pass through the trained weights
    for t in range(T):
        mulu = np.dot(U, x)
        mulw = np.dot(W, prev_s)
        add = mulw + mulu
        s = sigmoid(add)
        mulv = np.dot(V, s)
        prev_s = s

    preds.append(mulv)

preds = np.array(preds)

plt.plot(preds[:, 0, 0], 'g')
plt.plot(Y[:, 0], 'r')
plt.show()

# testing set predictions
preds = []
for i in range(Y_val.shape[0]):
    x, y = X_val[i], Y_val[i]
    prev_s = np.zeros((hidden_dim, 1))
    # For each time step...
    for t in range(T):
        mulu = np.dot(U, x)
        mulw = np.dot(W, prev_s)
        add = mulw + mulu
        s = sigmoid(add)
        mulv = np.dot(V, s)
        prev_s = s

    preds.append(mulv)

preds = np.array(preds)

plt.plot(preds[:, 0, 0], 'g')
plt.plot(Y_val[:, 0], 'r')
plt.show()
