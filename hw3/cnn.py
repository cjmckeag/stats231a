import numpy as np
import gzip
import pickle
from tqdm import tqdm


# Step 1: Getting the data
def extract_data(filename, num_images, image_width):
    """
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w],
    where m is the number of training examples.

    Parameters
    ----------
    filename : str
        name of the .gz file containing the data
    num_images : int
        number of images to read from the file
    image_width : int
        pixel width of images to read

    Returns
    -------
    data : np.ndarray
        num_images x image_width*image_width numpy array containing pixel data
    """
    print('Extracting', filename)
    # access the bytestream of the tensors
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        # create buffer from bytestream
        buf = bytestream.read(image_width * image_width * num_images)
        # convert buffer into 1D numpy array
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # reshape array into (num_images, image_width*image_width) shape array
        data = data.reshape(num_images, image_width * image_width)
        return data


def extract_labels(filename, num_images):
    """
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.

    Parameters
    ----------
    filename : str
        name of the .gz file containing the labels
    num_images : int
        number of labels to read from the file

    Returns
    -------
    labels : np.ndarray
        1D numpy array containing corresponding image labels
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        # create labels buffer from bytestream
        buf = bytestream.read(1 * num_images)
        # convert buffer into 1D numpy array of labels
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


# Step 2: Initialize parameters
def initialize_filter(size, scale=1.0):
    """
    Initialize the filters for the convolutional layers using a normal distribution with a
    standard deviation inversely proportional to the square root of the number of units.
    Using the standard normal distribution makes for a smoother training process.

    Parameters
    ----------
    size : Tuple[int, int, int, int]
        number of units
    scale : float
        scale, default=1.0

    Returns
    -------
    (size) random normal RVs with mean 0 and scale (scale/np.sqrt(np.prod(size)))
    """
    # create stddev inverse prop to sqrt(number of units)
    stddev = scale / np.sqrt(np.prod(size))
    # return a normal RV with mean 0 and using stddev
    return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weight(size):
    """
    Initialize the weights for the dense layers with a random normal distribution.

    Parameters
    ----------
    size : Tuple[int, int]
        number of units

    Returns
    -------
    (size) standard normal RVs * 0.01
    """
    # output a standard normal RV * 0.01.
    return np.random.standard_normal(size=size) * 0.01


# Step 3: Define the backpropagation operations
def convolution_backward(dconv_prev, conv_in, filt, s):
    """
    Backpropagation of gradients through a convolutional layer.

    Parameters
    ----------
    dconv_prev : np.ndarray
        previous convolutional layer gradient
    conv_in : np.ndarray
        convolutional layer input
    filt : np.ndarray
        filter
    s : int
        stride

    Returns
    -------
    dout : np.ndarray
        loss gradient of the input to the convolutional operation
    dfilt : np.ndarray
        loss gradient of filter, used to update hte filter
    dbias : np.ndarray
        loss gradient of the bias
    """
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    # initialize derivatives
    dout = np.zeros(conv_in.shape)
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f, 1))
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y + f, curr_x:curr_x + f]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])

    return dout, dfilt, dbias


def nanargmax(arr):
    """
    return index of the largest non-nan value in the array. Output is an ordered pair tuple

    Parameters
    ----------
    arr : np.ndarray
        array to find largest non-nan value from

    Returns
    -------
    idxs : tuple of np.ndarray
        an ordered pair tuple containing index of the largest non-nan value
    """
    idx = np.nanargmax(arr)
    # converts flat array idx into a tuple of coordinate arrays
    idxs = np.unravel_index(idx, arr.shape)
    return idxs


def maxpool_backward(dpool, orig, f, s):
    """
    Backpropagation through a maxpooling layer. The gradients are passed through the
    indices of greatest value in the original maxpooling during the forward step.

    Parameters
    ----------
    dpool : np.ndarray
        max-pooling layer gradient
    orig : np.ndarray
        previous maxpooling layer
    f : int
        kernel size
    s : int
        stride

    Returns
    -------
    dout : np.ndarray
        loss gradient of the input to the convolutional operation
    """
    (n_c, orig_dim, _) = orig.shape

    # initialize loss gradient
    dout = np.zeros(orig.shape)

    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
                # pass gradients through
                dout[curr_c, curr_y + a, curr_x + b] = dpool[curr_c, out_y, out_x]

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return dout


# Step 3.5: Define the forward operations
def convolution(image, filt, bias, s=1):
    """
    Convolves the filter over every part of the image, adding the bias at each step.

    Parameters
    ----------
    image : np.ndarray
        images
    filt : np.ndarray
        filter
    bias : np.ndarray
        bias
    s : int
        stride, default=1

    Returns
    -------
    out : np.ndarray
        convolutional layer
    """
    (n_f, n_c_f, f, _) = filt.shape  # filter dimensions
    n_c, in_dim, _ = image.shape  # image dimensions

    out_dim = int((in_dim - f) / s) + 1  # calculate output dimensions

    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

    out = np.zeros((n_f, out_dim, out_dim))

    # convolve the filter over every part of the image, adding the bias at each step.
    for curr_f in range(n_f):
        # initialize current and output y
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            # initialize current and output x
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                # convolution
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f]) + \
                                            bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return out


def maxpool(image, f=2, s=2):
    """
    Downsample `image` using kernel size `f` and stride `s`. Slides maxpool window over each part of the
    image and assigns the max value at each step to the output.

    Parameters
    ----------
    image : np.ndarray
        image to downsample
    f : int
        kernel size, default=2
    s : int
        stride, default=2

    Returns
    -------
    downsampled : np.ndarray
        downsampled image
    """
    n_c, h_prev, w_prev = image.shape

    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    # initialize downsampled image values
    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                # assign max value to output
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f, curr_x:curr_x + f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled


def softmax(x):
    """
    performs softmax activation

    Parameters
    ----------
    x : np.ndarray
        array to perform softmax activation on

    Returns
    -------
    out / np.sum(out) : np.ndarray
        softmax activated x
    """
    out = np.exp(x)
    return out / np.sum(out)


def categorical_cross_entropy(probs, label):
    """
    Performs categorical cross-entropy

    Parameters
    ----------
    probs : np.ndarray
        array of predicted probabilities
    label : np.ndarray
        array of true labels

    Returns
    -------
    -np.sum(label * np.log(probs)) : float
        categorical cross entropy of probs and label
    """
    return -np.sum(label * np.log(probs))


# Step 4: Building the network
def conv(image, label, params, conv_s, pool_f, pool_s):
    """
    Combines the forward and backward operations to build the network. Takes the network's parameters and
    hyperparameters as inputs and spits out the gradients.

    Parameters
    ----------
    image : np.ndarray
        images to build network
    label : np.ndarray
        labels corresponding to images
    params : list
        network's parameters
    conv_s : int
        convolutional layer stride
    pool_f : int
        pooling layer kernel size
    pool_s : int
        pooling layer stride

    Returns
    -------
    grads : list
        gradients
    loss : float
        categorical cross-entropy loss
    """

    # set parameters
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    ################################################
    # Forward Operation #
    ################################################
    conv1 = convolution(image, f1, b1, conv_s)  # convolution operation
    conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity

    conv2 = convolution(conv1, f2, b2, conv_s)  # second convolution operation
    conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_f, pool_s)  # maxpooling operation

    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

    z = w3.dot(fc) + b3  # first dense layer
    z[z <= 0] = 0  # pass through ReLU non-linearity

    out = w4.dot(z) + b4  # second dense layer

    probs = softmax(out)  # predict class probabilities with the softmax activation function

    ################################################
    # Loss #
    ################################################

    loss = categorical_cross_entropy(probs, label)  # categorical cross-entropy loss

    ################################################
    # Backward Operation #
    ################################################
    dout = probs - label  # derivative of loss w.r.t. final dense layer output
    dw4 = dout.dot(z.T)  # loss gradient of final dense layer weights
    db4 = np.sum(dout, axis=1).reshape(b4.shape)  # loss gradient of final dense layer biases

    dz = w4.T.dot(dout)  # loss gradient of first dense layer outputs
    dz[z <= 0] = 0  # backpropagate through ReLU
    dw3 = dz.dot(fc.T)
    db3 = np.sum(dz, axis=1).reshape(b3.shape)

    dfc = w3.T.dot(dz)  # loss gradients of fully-connected layer (pooling layer)
    dpool = dfc.reshape(pooled.shape)  # reshape fully connected into dimensions of pooling layer

    # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    dconv2 = maxpool_backward(dpool, conv2, pool_f, pool_s)
    dconv2[conv2 <= 0] = 0  # backpropagate through ReLU

    # backpropagate previous gradient through second convolutional layer.
    dconv1, df2, db2 = convolution_backward(dconv2, conv1, f2, conv_s)
    dconv1[conv1 <= 0] = 0  # backpropagate through ReLU

    # backpropagate previous gradient through first convolutional layer.
    dimage, df1, db1 = convolution_backward(dconv1, image, f1, conv_s)

    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

    return grads, loss


# Step 5: Training the network
def adam_gd(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    """
    Update the parameters through Adam gradient descent. Forces the network's parameters to learn
    meaningful representations.

    Parameters
    ----------
    batch : np.ndarray
        batch of images to input
    num_classes : int
        number of classes in labels
    lr : float
        learning rate
    dim : int
        dimension
    n_c : int
        first dim of image shape
    beta1 : float
        weight to update momentum
    beta2 : float
        weight to update RMSProp
    params : list
        network parameters
    cost : list
        cost

    Returns
    -------
    params : list
        network parameters
    cost : list
        cost, summation of loss
    """
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    x = batch[:, 0:-1]  # get batch inputs
    x = x.reshape(len(batch), n_c, dim, dim)
    y = batch[:, -1]  # get batch labels

    cost_ = 0
    batch_size = len(batch)

    # initialize gradients and momentum, RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)

    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)

    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)

    for i in range(batch_size):
        # image for this batch
        c = x[i]
        # one-hot encoded label
        lbl = np.eye(num_classes)[int(y[i])].reshape(num_classes, 1)  # convert label to one-hot

        # collect gradients for training example
        grads, loss = conv(c, lbl, params, 1, 2, 2)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

        df1 += df1_
        db1 += db1_
        df2 += df2_
        db2 += db2_
        dw3 += dw3_
        db3 += db3_
        dw4 += dw4_
        db4 += db4_

        cost_ += loss

    # Parameter Update

    v1 = beta1 * v1 + (1 - beta1) * df1 / batch_size  # momentum update
    s1 = beta2 * s1 + (1 - beta2) * (df1 / batch_size) ** 2  # RMSProp update
    f1 -= lr * v1 / np.sqrt(s1 + 1e-7)  # combine momentum and RMSProp to perform update with Adam

    bv1 = beta1 * bv1 + (1 - beta1) * db1 / batch_size
    bs1 = beta2 * bs1 + (1 - beta2) * (db1 / batch_size) ** 2
    b1 -= lr * bv1 / np.sqrt(bs1 + 1e-7)

    v2 = beta1 * v2 + (1 - beta1) * df2 / batch_size
    s2 = beta2 * s2 + (1 - beta2) * (df2 / batch_size) ** 2
    f2 -= lr * v2 / np.sqrt(s2 + 1e-7)

    bv2 = beta1 * bv2 + (1 - beta1) * db2 / batch_size
    bs2 = beta2 * bs2 + (1 - beta2) * (db2 / batch_size) ** 2
    b2 -= lr * bv2 / np.sqrt(bs2 + 1e-7)

    v3 = beta1 * v3 + (1 - beta1) * dw3 / batch_size
    s3 = beta2 * s3 + (1 - beta2) * (dw3 / batch_size) ** 2
    w3 -= lr * v3 / np.sqrt(s3 + 1e-7)

    bv3 = beta1 * bv3 + (1 - beta1) * db3 / batch_size
    bs3 = beta2 * bs3 + (1 - beta2) * (db3 / batch_size) ** 2
    b3 -= lr * bv3 / np.sqrt(bs3 + 1e-7)

    v4 = beta1 * v4 + (1 - beta1) * dw4 / batch_size
    s4 = beta2 * s4 + (1 - beta2) * (dw4 / batch_size) ** 2
    w4 -= lr * v4 / np.sqrt(s4 + 1e-7)

    bv4 = beta1 * bv4 + (1 - beta1) * db4 / batch_size
    bs4 = beta2 * bs4 + (1 - beta2) * (db4 / batch_size) ** 2
    b4 -= lr * bv4 / np.sqrt(bs4 + 1e-7)

    cost_ = cost_ / batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    return params, cost


def train(num_classes=10, lr=0.01, beta1=0.95, beta2=0.99, img_dim=28, img_depth=1, f=5,
          num_filt1=8, num_filt2=8, batch_size=32, num_epochs=2, save_path='params.pkl'):
    """
    Trains the network. Gets training data, initializes parameters,

    Parameters
    ----------
    num_classes : int
        number of classes in labels
    lr : float
        learning rate
    beta1 : float
        weight to update momentum
    beta2 : float
        weight to update RMSprop
    img_dim : int
        image dimension
    img_depth : int
        image depth
    f : int
        kernel size
    num_filt1 : int
        number of filters
    num_filt2 : int
        number of filters
    batch_size : int
        size of batch to train on
    num_epochs : int
        number of training epochs
    save_path : str
        name of path to save network parameters

    Returns
    -------
    cost : list
        sum of loss from training
    """

    # Get training data
    m = 50000
    x = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m, 1)
    x -= int(np.mean(x))
    x /= int(np.std(x))
    train_data = np.hstack((x, y_dash))

    np.random.shuffle(train_data)

    # Initializing all the parameters
    f1, f2, w3, w4 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f), (128, 800), (10, 128)
    f1 = initialize_filter(f1)
    f2 = initialize_filter(f2)
    w3 = initialize_weight(w3)
    w4 = initialize_weight(w4)

    b1 = np.zeros((f1.shape[0], 1))
    b2 = np.zeros((f2.shape[0], 1))
    b3 = np.zeros((w3.shape[0], 1))
    b4 = np.zeros((w4.shape[0], 1))

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    cost = []

    print("LR:" + str(lr) + ", Batch Size:" + str(batch_size))

    for epoch in range(num_epochs):
        # shuffle training data
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for c, batch in enumerate(t):
            # perform AdamGD
            params, cost = adam_gd(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))

    # save parameters in local path
    with open(save_path, 'wb') as file:
        pickle.dump(params, file)

    return cost
