#!/usr/bin/env python
# coding: utf-8



import os, math
import random
import sys
import datetime
## pip3 install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')




## constants
TRAIN_DIR = "./input/train/"
TEST_DIR = "./input/test/"
TRAIN_SIZE = 25000
TEST_SIZE = 12500
DEV_RATIO = 0.1
IMAGE_HEIGHT = IMAGE_WIDTH = 128

LEARNING_RATE = 0.0001
MINIBATCH_SIZE = 32
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3
OUTPUT_SIZE = 2




## data utility functions
def split_data(two_dims_datas, split_ratio=DEV_RATIO):
    left_count = int(two_dims_datas.shape[1] * split_ratio)
    left_datas = two_dims_datas[:, :left_count]
    right_datas = two_dims_datas[:, left_count:]
    print("input datas shape: {}, left datas shape:{},     right datas shape: {}".format(two_dims_datas.shape, left_datas.shape, right_datas.shape))
    return left_datas, right_datas




S = np.array([1, 2, 2])
S.astype(np.float)




def load_images(dirname="./input/train/", file_count=1000, shuffle=True,
                image_width=128, image_height=128, image_channels=3):
    """
    Arguments:
    dirname -- dirname which contains image files.Str.Default TRAIN_DIR
    file_count -- number of files want to load.
    shuffle -- if True, returns images and labels have shuffled.
    image_width -- Image widht.Int .Default 128.
    image_height -- Image height.Int .Default 128.
    image_channels -- number of Image channel.Int .Default 3.
    
    Returns:
    images -- numpy array containing images data.
              (n_x, m) = (image_width * image_height * image_channels, m)
    labels -- numpy array containing image labels.[if is_dog: 1].(1, m)
    """
    all_filenames = os.listdir(dirname)
    random.shuffle(all_filenames)
    filenames = all_filenames[:file_count]
    
    ## images
    images = np.zeros((file_count, IMAGE_HEIGHT*IMAGE_WIDTH*3))
    for i in range(file_count):
        imgnd_origin = cv2.imread(dirname+filenames[i])
        imgnd_resized = cv2.resize(imgnd_origin, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
        imgnd_flatten = imgnd_resized.reshape(1,-1)
        images[i] = imgnd_flatten
    
    ## labels from filenames
    labels_list = ["dog" in filename for filename in filenames]
    labels = np.array(labels_list, dtype='int8').reshape(file_count, 1) ## 1 if dog
    
    ## shuffle
    if shuffle:
        permutation = list(np.random.permutation(labels.shape[0]))
        labels = labels[permutation, :]
        images = images[permutation, :]

    ## normalization
    images = images/255.0
    images.astype(np.float)
    
    return images.T, labels.T

def random_mini_batches(X, Y, mini_batch_size = 64, shuffle=True):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    if shuffle:
        permutation = list(np.random.permutation(m))
        X = X[:, permutation]
        Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, mini_batch_size*k:mini_batch_size*(k+1)]
        mini_batch_Y = Y[:, mini_batch_size*k:mini_batch_size*(k+1)]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, -(m-mini_batch_size* math.floor(m/mini_batch_size)):]
        mini_batch_Y = Y[:, -(m-mini_batch_size* math.floor(m/mini_batch_size)):]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches




images, labels = load_images(file_count=100)
dev_images, train_images = split_data(images)
dev_labels, train_labels = split_data(labels)




def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters




## Linear Forward
def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.add(np.matmul(W, A_prev), b)
    
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (A_prev, W, b)
    
    return Z, cache




def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

## Linear-Activation Forward
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        #print(57, "Z, linear_cache = linear_forward(A_prev, W, b)", Z)
        A, activation_cache = sigmoid(Z)
        #print(58, "A, activation_cache = sigmoid(Z)", A)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache




## L-Layer Model
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward(A_prev, W, b, 'relu') (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_activation_forward(A_prev, W, b, 'sigmoid') (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    #print("isnan in X",np.argwhere(np.isnan(X)))
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu'
        )
        caches.append(cache)
        #print(28, "linear_activation_forward", l, A)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(
            A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid'
        )
    caches.append(cache)
    #print("isnan in AL",np.argwhere(np.isnan(AL)))
    #print("37 AL equal 1", np.argwhere(AL==1))
    assert(AL.shape == (1,X.shape[1]))
    AL = np.minimum(AL, 0.99999)
            
    return AL, caches




def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost




## Linear backward
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ##### A_prev.T.shape = (m, n_(L-1)), dZ.shape = (1, n_L)
    dW = (1.0/m) * np.matmul(dZ, A_prev.T)
    db = (1.0/m) * np.sum(dZ, axis=-1, keepdims=True)
    dA_prev = np.matmul(np.transpose(W), dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db




def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

## Linear-Activation backward
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db




np.minimum(dev_labels, 0.999)




def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    #print(25, np.divide(Y, AL))
    #print(26, np.divide(1 - Y, 1 - AL))
    #print("26","before L_model_backward", "AL isnan location", np.argwhere(np.isnan(AL)))
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    #print("28","after L_model_backward", "dAL isnan location", np.argwhere(np.isnan(dAL)), np.take(AL, np.argwhere(np.isnan(dAL))))
    #print(AL.shape, dAL.shape)
    #print("isnan dAL:", dAL[0,734], "isnan AL:", AL[0,734])
    
    #print("28", dAL)
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    
    return grads




def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        #print("before update", "W"+str(l+1), parameters["W" + str(l+1)], "dW" + str(l+1), grads["dW" + str(l+1)])
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        #print("after update", "W"+str(l+1), parameters["W" + str(l+1)], "dW" + str(l+1), grads["dW" + str(l+1)])
        
    return parameters




def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
    return v




# update_parameters_with_momentum

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * grads["db" + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
        
    return parameters, v




# GRADED FUNCTION: initialize_adam

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return v, s




# GRADED FUNCTION: update_parameters_with_adam

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        #print("t:", t)
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1 - np.power(beta1, 2))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1 - np.power(beta1, 2))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.power(grads["dW" + str(l+1)], 2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.power(grads["db" + str(l+1)], 2)
        #print("46", grads["dW" + str(l+1)])
        #print("47", s["dW" + str(l+1)])
        
        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        
        
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1 - np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1 - np.power(beta2, t))
        #print("54", s_corrected["dW" + str(l+1)])

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        #parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)]/(epsilon + np.power(s_corrected["dW" + str(l+1)], 0.5)))       ##
        #parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)]/(epsilon + np.power(s_corrected["db" + str(l+1)], 0.5)))    ###
        #print((np.sqrt(s_corrected["dW" + str(l+1)])))
        #print(learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon))
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)             
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
    return parameters, v, s




layers_dims = [49152, 5, 20, 1]




def L_layer_model(X, Y, layers_dims, parameters=None, learning_rate = 0.075,
                  num_epochs = 3000, optimizer="adam", mini_batch_size = 64,
                  beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    parameters -- An initial value of the weights, default is None.Parameters initialited by this value.
    learning_rate -- learning rate of the gradient descent update rule
    num_epochs -- number of epochs
    print_cost -- if True, it prints the cost every 100 steps
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                         # keep track of cost
    t = 0                            # initializing the counter required for Adam update
    
    # Parameters initialization.
    ## Parameters: python dictionary containing your parameters "W1", "b1", ..., "WL", "bL"
    if not parameters:
        parameters = initialize_parameters_deep(layers_dims)
    
    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # Loop (gradient descent)
    for i in range(num_epochs):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        '''
        AL -- last post-activation value
        caches -- list of caches containing:
                every cache of linear_activation_forward('relu') (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_activation_forward('sigmoid') (there is one, indexed L-1)
        '''
        minibatches = random_mini_batches(X, Y, mini_batch_size, shuffle=True)
        
        for minibatch in minibatches:
            #print("55 isnan in X",np.argwhere(np.isnan(X)))
            (minibatch_X, minibatch_Y) = minibatch
            AL, caches = L_model_forward(minibatch_X, parameters)
            #print("57 isnan in AL",np.argwhere(np.isnan(AL)))
            #print("58 caches",caches)
            # Compute cost.
            cost = compute_cost(AL, minibatch_Y)
    
            # Backward propagation.
            grads = L_model_backward(AL, minibatch_Y, caches)
            #print("64 grads",grads)
            #print("63", grads)
        
            # Update parameters.
            if optimizer == "gd":
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('epoches (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters, costs




trained_parameters, costs = L_layer_model(train_images, train_labels, layers_dims, parameters=None, learning_rate = 0.075,
                  num_epochs = 1001, optimizer="gd", mini_batch_size = 64,
                  beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, print_cost=True)




def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p




pred_train = predict(train_images, train_labels, trained_parameters)
pred_test = predict(dev_images, dev_labels, trained_parameters)




## First, let's take a look at some images the L-layer model labeled incorrectly. This will show a few mislabeled images. 
def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(IMAGE_WIDTH,IMAGE_HEIGHT,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))




classes = np.array([b'cat',b'dog'])




print_mislabeled_images(classes, dev_images, dev_labels, pred_test)




images, labels = load_images(file_count=100)
dev_images, train_images = split_data(images)
dev_labels, train_labels = split_data(labels)




trained_parameters, costs = L_layer_model(train_images, train_labels, layers_dims, parameters=None,
                  num_epochs = 1001, optimizer="gd", mini_batch_size = 64, learning_rate=0.0075,
                  beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, print_cost=True)




pred_train = predict(train_images, train_labels, trained_parameters)
pred_test = predict(dev_images, dev_labels, trained_parameters)
print_mislabeled_images(classes, dev_images, dev_labels, pred_test)




trained_parameters, costs = L_layer_model(train_images, train_labels, layers_dims, parameters=None, learning_rate = 0.0075,
                  num_epochs = 1001, optimizer="momentum", mini_batch_size = 64,
                  beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, print_cost=True)




pred_train = predict(train_images, train_labels, trained_parameters)
pred_test = predict(dev_images, dev_labels, trained_parameters)
print_mislabeled_images(classes, dev_images, dev_labels, pred_test)




trained_parameters, costs = L_layer_model(train_images, train_labels, layers_dims, parameters=None, learning_rate = 0.0075,
                  num_epochs = 1001, optimizer="adam", mini_batch_size = 64,
                  beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, print_cost=True)




pred_train = predict(train_images, train_labels, trained_parameters)
pred_test = predict(dev_images, dev_labels, trained_parameters)
print_mislabeled_images(classes, dev_images, dev_labels, pred_test)

