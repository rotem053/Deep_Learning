import numpy as np

# ==========================================
# 1. FORWARD PROPAGATION 
# ==========================================
def initialize_parameters(layer_dims):
    """ Returns parameters dict {W1, b1, ... WL, bL} 
        The key of the dict will be a f-string W{l} or b{l}
        b will be initialized to zero"""
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])  # He init
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    """ Returns Z, linear_cache """
    Z = (W @ A) + b
    linear_cache = { "A": A, "W": W, "b": b}

    return Z, linear_cache

def softmax(Z):
    """ Returns A, activation_cache """
    shifted_Z = Z - np.max(Z, axis=0, keepdims=True)
    
    exp_Z = np.exp(shifted_Z)
    sum_exp_Z = np.sum(exp_Z, axis=0, keepdims=True)

    A = exp_Z / sum_exp_Z
    activation_cache = {"Z": Z}
    return A, activation_cache

def relu(Z):
    """ Returns A, activation_cache """
    A = np.maximum(0, Z)
    activation_cache = {"Z": Z}
    
    return A, activation_cache


def linear_activation_forward(A_prev, W, b, activation, use_batchnorm = False):
    """ Returns A, cache (combines linear & activation caches) """
    linear_z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "softmax":
        A,activation_cache = softmax(linear_z)
    elif activation == "relu":
        A,activation_cache = relu(linear_z)
        if use_batchnorm:
            A = apply_batchnorm(A)
    else:
        raise ValueError(f"Activation function '{activation}' is not supported! Use 'relu' or 'softmax'.")
    
    cache = (linear_cache, activation_cache)

    return A, cache


def l_model_forward(X, parameters, use_batchnorm):
    L = len(parameters) // 2
    curr_A = X
    all_cache = []  

    for l in range(1, L):
        curr_A, curr_cache = linear_activation_forward(curr_A, parameters[f'W{l}'], parameters[f'b{l}'], "relu",use_batchnorm) 
        all_cache.append(curr_cache)  

    AL, curr_cache = linear_activation_forward(curr_A, parameters[f'W{L}'], parameters[f'b{L}'], "softmax")
    all_cache.append(curr_cache)  

    return AL, all_cache

# #compute_cost as written before l2 regulariztion
# def compute_cost(AL, Y):
#     """ Categorical Cross-Entropy """
#     m = Y.shape[1]
#     cost = (-(1/m)) * np.sum(Y * np.log(AL + 1e-8))
#     cost = np.squeeze(cost)
    
#     return cost

def compute_cost(AL, Y, parameters = None, lambd=0):
    """
    AL -- vector of predictions
    Y -- true labels
    parameters -- dictionary containing W1, b1...
    lambd -- regularization hyperparameter (scalar)
    """
    m = Y.shape[1]
    
    cross_entropy_cost = -np.mean(np.sum(Y * np.log(AL + 1e-8), axis=0))
    
    # 2. L2 Regularization 
    l2_cost = 0
    if lambd > 0:
        L = len(parameters) // 2
        sum_weights_squared = 0
        for l in range(1, L + 1):
            sum_weights_squared += np.sum(np.square(parameters[f"W{l}"]))
        l2_cost = (lambd / (2 * m)) * sum_weights_squared
        
    return cross_entropy_cost + l2_cost

def apply_batchnorm(A):
    mean = np.mean(A, axis=1, keepdims=True)
    var = np.var(A, axis=1, keepdims=True)
    
    epsilon = 1e-8
    A_norm = (A - mean) / np.sqrt(var + epsilon)
    
    return A_norm

# ==========================================
# 2. BACKWARD PROPAGATION 
# ==========================================
def linear_backward(dZ, cache, lambd=0):
    """ Returns dA_prev, dW, db """
    A_prev = cache['A']
    W = cache['W']
    b = cache['b'] 
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    if lambd > 0:
        dW = dW + (lambd / m) * W
    
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation,use_batchnorm = False,lambd=0):
    """ Returns dA_prev, dW, db """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)

    if use_batchnorm:
        Z_original = activation_cache['Z']
        dZ = batchnorm_backward(dZ, Z_original)

    dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)
    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    """ Returns dZ """
    Z = activation_cache["Z"] 

    dZ = np.array(dA, copy=True) 
    
    dZ[Z <= 0] = 0
    
    return dZ

def softmax_backward(dA, activation_cache):
    """
    Returns dZ
    """
    Z = activation_cache["Z"]
    AL, _ = softmax(Z)  
    dZ = AL - dA      
    return dZ

def l_model_backward(AL, Y, caches, use_batchnorm = False,lambd=0):
    """ The full backward loop. Returns grads dict """
    grads = {}
    L = len(caches) 
    assert (AL.shape == Y.shape), f"Error: AL shape {AL.shape} is not equal to Y shape {Y.shape}"

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(Y, current_cache, "softmax",lambd)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu",use_batchnorm,lambd)
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads

    

def update_parameters(parameters, grads, learning_rate):
    """ Gradient Descent update step """
    L = len(parameters) // 2

    for l in range(L):
        layer_num = l + 1
        
        parameters["W" + str(layer_num)] = parameters["W" + str(layer_num)] - learning_rate * grads["dW" + str(layer_num)]
        
        parameters["b" + str(layer_num)] = parameters["b" + str(layer_num)] - learning_rate * grads["db" + str(layer_num)]
        
    return parameters
    
def batchnorm_backward(dZ_norm, Z_original):
    m = dZ_norm.shape[1]
    epsilon = 1e-8
    
    mu = np.mean(Z_original, axis=1, keepdims=True)
    var = np.var(Z_original, axis=1, keepdims=True)
    
    std_inv = 1.0 / np.sqrt(var + epsilon)
    Z_norm = (Z_original - mu) * std_inv
    
    dZ = (1./m) * std_inv * (
        m * dZ_norm - 
        np.sum(dZ_norm, axis=1, keepdims=True) - 
        Z_norm * np.sum(dZ_norm * Z_norm, axis=1, keepdims=True)
    )
    
    return dZ