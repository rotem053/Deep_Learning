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

    for l in range(1,L):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l],layer_dims[l-1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    """ Returns Z, linear_cache """
    Z = (W @ A) + b
    linear_cache = { "A": A, "W": W, "b": b}

    return Z, linear_cache

def softmax(Z):
    """ Returns A, activation_cache """
    exp_Z = np.exp(Z)
    sum_exp_Z = np.sum(exp_Z, axis=0, keepdims=True)

    A = exp_Z / sum_exp_Z
    activation_cache = {"Z": Z}
    return A, activation_cache

def relu(Z):
    """ Returns A, activation_cache """
    A = np.maximum(0, Z)
    activation_cache = {"Z": Z}
    
    return A, activation_cache


def linear_activation_forward(A_prev, W, b, activation):
    """ Returns A, cache (combines linear & activation caches) """
    linear_z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "softmax":
        A,activation_cache = softmax(linear_z)
    elif activation == "relu":
        A,activation_cache = relu(linear_z)
    else:
        raise ValueError(f"Activation function '{activation}' is not supported! Use 'relu' or 'softmax'.")
    
    cache = (linear_cache, activation_cache)
    return A, cache


def l_model_forward(X, parameters, use_batchnorm):
    print(f"Debug: parameters keys are: {list(parameters.keys())}")
    """ The full forward loop. Returns AL, list_of_caches """
    L = len(parameters) // 2 # number of layers
    curr_A = X
    all_cache = []

    for l in range(1,L):
        if use_batchnorm:
            curr_A = apply_batchnorm(curr_A)
        curr_A,curr_cache = linear_activation_forward(curr_A,parameters[f'W{l}'],parameters[f'b{l}'],"relu")
        all_cache.append(curr_cache)
    
    AL,curr_cache = linear_activation_forward(curr_A,parameters[f'W{L}'],parameters[f'b{L}'],"softmax")
    all_cache.append(curr_cache)

    return AL, all_cache

def compute_cost(AL, Y):
    """ Categorical Cross-Entropy """
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y * np.log(AL + 1e-8))
    cost = np.squeeze(cost)
    
    return cost

def apply_batchnorm(A):
    mean = np.mean(A, axis=1, keepdims=True)
    var = np.var(A, axis=1, keepdims=True)
    
    epsilon = 1e-8
    A_norm = (A - mean) / np.sqrt(var + epsilon)
    
    return A_norm

# ==========================================
# 2. BACKWARD PROPAGATION 
# ==========================================
def linear_backward(dZ, cache):
    """ Returns dA_prev, dW, db """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    # ה-cache שהעברנו הוא טאפל שמכיל שני מילונים
    linear_cache_dict = cache[0]
    activation_cache_dict = cache[1]
    
    # שליפת המטריצות האמיתיות מתוך המילון - זה הצעד הקריטי!
    A_prev = linear_cache_dict["A"]
    W = linear_cache_dict["W"]
    b = linear_cache_dict["b"]
    
    # בניית הטאפל שהפונקציה linear_backward מצפה לקבל לפי ההוראות
    # ההוראות דורשות: A_prev, W, b [cite: 65]
    actual_linear_cache = (A_prev, W, b)
    
    if activation == "softmax":
        # פירוק ה-dA ששלחנו (AL ו-Y)
        AL, Y = dA 
        dZ = softmax_backward(AL, Y) # [cite: 74, 90]
        dA_prev, dW, db = linear_backward(dZ, actual_linear_cache) # [cite: 63]
        
    elif activation == "relu":
        # עבור ReLU, ה-activation_cache מכיל את Z [cite: 38, 87]
        Z = activation_cache_dict["Z"]
        dZ = relu_backward(dA, Z) # [cite: 83]
        dA_prev, dW, db = linear_backward(dZ, actual_linear_cache)
    
    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    """ Returns dZ """
    Z = activation_cache 

    dZ = np.array(dA, copy=True) 
    
    dZ[Z <= 0] = 0
    
    return dZ

def softmax_backward(AL, Y):
    """ Returns dZ """
    dZ = AL - Y
    
    return dZ

def l_model_backward(AL, Y, caches):
    """ The full backward loop. Returns grads dict """
    grads = {}
    L = len(caches) 
    
    # 1. שכבה אחרונה - SOFTMAX
    current_cache = caches[L-1]
    # שולחים את (AL, Y) יחד כדי לא לשנות את מספר הפרמטרים של הפונקציה הבאה
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward((AL, Y), current_cache, "softmax")
    
    # 2. שאר השכבות - RELU
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
        
    return grads
    

def update_parameters(parameters, grads, learning_rate):
    """ Gradient Descent update step """
    L = len(parameters) // 2

    for l in range(L):
        layer_num = l + 1
        
        parameters["W" + str(layer_num)] = parameters["W" + str(layer_num)] - learning_rate * grads["dW" + str(layer_num)]
        
        parameters["b" + str(layer_num)] = parameters["b" + str(layer_num)] - learning_rate * grads["db" + str(layer_num)]
        
    return parameters
    