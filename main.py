"""
Deep Learning Task 1
Written By: Karin Fridkin and Rotem ----
Description: building a simple neural network “from scratch”.
"""
print("Starting the script...", flush=True)

import numpy as np
from sklearn.datasets import fetch_openml

def load_and_preprocess_mnist():
    """ Load, Flatten (784, m), Normalize, One-hot encode Y """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist["data"], mnist["target"]
    
    X = X.T / 255.
    
    # הוספנו astype(int) כדי לוודא שהתיוגים הם מספרים ולא מחרוזות
    train_x_full = X[:, :60000]
    train_y_full = y[:60000].astype(int) 
    test_x = X[:, 60000:]
    test_y = y[60000:].astype(int)
    
    # One-hot encoding
    def one_hot(Y):
        oh = np.zeros((10, Y.size))
        oh[Y, np.arange(Y.size)] = 1
        return oh
    
    test_y_oh = one_hot(test_y)
    
    # פיצול 20% ל-Validation רנדומלי מתוך ה-Train
    m_total = train_x_full.shape[1]  # תיקון ל-[1] 
    m_val = int(m_total * 0.2)
    np.random.seed(10)
    perm = np.random.permutation(m_total)
    
    val_x = train_x_full[:, perm[:m_val]]
    val_y = one_hot(train_y_full[perm[:m_val]])
    train_x = train_x_full[:, perm[m_val:]]
    train_y = one_hot(train_y_full[perm[m_val:]])
    
    print("Data loaded successfully!", flush=True)
    return train_x, train_y, val_x, val_y, test_x, test_y_oh


import numpy as np
GLOBAL_USE_BATCHNORM = False
GLOBAL_LAMBD = 0
GLOBAL_PARAMETERS = {} 
GLOBAL_X_VAL = None    
GLOBAL_Y_VAL = None
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
   
def linear_activation_forward(A_prev, W, b, activation):
    global GLOBAL_USE_BATCHNORM
    linear_z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "softmax":
        A_raw, activation_cache = softmax(linear_z)
    elif activation == "relu":
        A_raw, activation_cache = relu(linear_z)
        
    if GLOBAL_USE_BATCHNORM and activation != "softmax":
        A = apply_batchnorm(A_raw)
        activation_cache["A_pre_norm"] = A_raw
    else:
        A = A_raw
        
    cache = (linear_cache, activation_cache)
    return A, cache

def l_model_forward(X, parameters, use_batchnorm):
    global GLOBAL_USE_BATCHNORM
    GLOBAL_USE_BATCHNORM = use_batchnorm 
    L = len(parameters) // 2
    curr_A = X
    all_cache = []
    
    for l in range(1, L):
        curr_A, curr_cache = linear_activation_forward(curr_A, parameters[f'W{l}'], parameters[f'b{l}'], "relu")
        all_cache.append(curr_cache)
        
    AL, curr_cache = linear_activation_forward(curr_A, parameters[f'W{L}'], parameters[f'b{L}'], "softmax")
    all_cache.append(curr_cache)
    return AL, all_cache

def compute_cost(AL, Y):
    global GLOBAL_LAMBD, GLOBAL_PARAMETERS
    m = Y.shape[1]
    cross_entropy_cost = -np.mean(np.sum(Y * np.log(AL + 1e-8), axis=0))
    
    l2_cost = 0
    if GLOBAL_LAMBD > 0 and len(GLOBAL_PARAMETERS) > 0:
        L = len(GLOBAL_PARAMETERS) // 2
        sum_weights_squared = 0
        for l in range(1, L + 1):
            sum_weights_squared += np.sum(np.square(GLOBAL_PARAMETERS[f"W{l}"]))
        l2_cost = (GLOBAL_LAMBD / (2 * m)) * sum_weights_squared
        
    return cross_entropy_cost + l2_cost

def apply_batchnorm(A):
    mean = np.mean(A, axis=1, keepdims=True)
    var = np.var(A, axis=1, keepdims=True)
    
    epsilon = 1e-8
    A_norm = (A - mean) / np.sqrt(var + epsilon)
    
    return A_norm

def linear_backward(dZ, cache):
    global GLOBAL_LAMBD
    A_prev = cache['A']
    W = cache['W']
    b = cache['b']
    m = A_prev.shape[1]
    
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    if GLOBAL_LAMBD > 0:
        dW = dW + (GLOBAL_LAMBD / m) * W
        
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    global GLOBAL_USE_BATCHNORM
    linear_cache, activation_cache = cache
    
    if GLOBAL_USE_BATCHNORM and activation != "softmax":
        A_pre_norm = activation_cache['A_pre_norm']
        dA = batchnorm_backward(dA, A_pre_norm)
        
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
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

def l_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(Y, current_cache, "softmax")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
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

def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    global GLOBAL_USE_BATCHNORM, GLOBAL_LAMBD, GLOBAL_PARAMETERS, GLOBAL_X_VAL, GLOBAL_Y_VAL
    np.random.seed(42)
    costs = []
    parameters = initialize_parameters(layers_dims)
    GLOBAL_PARAMETERS = parameters # מעדכנים כדי שפונקציית העלות תכיר את המשקולות
    m = X.shape[1]
    best_val_cost    = np.inf
    no_improve_count = 0
    iteration        = 0
    
    for epoch in range(num_iterations):
        permutation  = np.random.permutation(m)
        X_shuffled   = X[:, permutation]
        Y_shuffled   = Y[:, permutation]
        for start in range(0, m, batch_size):
            end     = min(start + batch_size, m)
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]
            
            AL, caches  = l_model_forward(X_batch, parameters, GLOBAL_USE_BATCHNORM)
            cost        = compute_cost(AL, Y_batch)
            grads       = l_model_backward(AL, Y_batch, caches)
            parameters  = update_parameters(parameters, grads, learning_rate)
            GLOBAL_PARAMETERS = parameters # שומרים משקולות מעודכנות
            iteration += 1
            
            if iteration % 100 == 0:
                train_AL, _ = l_model_forward(X, parameters, GLOBAL_USE_BATCHNORM)
                train_cost  = compute_cost(train_AL, Y)
                val_AL,  _ = l_model_forward(GLOBAL_X_VAL, parameters, GLOBAL_USE_BATCHNORM)
                val_cost   = compute_cost(val_AL, GLOBAL_Y_VAL)
                costs.append(train_cost)
                print(f"Iteration {iteration} | Train Cost: {train_cost:.6f} | Val Cost: {val_cost:.6f}")
                
                if val_cost >= best_val_cost - 1e-5:
                    no_improve_count += 1
                else:
                    no_improve_count  = 0
                    best_val_cost     = val_cost
                    
                if no_improve_count >= 1:
                    print(f"\n⛔ Early stopping at iteration {iteration} (epoch {epoch})")
                    return parameters, costs
    return parameters, costs

def predict(X, Y, parameters):
    global GLOBAL_USE_BATCHNORM
    AL, _ = l_model_forward(X, parameters, use_batchnorm=GLOBAL_USE_BATCHNORM)
    predictions = np.argmax(AL, axis=0)
    labels      = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == labels)
    return accuracy

def run_experiments():
    # הוספנו את GLOBAL_PARAMETERS כדי שנוכל לאפס אותו כאן
    global GLOBAL_USE_BATCHNORM, GLOBAL_LAMBD, GLOBAL_X_VAL, GLOBAL_Y_VAL, GLOBAL_PARAMETERS
    
    # 1. טעינת הנתונים
    print("Loading data...", flush=True)
    train_x, train_y, val_x, val_y, test_x, test_y_oh = load_and_preprocess_mnist()
    
    # שמירת סט התיקוף (Validation) במשתנים הגלובליים
    GLOBAL_X_VAL = val_x
    GLOBAL_Y_VAL = val_y
    
    # הגדרת ממדי הרשת והיפר-פרמטרים
    layers_dims = list((784, 20, 7, 5, 10))
    learning_rate = 0.009
    num_iterations = 3000
    batch_size = 512
    
    # ==========================================
    # ניסוי 1: מודל בסיס
    # ==========================================
    GLOBAL_USE_BATCHNORM = False
    GLOBAL_LAMBD = 0
    GLOBAL_PARAMETERS = {} # <--- איפוס מוחלט למניעת זליגה!
    
    print("\n--- Starting Training (No Batchnorm, No L2) ---", flush=True)
    parameters, costs = l_layer_model(
        train_x, train_y, layers_dims, learning_rate, num_iterations, batch_size
    )
    
    print("\n--- Calculating Accuracies (No Batchnorm) ---", flush=True)
    train_acc = predict(train_x, train_y, parameters)
    val_acc = predict(val_x, val_y, parameters)
    test_acc = predict(test_x, test_y_oh, parameters)
    
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # ==========================================
    # ניסוי 2: מודל עם Batchnorm 
    # ==========================================
    GLOBAL_USE_BATCHNORM = True
    GLOBAL_LAMBD = 0
    GLOBAL_PARAMETERS = {} # <--- איפוס מוחלט למניעת זליגה!
    
    print("\n--- Starting Training (WITH Batchnorm, No L2) ---", flush=True)
    parameters_bn, costs_bn = l_layer_model(
        train_x, train_y, layers_dims, learning_rate, num_iterations, batch_size
    )
    
    print("\n--- Calculating Accuracies (WITH Batchnorm) ---", flush=True)
    train_acc_bn = predict(train_x, train_y, parameters_bn)
    val_acc_bn = predict(val_x, val_y, parameters_bn)
    test_acc_bn = predict(test_x, test_y_oh, parameters_bn)
    
    print(f"Train Accuracy: {train_acc_bn * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc_bn * 100:.2f}%")
    print(f"Test Accuracy: {test_acc_bn * 100:.2f}%")

    # ==========================================
    # ניסוי 3: מודל עם L2 Regularization 
    # ==========================================
    GLOBAL_USE_BATCHNORM = False
    GLOBAL_LAMBD = 0.1 
    GLOBAL_PARAMETERS = {} # <--- איפוס מוחלט למניעת זליגה!
    
    print("\n--- Starting Training (No Batchnorm, WITH L2) ---", flush=True)
    parameters_l2, costs_l2 = l_layer_model(
        train_x, train_y, layers_dims, learning_rate, num_iterations, batch_size
    )
    
    print("\n--- Calculating Accuracies (WITH L2) ---", flush=True)
    train_acc_l2 = predict(train_x, train_y, parameters_l2)
    val_acc_l2 = predict(val_x, val_y, parameters_l2)
    test_acc_l2 = predict(test_x, test_y_oh, parameters_l2)
    
    print(f"Train Accuracy: {train_acc_l2 * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc_l2 * 100:.2f}%")
    print(f"Test Accuracy: {test_acc_l2 * 100:.2f}%")

    # ==========================================
    # השוואת משקולות 
    # ==========================================
    def calculate_weights_norm(params):
        L = len(params) // 2
        total_norm = 0
        for l in range(1, L + 1):
            total_norm += np.sum(np.square(params[f"W{l}"]))
        return total_norm

    norm_base = calculate_weights_norm(parameters) 
    norm_l2 = calculate_weights_norm(parameters_l2) 

    print(f"\n--- Weights Comparison (Sum of Squared Weights) ---")
    print(f"Weights Norm without L2: {norm_base:.4f}")
    print(f"Weights Norm with L2 (lambd={GLOBAL_LAMBD}): {norm_l2:.4f}")

if __name__ == "__main__":
    run_experiments()
