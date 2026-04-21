from layers_utils import *

# ==========================================
# 3. TRAIN THE NETWORK AND PRODUCE PREDICTIONS
# ==========================================
def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False, X_val=None, Y_val=None):
    np.random.seed(42)
    costs = []
    parameters = initialize_parameters(layers_dims)
    m = X.shape[1]
    
    best_val_cost    = np.inf
    no_improve_count = 0
    iteration        = 0
    
    for epoch in range(num_iterations):
        # Shuffle
        permutation  = np.random.permutation(m)
        X_shuffled   = X[:, permutation]
        Y_shuffled   = Y[:, permutation]
        
        # Mini-batch loop
        for start in range(0, m, batch_size):
            end     = min(start + batch_size, m)
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]
            
            # Forward -> Cost -> Backward -> Update
            AL, caches  = l_model_forward(X_batch, parameters, use_batchnorm)
            cost        = compute_cost(AL, Y_batch)
            grads       = l_model_backward(AL, Y_batch, caches)
            parameters  = update_parameters(parameters, grads, learning_rate)
            
            iteration += 1
            
            # Record + check stopping criterion every 100 iterations
            if iteration % 100 == 0:
                val_AL,  _ = l_model_forward(X_val, parameters, use_batchnorm)
                val_cost   = compute_cost(val_AL, Y_val)
                costs.append(cost)
                print(f"Iteration {iteration} | Train Cost: {cost:.6f} | Val Cost: {val_cost:.6f}")
                
                # Stopping criterion
                if best_val_cost - val_cost < 1e-4:
                    no_improve_count += 1
                else:
                    no_improve_count  = 0
                    best_val_cost     = val_cost
                
                if no_improve_count >= 100:
                    print(f"\n⛔ Early stopping at iteration {iteration} (epoch {epoch})")
                    return parameters, costs
    
    return parameters, costs
# def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size = 64):
#     """ 
#     The main training loop function
#     X - the training data 
#     Y - the true labels of the training data
#     """
#     use_batchnorm = False
#     np.random.seed(42)
#     costs = []
    
#     parameters = initialize_parameters(layers_dims)
#     m = X.shape[1]
    
#     for i in range(num_iterations):
        
#         # Shuffle
#         permutation = np.random.permutation(m)
#         X_shuffled = X[:, permutation]
#         Y_shuffled = Y[:, permutation]
        
#         # Mini-batch loop
#         for start in range(0, m, batch_size):
#             end = min(start + batch_size, m)
#             X_batch = X_shuffled[:, start:end]
#             Y_batch = Y_shuffled[:, start:end]
            
#             # 1. Forward
#             AL, caches = l_model_forward(X_batch, parameters, use_batchnorm)
            
#             # 2. Cost
#             cost = compute_cost(AL, Y_batch)
            
#             # 3. Backward
#             grads = l_model_backward(AL, Y_batch, caches)
            
#             # 4. Update
#             parameters = update_parameters(parameters, grads, learning_rate)
        
#         # Record cost every 100 iterations
#         if i % 100 == 0:
#             print(f"Cost after iteration {i}: {cost:.6f}")
#             costs.append(cost)
    
#     return parameters, costs


def predict(X, Y, parameters):
    """
    Returns accuracy (float between 0 and 1)
    """
    # Forward pass (without batchnorm)
    AL, _ = l_model_forward(X, parameters, use_batchnorm=False)

    # Predicted class = index of highest probability for each example
    predictions = np.argmax(AL, axis=0) 
    labels      = np.argmax(Y,  axis=0)  

    accuracy = np.mean(predictions == labels)

    return accuracy