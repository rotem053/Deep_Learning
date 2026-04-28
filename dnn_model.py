from layers_utils import *

# ==========================================
# 3. TRAIN THE NETWORK AND PRODUCE PREDICTIONS
# ==========================================
def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False, X_val=None, Y_val=None, lambd=0):
    np.random.seed(42)
    costs = []
    parameters = initialize_parameters(layers_dims)
    m = X.shape[1]
    
    best_val_cost    = np.inf
    training_steps        = 0
    
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
            cost        = compute_cost(AL, Y_batch,lambd)
            grads       = l_model_backward(AL, Y_batch, caches,use_batchnorm,lambd)
            parameters  = update_parameters(parameters, grads, learning_rate)
            
            training_steps += 1
            
            # Record + check stopping criterion every 100 training steps
            if training_steps % 100 == 0:
                train_AL, _ = l_model_forward(X, parameters, use_batchnorm)
                train_cost  = compute_cost(train_AL, Y)
                val_AL,  _ = l_model_forward(X_val, parameters, use_batchnorm)
                val_cost   = compute_cost(val_AL, Y_val,lambd)
                costs.append(train_cost)
                print(f"training steps {training_steps} | Train Cost: {train_cost:.6f} | Val Cost: {val_cost:.6f}")

                # Stopping criterion
                if val_cost >= best_val_cost - 1e-5:
                    print(f"\n⛔ Early stopping at training steps {training_steps} (epoch {epoch})")
                    return parameters, costs
                else:
                    best_val_cost     = val_cost
                
    
    return parameters, costs


def predict(X, Y, parameters, use_batchnorm=False):
    """
    Returns accuracy (float between 0 and 1)
    """
    AL, _ = l_model_forward(X, parameters, use_batchnorm)

    predictions = np.argmax(AL, axis=0) 
    labels      = np.argmax(Y,  axis=0)  

    accuracy = np.mean(predictions == labels)

    return accuracy

