import numpy as np
from layers_utils import *

# ==========================================
# 3. TRAIN THE NETWORK AND PRODUCE PREDICTIONS
# ==========================================

def l_layer_model(X, Y, val_x, val_y, layers_dims, learning_rate, num_iterations, batch_size):
    """
    Implements a L-layer neural network.
    """
    costs = []
    # משיכת המימד השני של X שהוא מספר הדוגמאות/תמונות
    m = X.shape[1] 
    
    # 1. INITIALIZE PARAMETERS
    parameters = initialize_parameters(layers_dims)
    
    # משתנים למעקב אחרי תנאי העצירה 
    best_val_acc = 0
    steps_without_improvement = 0

    # Training loop 
    for i in range(num_iterations):
        
        # --- Mini-batch selection ---
        permutation = np.random.permutation(m)
        X_batch = X[:, permutation[0:batch_size]]
        Y_batch = Y[:, permutation[0:batch_size]]
        
        # 2. FORWARD PROPAGATION
        AL, caches = l_model_forward(X_batch, parameters, use_batchnorm=False)
        
        # 3. COMPUTE COST
        cost = compute_cost(AL, Y_batch)
        
        # 4. BACKWARD PROPAGATION
        grads = l_model_backward(AL, Y_batch, caches)
        
        # 5. UPDATE PARAMETERS
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # בדיקה כל 100 איטרציות 
        if i % 100 == 0:
            costs.append(cost)
            # חישוב דיוק על הוולידציה לצורך תנאי העצירה 
            current_val_acc = predict(val_x, val_y, parameters)
            print(f"Iteration {i}: Cost {cost:.4f}, Validation Accuracy {current_val_acc:.4f}")
            
            # בדיקת תנאי עצירה: האם היה שיפור? 
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                steps_without_improvement = 0
            else:
                steps_without_improvement += 100
            
            if steps_without_improvement >= 100:
                print(f"Stopping training: No improvement for 100 steps.")
                break
                
    return parameters, costs

def predict(X, Y, parameters):
    """
    Calculates the accuracy of the trained network.
    """
    # 1. Forward propagation 
    AL, _ = l_model_forward(X, parameters, use_batchnorm=False)
    
    # 2. Get predictions and true labels 
    predictions = np.argmax(AL, axis=0)
    true_labels = np.argmax(Y, axis=0)
    
    # 3. Calculate accuracy 
    accuracy = np.mean(predictions == true_labels)
    
    return accuracy