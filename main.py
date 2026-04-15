"""
    Deep Learning Task 1
    Written By: Karin Fridkin and Rotem ----
    Description: building a simple neural network “from scratch”.
"""

import numpy as np
from tensorflow.keras.datasets import minst
from dnn_model import l_layer_model, predict

def load_and_preprocess_mnist():
    """ Load, Flatten (784, m), Normalize, One-hot encode Y """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    
    X = X.T / 255.
    
    train_x_full = X[:, :60000]
    train_y_full = y[:60000]
    test_x = X[:, 60000:]
    test_y = y[60000:]
    
    # One-hot encoding
    def one_hot(Y):
        oh = np.zeros((10, Y.size))
        oh[Y, np.arange(Y.size)] = 1
        return oh
    
    test_y_oh = one_hot(test_y)
    
    # פיצול 20% ל-Validation רנדומלי מתוך ה-Train
    m_total = train_x_full.shape[1]
    m_val = int(m_total * 0.2)
    perm = np.random.permutation(m_total)
    
    val_x = train_x_full[:, perm[:m_val]]
    val_y = one_hot(train_y_full[perm[:m_val]])
    train_x = train_x_full[:, perm[m_val:]]
    train_y = one_hot(train_y_full[perm[m_val:]])
    
    print("Data loaded successfully!")
    return train_x, train_y, val_x, val_y, test_x, test_y_oh

def run_experiments():
    # 1. Load Data & Split (Train, Val, Test)
    
    # 2. Section 3: Baseline (No Batchnorm, No L2)
    #    - Train until stopping criterion
    #    - Plot costs, Print Accuracies
    
    # 3. Section 4: Batchnorm Experiment
    #    - use_batchnorm = True
    #    - Compare speed and performance
    
    # 4. Section 5: L2 Regularization
    #    - L2_lambda > 0
    #    - Weight analysis
    pass

if __name__ == "__main__":
    run_experiments()