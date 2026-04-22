"""
    Deep Learning Task 1
    Written By: Karin Fridkin and Rotem Trabelsi
    Description: building a simple neural network “from scratch”.
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.datasets import fetch_openml
from dnn_model import l_layer_model, predict

def load_and_preprocess_mnist():
    """ Load, Flatten (784, m), Normalize, One-hot encode Y """
    print("Loading MNIST data... (this may take a moment)")
    mnist_data = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist_data["data"], mnist_data["target"]
    
    # שימוש ב-float32 כדי לחסוך חצי מכמות הזיכרון (מונע ArrayMemoryError)
    X = (X.T / 255.).astype(np.float32)
    
    train_x_full = X[:, :60000]
    train_y_full = y[:60000]
    test_x = X[:, 60000:]
    test_y = y[60000:]
   
    # One-hot encoding חסכוני בזיכרון
    def one_hot(Y):
        Y = Y.astype(int) 
        oh = np.zeros((10, Y.size), dtype=np.float32)
        oh[Y, np.arange(Y.size)] = 1
        return oh
    
    test_y_oh = one_hot(test_y)
    
    # פיצול ל-Validation
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
    # 1. Load Data
    train_x, train_y, val_x, val_y, test_x, test_y_oh = load_and_preprocess_mnist()
    
    # 2. הגדרת הפרמטרים של הרשת [סעיף 4.ב]
    layers_dims = [784, 20, 7, 5, 10]
    learning_rate = 0.009 
    batch_size = 256  # Batch Size קטן יותר כדי שהמחשב לא ייחנק בסעיף 5
    num_iterations = 10000
    
    print("\nStarting Training (Section 5) - WITH Batchnorm...")
    
    # קריאה לפונקציית האימון - ודאי ששינית ל-True בתוך dnn_model.py
    parameters, costs = l_layer_model(
        train_x, 
        train_y, 
        val_x, 
        val_y, 
        layers_dims, 
        learning_rate, 
        num_iterations, 
        batch_size
    )
    
    # בדיקת דיוק סופי על Test
    print("\nTraining finished! Checking final accuracy on Test Set:")
    test_acc = predict(test_x, test_y_oh, parameters)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    train_acc = predict(train_x, train_y, parameters)
    val_acc = predict(val_x, val_y, parameters)
    test_acc = predict(test_x, test_y_oh, parameters)
    
    print(f"Final Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    run_experiments()