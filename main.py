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
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist["data"], mnist["target"]
    
    X = X.T / 255.
    
    train_x_full = X[:, :60000]
    train_y_full = y[:60000]
    test_x = X[:, 60000:]
    test_y = y[60000:]
   
    # One-hot encoding
    def one_hot(Y):
        Y = Y.astype(int) 
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
    # 1. Load Data
    train_x, train_y, val_x, val_y, test_x, test_y_oh = load_and_preprocess_mnist()
    
    # 2. הגדרת הפרמטרים של הרשת [6]
    # גודל קלט 784, ושכבות של 20, 7, 5, 10
    layers_dims = [784, 20, 7, 5, 10]
    learning_rate = 0.009 
    batch_size = 128 
    num_iterations = 3000 # נריץ עד מקסימום 3000 סיבובים, או עד שייעצר לבד בזכות הלוגיקה שלך
    
    print("\nStarting Training (Section 3) - No Batchnorm...")
    
    # קריאה לפונקציית האימון שכתבת!
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
    
    # בדיקת דיוק סופי על Test [11]
    print("\nTraining finished! Checking final accuracy on Test Set:")
    test_acc = predict(test_x, test_y_oh, parameters)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    run_experiments()