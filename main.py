"""
    Deep Learning Task 1
    Written By: Karin Fridkin and Rotem ----
    Description: building a simple neural network “from scratch”.
"""

import numpy as np
from dnn_model import l_layer_model, predict

def load_and_preprocess_mnist():
    """ Load, Flatten (784, m), Normalize, One-hot encode Y """
    pass

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