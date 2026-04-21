import numpy as np
from dnn_model import *
# ==========================================
# TESTS
# ==========================================

def test_initialize_parameters():
    layer_dims = [784, 20, 7, 5, 10]
    parameters = initialize_parameters(layer_dims)
    
    for l in range(1, len(layer_dims)):
        assert parameters[f"W{l}"].shape == (layer_dims[l], layer_dims[l-1]), f"W{l} shape is wrong"
        assert parameters[f"b{l}"].shape == (layer_dims[l], 1),              f"b{l} shape is wrong"
        assert np.all(parameters[f"b{l}"] == 0),                             f"b{l} should be zeros"
    
    print("✅ test_initialize_parameters passed")


def test_linear_forward():
    A = np.array([[1, 2], [3, 4]])
    W = np.array([[1, 0], [0, 1]])
    b = np.array([[1], [0]])
    
    Z, cache = linear_forward(A, W, b)
    
    expected_Z = np.array([[2, 3], [3, 4]])
    assert Z.shape == A.shape,             "Z shape is wrong"
    assert np.allclose(Z, expected_Z),     "Z values are wrong"
    assert np.array_equal(cache["A"], A),  "cache A is wrong"
    assert np.array_equal(cache["W"], W),  "cache W is wrong"
    assert np.array_equal(cache["b"], b),  "cache b is wrong"
    
    print("✅ test_linear_forward passed")


def test_softmax():
    Z = np.array([[1.0], [2.0], [3.0]])
    A, cache = softmax(Z)
    
    assert A.shape == Z.shape,          "A shape is wrong"
    assert np.allclose(np.sum(A), 1.0), "softmax should sum to 1"
    assert np.all(A >= 0),              "softmax values should be non-negative"
    assert np.array_equal(cache["Z"], Z), "cache Z is wrong"
    
    print("✅ test_softmax passed")


def test_relu():
    Z = np.array([[-1, 0, 1], [2, -3, 4]])
    A, cache = relu(Z)
    
    expected_A = np.array([[0, 0, 1], [2, 0, 4]])
    assert np.array_equal(A, expected_A),   "ReLU values are wrong"
    assert np.array_equal(cache["Z"], Z),   "cache Z is wrong"
    
    print("✅ test_relu passed")


def test_compute_cost():
    AL = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y  = np.array([[1.0, 0.0], [0.0, 1.0]])
    cost = compute_cost(AL, Y)
    
    assert abs(cost) < 1e-5, "cost should be near 0 for perfect predictions"
    
    print("✅ test_compute_cost passed")


def test_apply_batchnorm():
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    NA = apply_batchnorm(A)
    
    assert NA.shape == A.shape,                             "shape should be unchanged"
    assert np.allclose(np.mean(NA, axis=1), 0, atol=1e-5), "mean should be ~0"
    assert np.allclose(np.var(NA, axis=1),  1, atol=1e-5), "variance should be ~1"
    
    print("✅ test_apply_batchnorm passed")


def test_linear_backward():
    np.random.seed(1)
    A_prev = np.random.randn(3, 4)
    W      = np.random.randn(2, 3)
    b      = np.random.randn(2, 1)
    dZ     = np.random.randn(2, 4)
    cache = {"A": A_prev, "W": W, "b": b}
    
    dA_prev, dW, db = linear_backward(dZ, cache)
    
    assert dA_prev.shape == A_prev.shape, "dA_prev shape is wrong"
    assert dW.shape == W.shape,           "dW shape is wrong"
    assert db.shape == b.shape,           "db shape is wrong"
    
    print("✅ test_linear_backward passed")


def test_l_model_forward_backward():
    np.random.seed(42)
    X          = np.random.randn(784, 10)
    Y          = np.zeros((10, 10))
    Y[np.random.randint(0, 10, 10), np.arange(10)] = 1  # one-hot
    layer_dims = [784, 20, 7, 5, 10]
    parameters = initialize_parameters(layer_dims)
    
    AL, caches = l_model_forward(X, parameters, use_batchnorm=False)
    print(type(caches[0]))
    assert AL.shape == (10, 10),               "AL shape is wrong"
    assert np.allclose(np.sum(AL, axis=0), 1), "AL columns should sum to 1"
    
    grads = l_model_backward(AL, Y, caches)
    for l in range(1, len(layer_dims)):
        assert f"dW{l}" in grads, f"dW{l} missing from grads"
        assert f"db{l}" in grads, f"db{l} missing from grads"
    
    print("✅ test_l_model_forward_backward passed")


def test_update_parameters():
    np.random.seed(1)
    layer_dims = [3, 2]
    parameters = initialize_parameters(layer_dims)
    grads = {
        "dW1": np.random.randn(2, 3),
        "db1": np.random.randn(2, 1)
    }
    learning_rate = 0.01
    W1_before = parameters["W1"].copy()
    
    parameters = update_parameters(parameters, grads, learning_rate)
    
    expected_W1 = W1_before - learning_rate * grads["dW1"]
    assert np.allclose(parameters["W1"], expected_W1), "W1 update is wrong"
    
    print("✅ test_update_parameters passed")


def test_predict():
    np.random.seed(42)
    X          = np.random.randn(784, 20)
    Y          = np.zeros((10, 20))
    Y[np.random.randint(0, 10, 20), np.arange(20)] = 1
    layer_dims = [784, 20, 7, 5, 10]
    parameters = initialize_parameters(layer_dims)
    
    accuracy = predict(X, Y, parameters)
    print(accuracy)
    assert 0 <= accuracy <= 1, "accuracy should be between 0 and 1"
    
    print(f"✅ test_predict passed (accuracy={accuracy:.2f})")


# ==========================================
# RUN ALL TESTS
# ==========================================
if __name__ == "__main__":
    test_initialize_parameters()
    test_linear_forward()
    test_softmax()
    test_relu()
    test_compute_cost()
    test_apply_batchnorm()
    test_linear_backward()
    test_l_model_forward_backward()
    test_update_parameters()
    test_predict()
    print("\n🎉 All tests passed!")