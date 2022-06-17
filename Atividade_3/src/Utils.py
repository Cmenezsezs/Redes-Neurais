import numpy as np

def relu(x, gradient=False):
    if gradient: return 0.0 if x <=0 else 1.0
    return max(0.0, x)

def sigmoid(x, gradient=False):
    if gradient: return np.exp(-x) / ((1 + np.exp(-x))) ** 2
    return 1 / (1 + np.exp(-x))

def tanh(x, gradient=False):
    if gradient: return 1-np.tanh(x)**2
    return np.tanh(x);

# Loss function
def mse(y_true, y_pred, gradient=False):
    if gradient: return 2*(y_pred-y_true)/1
    return np.mean(np.power(y_true-y_pred, 2))

def mean(vet): return sum(vet)/len(vet)