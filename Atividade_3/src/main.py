import numpy as np
from MLP import MLP
from Utils import *
np.random.seed(1)

n_features_base = 2
tam_base_train = 500
tam_base_val = 50
tam_base_test = 100

X_train = np.array([np.random.rand(n_features_base) - 0.5 for _ in range(tam_base_train)])
y_train = np.array([np.mean(x) for x in X_train])

X_val = np.array([np.random.rand(n_features_base) - 0.5 for _ in range(tam_base_val)])
y_val = np.array([np.mean(x) for x in X_val])

X_test = np.array([np.random.rand(n_features_base) - 0.5 for _ in range(tam_base_test)])
y_test = np.array([np.mean(x) for x in X_test])

epochs = 10
lr = 0.001

num_inputs = len(X_train[0])
hiddens = [50, 100, 50]
num_outputs = 1
act_fns = ["none", "none", "none", "none"]
mlp = MLP(num_inputs, hiddens, num_outputs, act_fns)
mlp.mostrar_rede()

for epo in range(epochs):
    vet_loss_train = []
    for i in range(tam_base_train):
        inputs, gt = X_train[i], y_train[i]
        pred = mlp.forward(inputs, training=True)
        mlp.backpropagation([mse(gt, pred, gradient=True)], lr)
        vet_loss_train.append(mse(gt, pred, gradient=False))
    print(f"Epo: {epo}, Loss Train: {mean(vet_loss_train)}")

    vet_loss_val = []
    for i in range(tam_base_val):
        inputs, gt = X_val[i], y_val[i]
        pred = mlp.forward(inputs, training=False)
        vet_loss_val.append(mse(gt, pred, gradient=False))
    print(f"Epo: {epo}, Loss Val:   {mean(vet_loss_val)}")

    vet_loss_test = []
    for i in range(tam_base_test):
        inputs, gt = X_test[i], y_test[i]
        pred = mlp.forward(inputs, training=False)
        vet_loss_test.append(mse(gt, pred, gradient=False))
    print(f"Epo: {epo}, Loss Test:  {mean(vet_loss_test)}")
    print("------------------")
