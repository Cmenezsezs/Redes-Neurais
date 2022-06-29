import os
import numpy as np
from MLP import MLP
from Utils import *
from DataLoader import DataLoader
from sklearn.preprocessing import StandardScaler

seed = 10
np.random.seed(seed)

n_features = 5
n_output = 1
dir_base = os.path.join(os.getcwd(), 'base_dados')

dt = DataLoader(n_features, n_output, dir_base, seed=seed)
X_train, y_train, X_val, y_val, X_test, y_test = dt.get_dataset(
        "OAKLAND1", size_train=0.8, size_val=0.5, seed=seed)

# Normalização da base de dados N~(média=0, desv_pad=1)
escala_regr = StandardScaler() 
X_train = escala_regr.fit_transform(X_train)
X_val = escala_regr.transform(X_val)
X_test = escala_regr.transform(X_test)

escala_targ = StandardScaler() 
y_train = np.array([aux[0] for aux in escala_targ.fit_transform(y_train)])
y_val = np.array([aux[0] for aux in escala_targ.transform(y_val)])
y_test = np.array([aux[0] for aux in escala_targ.transform(y_test)])

tam_base_train = y_train.shape[0]
tam_base_val = y_val.shape[0]
tam_base_test = y_test.shape[0]

epochs = 5
lr = 0.003

hiddens = [16*8, 8*8, 4*8]
act_fns = ["none", "none", "none", "none"]
mlp = MLP(n_features, hiddens, n_output, act_fns)
mlp.mostrar_rede()

for epo in range(epochs):
    vet_loss_train = []
    for i in range(tam_base_train):
        inputs, gt = X_train[i], y_train[i]
        pred = mlp.forward(inputs, training=True)
        mlp.backpropagation([mse(gt, pred, gradient=True)], lr)
        vet_loss_train.append(mse(gt, pred, gradient=False))
        if i%100 == 0:
            print(f"[{i}] Epo: {epo+1}, Loss Train: {mean(vet_loss_train):.6f}", end="\r")
    print(f"Epo: {epo+1}, Loss Train: {mean(vet_loss_train):.6f}, lr: {lr:.5f}")

    vet_loss_val = []
    for i in range(tam_base_val):
        inputs, gt = X_val[i], y_val[i]
        pred = mlp.forward(inputs, training=False)
        vet_loss_val.append(mse(gt, pred, gradient=False))
        if i%100 == 0:
            print(f"[{i}] Epo: {epo+1}, Loss Test:  {mean(vet_loss_val):.6f}", end="\r")
    print(f"Epo: {epo+1}, Loss Val:   {mean(vet_loss_val):.6f}, lr: {lr:.5f}")
    print("------------------")

    lr *= 0.7

vet_loss_test = []
for i in range(tam_base_test):
    inputs, gt = X_test[i], y_test[i]
    pred = mlp.forward(inputs, training=False)
    vet_loss_test.append(mse(gt, pred, gradient=False))
    if i%100 == 0:
        print(f"[{i}] Epo: {epochs}, Loss Test:  {mean(vet_loss_test)}", end="\r")
print(f"Epo: {epochs}, Loss Test:  {mean(vet_loss_test)}")
print("------------------")

print(f"\nExemplos de predições da base de test:")
for i in range(10):
    print(f"input: {X_test[i]}, pred: {mlp.forward(X_test[i], training=False):.10}, real: {y_test[i]:.3}")
