import numpy as np
from MLP import MLP
from Utils import *
from DataLoader import DataLoader
seed = 1
np.random.seed(seed)

n_features = 5
n_output = 1
dir_base = "/home/baltz/Documentos/Mestrado/2022-1/Redes Neurais/git_repo/Redes_Neurais_Mestrado_2022-1/Atividade_3/base_dados/"

dt = DataLoader(n_features, n_output, dir_base)
X_train, y_train, X_test, y_test = dt.get_dataset("OAKLAND1")

# X_train = X_train[:100]
# y_train = y_train[:100]
# X_test = X_test[:100]
# y_test = y_test[:100]

tam_base_train = len(X_train)
# tam_base_val = 50
tam_base_test = len(X_test)

epochs = 3
lr = 0.0001

# hiddens = [int(50/2), int(100/2), int(50/2)]
hiddens = [32, 16, 8]
act_fns = ["sigmoid", "sigmoid", "none", "relu"]
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
            print(f"[{i}] Epo: {epo}, Loss Train: {mean(vet_loss_train)}", end="\r")
    print(f"Epo: {epo}, Loss Train: {mean(vet_loss_train)}")

    # vet_loss_val = []
    # for i in range(tam_base_val):
    #     inputs, gt = X_val[i], y_val[i]
    #     pred = mlp.forward(inputs, training=False)
    #     vet_loss_val.append(mse(gt, pred, gradient=False))
    # print(f"Epo: {epo}, Loss Val:   {mean(vet_loss_val)}")

    vet_loss_test = []
    for i in range(tam_base_test):
        inputs, gt = X_test[i], y_test[i]
        pred = mlp.forward(inputs, training=False)
        vet_loss_test.append(mse(gt, pred, gradient=False))
        if i%100 == 0:
            print(f"[{i}] Epo: {epo}, Loss Test:  {mean(vet_loss_test)}", end="\r")
    print(f"Epo: {epo}, Loss Test:  {mean(vet_loss_test)}")
    print("------------------")


print(f"\nExemplos de predições da base de test:")
for i in range(10):
    print(f"input: {X_test[i]}, pred: {mlp.forward(X_test[i]):.3}, real: {y_test[i]:.3}")