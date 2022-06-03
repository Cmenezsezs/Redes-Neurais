import tensorflow as tf
import pandas as pd
import numpy as np
import random
import math

import plotly.express as px
from matplotlib import pyplot as plt

from tensorflow.math import sigmoid
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

lr = 0.00001
epocas = 500
batch_size = 10
wds = [0.5, 0.25, 0.1, 0.05, 0.01, 0.025, 0.05, None]
seed = 1
tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)

# BASE DE DADOS
"""##########################################################################"""
# range, train samples, test samples
r, q_train, q_test = [0, 1], 200, 1000

# sigmoid(x1 + 2x2) + 0.5(x1x2)^2 + 0.5N
def fn_base(x1, x2):
    return sigmoid(x1 + 2*x2) + 0.5 * (x1*x2)**2 + 0.5 * np.random.normal()
X_train = []
for a1 in np.arange(r[0], r[1], (r[1] - r[0])/math.ceil(math.sqrt(q_train))):
    for a2 in np.arange(r[0], r[1], (r[1] - r[0])/round(math.sqrt(q_train))):
        X_train.append([a1, a2])
y_train = list(map(lambda x: fn_base(x[0], x[1]), X_train))

aux = list(zip(X_train, y_train)); random.shuffle(aux)

X_train = list(map(lambda x: x[0], aux[:int(-len(y_train)*0.7)]))
y_train = list(map(lambda x: x[1], aux[:int(-len(y_train)*0.7)]))

X_val = list(map(lambda x: x[0], aux[int(-len(y_train)*0.7):]))
y_val = list(map(lambda x: x[1], aux[int(-len(y_train)*0.7):]))

X_test = np.random.rand(q_test, 2); X_test *= (r[1] - r[0]); X_test += r[0]
y_test = list(map(lambda x: fn_base(x[0], x[1]), X_test))

X_train, y_train = np.array(X_train), np.array(y_train)
X_val,   y_val   = np.array(X_val),   np.array(y_val)
X_test,  y_test  = np.array(X_test),  np.array(y_test)

tam_train = len(y_train)
tam_val = len(y_val)
tam_test = len(y_test)

# REDE NEURAL
"""##########################################################################"""
class Rede(tf.keras.models.Model):
    def __init__(self, wd):
        super(Rede, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(
            units=32, activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(l2=wd))
        self.dense_2 = tf.keras.layers.Dense(
            units=64, activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(l2=wd))
        self.dense_3 = tf.keras.layers.Dense(
            units=1, activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(l2=wd))

    def __call__(self, inputs, training=False):
        out = self.dense_1(inputs)
        out = self.dense_2(out)
        out = self.dense_3(out)
        return out

# TREINAMENTO E VALIZAÇÃO DA REDE NEURAL
"""##########################################################################"""

otimizador = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.MeanSquaredError()

for wd in wds:
    rede = Rede(wd)
    loss_train_todas_epos = []
    loss_val_todas_epos = []
    loss_test_todas_epos = []
    for epo in range(epocas):

        # TREINAMENTO
        loss_train_epo_atual = 0.0
        for i in range(math.ceil(tam_train/batch_size)):
            # Coleta os dados por batch (de acordo com o tamanho do batch)
            inp = X_train[i*batch_size : min((i+1)*batch_size, tam_train)]
            gt = X_val[i*batch_size : min((i+1)*batch_size, tam_train)]

            # (FORWARD)
            with tf.GradientTape() as tape:
                pred = rede(inp)
                loss = loss_fn(pred, gt)

            # BACK PROPAGATION (BACKWARD)
            # Calcula o gradiente com base no loss, para atualizar os pesos
            gradientes = tape.gradient(loss, rede.trainable_weights)
            # Realiza a atualização dos pesos com base nos gradientes
            otimizador.apply_gradients(zip(gradientes, rede.trainable_weights))

            loss_train_epo_atual += loss
        loss_train_epo_atual /= math.ceil(tam_train/batch_size)
        print(f"Epoca {epo}/{epocas} | Loss Treino:   {loss:.5f} | [wd={wd}]")
        loss_train_todas_epos.append(loss_train_epo_atual)

        # VALIDAÇÃO
        loss_val_epo_atual = 0.0
        for i in range(math.ceil(tam_val/batch_size)):
            # Coleta os dados por batch (de acordo com o tamanho do batch)
            inp = X_val[i*batch_size : min((i+1)*batch_size, tam_val)]
            gt = X_val[i*batch_size : min((i+1)*batch_size, tam_val)]

            # Validação possui apenas forward
            pred = rede(inp)
            loss = loss_fn(pred, gt)

            loss_val_epo_atual += loss
        loss_val_epo_atual /= math.ceil(tam_val/batch_size)
        print(f"Epoca {epo}/{epocas} | Loss Validação: {loss:.5f} | [wd={wd}]")
        loss_val_todas_epos.append(loss_val_epo_atual)

        # TESTE
        loss_test_epo_atual = 0.0
        for i in range(math.ceil(tam_test/batch_size)):
            # Coleta os dados por batch (de acordo com o tamanho do batch)
            inp = X_test[i*batch_size : min((i+1)*batch_size, tam_test)]
            gt = X_test[i*batch_size : min((i+1)*batch_size, tam_test)]

            # Teste possui apenas forward
            pred = rede(inp)
            loss = loss_fn(pred, gt)

            loss_test_epo_atual += loss
        loss_test_epo_atual /= math.ceil(tam_test/batch_size)
        print(f"Epoca {epo}/{epocas} | Loss Teste: {loss:.5f} | [wd={wd}]")
        loss_test_todas_epos.append(loss_test_epo_atual)


    # GERA OS GRÁFICOS DE LOSS DA VALIDAÇÃO E TREINAMENTO PARA CADA wd
    ############################################################################
    df = pd.DataFrame(np.stack(
            (np.array(loss_train_todas_epos), np.array(loss_val_todas_epos), np.array(loss_test_todas_epos)),
            axis=-1),
        columns = ["Loss Treino", "Loss Validação", "Loss Test"])
    fig = px.line(
        df, y=df.columns,
        title=f"Losses da Execução do Modelo | wd={wd}")
    # fig.show()
    fig.write_html(f"./lr_{lr}-epoch_{epocas}-wd_{wd}.html")
    ############################################################################

    print(f"\nTREINAMENTO DO wd={wd} CONCLUÍDO.\n")
