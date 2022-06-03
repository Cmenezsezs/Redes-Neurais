import numpy as np
from rbf_network import RBFNetwork
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def funcao_1(x):
    return (1 - x - 2 * (x ** 2)) * np.exp(- (x ** 2) / 2)

def funcao_2(x):
    return 2 * np.sin(x[:,0] / 2) * np.cos(x[:,1] / 2)

def datasets(dados, funcao):
    y = funcao(dados)
    return y.reshape((-1, 1))

def split_regr_targ(serie, num_in, num_out, num_lag):
    
    num_instancias = len(serie) - ((num_in - 1) * num_lag + num_out)
    
    target = np.zeros((num_instancias, num_out))
    regressoras = np.zeros((num_instancias, num_in))
    
    for targ in range(num_out):
        indice_in = ((num_in - 1) * num_lag) + targ + 1
        indice_out = indice_in + num_instancias
        target[:,targ] = serie[indice_in:indice_out]
    
    for regr in range(num_in):
        indice_in = ((num_in - 1) * num_lag) - regr * num_lag
        indice_out = indice_in + num_instancias
        regressoras[:,regr] = serie[indice_in:indice_out]
        
    matriz = np.concatenate([regressoras, target], axis=1)
    matriz = matriz[~np.isnan(matriz).any(axis=1)]
    
    regressoras = matriz[:,:-num_out]
    target = matriz[:,-num_out:]
    
    return regressoras, target


def dataset(N, seed=0):
    np.random.seed(seed)
    saida = [0] * 4
    for _ in range(N):
        aux = 1.8 * saida[-1] - 2 * saida[-2] + 1.2 * saida[-3] - 0.4 * saida[-4] + np.random.normal()
        saida.append(aux)
    
    return np.array(saida[4:])


# seed = 0
# np.random.seed(seed)
# x = np.linspace(0, 10, 100)
# X = []
# for i in x:
#     for j in x:
#         X.append([i, j])

# X = np.array(X)
# y = datasets(X, funcao_2)
seed = 0
N_dados = 1000
serie = dataset(N_dados)


num_treino = int(len(serie) * 0.8)
serie_treino = serie[:num_treino]
serie_teste = serie[num_treino:]

num_in = 4
num_out = 1
num_lag = 1

X_train, y_train = split_regr_targ(serie_treino, num_in, num_out, num_lag)
X_test, y_test = split_regr_targ(serie_teste, num_in, num_out, num_lag)


parametros = []
for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    modelo = RBFNetwork()
    modelo.fit(X_train, y_train, 100, lr, seed)
    previsoes = modelo.predict(X_test)
    parametros.append([lr, mean_squared_error(y_test, previsoes)])
    
parametros = np.array(parametros)
best_ = print(parametros[np.argmin(parametros[:, 1]),:])
plt.plot(y_test)
plt.plot(previsoes)
plt.show()