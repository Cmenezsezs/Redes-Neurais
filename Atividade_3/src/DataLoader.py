import os
import pandas as pd
import numpy as np

from math import floor
from sklearn.preprocessing import StandardScaler

class DataLoader():

    def __init__(self, num_inputs, num_outputs, dir_dataset, seed=None):
        self.dir_dataset = dir_dataset
        # Definicao do número de variáveis regressoras:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.seed = seed

    # Separacao das sequencias de treino e teste:
    def __split_regr_targ(self, serie, num_in, num_out, num_lag=1, 
                          shuffle=True, seed=0):
        serie = serie.flatten()
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
        
        if shuffle:
            np.random.seed(seed)
            idx = list(range(len(regressoras)))
            np.random.shuffle(idx)
            regressoras = regressoras[idx,:]
            target = target[idx,:]
            
        regressoras = np.flip(regressoras, axis=1)
        
        return regressoras, target

    def get_dataset(self, central, size_train, size_val, seed=None):
        # Abrindo dados
        df = pd.read_csv(f"{self.dir_dataset}/Potência_Horaria_{central}.csv",
                            index_col=0, parse_dates=True)
        
        # Tamanho do conjunto de treinamento e teste
        size_train = floor(size_train * len(df))
        size_val = int((len(df)-size_train) * size_val)
        
        # Treino, valacao e teste
        df_treino = df.iloc[:size_train]
        df_val = df.iloc[size_train:size_train + size_val]
        df_teste = df.iloc[size_train + size_val:]
        
        # Conjuntos de Variáveis Regressoras e Targets:
        X_treino, y_treino = self.__split_regr_targ(
            df_treino.values, self.num_inputs, self.num_outputs, 
            seed=self.seed)
        X_val, y_val = self.__split_regr_targ(
            df_val.values, self.num_inputs, self.num_outputs,
            shuffle=False)
        X_teste, y_teste = self.__split_regr_targ(
            df_teste.values, self.num_inputs, self.num_outputs,
            shuffle=False)
        
        return X_treino, y_treino, X_val, y_val, X_teste, y_teste

if __name__ == '__main__':
    dir_base = os.path.join(os.getcwd(), "base_dados")
    num_inputs = 5
    num_output = 3
    dt = DataLoader(num_inputs, num_output, dir_base)
    X_treino, y_treino, X_val, y_val, X_teste, y_teste = dt.get_dataset(
        "OAKLAND1", size_train=0.8, size_val=0.5)

    # print(list(reversed(X_treino[0])))
    print(X_treino[0])
    print(y_treino[0])
