import pandas as pd
import numpy as np

class DataLoader():

    def __init__(self, num_inputs, num_outputs, dir_dataset):
        self.dir_dataset = dir_dataset
        # Definicao do número de variáveis regressoras:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    # Separacao das sequencias de treino e teste:
    def __split_regr_targ(self, serie, num_in, num_out, num_lag=1):
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
        
        return regressoras, target

    def get_dataset(self, central):
        # centrais: ("OAKLAND1", "MERCER01", "GUNNING1")

        treino = pd.date_range(start="2018-01-01", end="2018-10-01", freq="30T", inclusive='neither')
        teste = pd.date_range(start="2018-10-01", end="2019-01-01", freq="30T", inclusive='left')

        # Abrindo dados
        df = pd.read_csv(f"{self.dir_dataset}/Potência_Horaria_{central}.csv",
                            index_col=0, parse_dates=True)
        
        # Treino, validacao e teste
        df_treino = df[central][treino]
        df_teste = df[central][teste]
        
        # Conjuntos de Variáveis Regressoras e Targets:
        X_treino, y_treino = self.__split_regr_targ(
            df_treino.values, self.num_inputs, self.num_outputs)
        X_teste, y_teste = self.__split_regr_targ(
            df_teste.values, self.num_inputs, self.num_outputs)

        y_treino = np.array(list(map(lambda x: x[0], y_treino)))
        y_teste = np.array(list(map(lambda x: x[0], y_teste)))

        return X_treino, y_treino, X_teste, y_teste

if __name__ == '__main__':
    dir_base = "/home/baltz/Documentos/Mestrado/2022-1/Redes Neurais/git_repo/Redes_Neurais_Mestrado_2022-1/Atividade_3/base_dados"
    num_inputs = 5
    num_output = 3
    dt = Dataloader(num_inputs, num_output, dir_base)
    X_treino, y_treino, X_teste, y_teste = dt.get_dataset("OAKLAND1")
    # print(list(reversed(X_treino[0])))
    print(X_treino[0])
    print(y_treino[0])
