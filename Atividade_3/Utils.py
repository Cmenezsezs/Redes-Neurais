import numpy as np

# Separação das sequências de treino e teste:
def split_regr_targ(serie, num_in, num_out, num_lag=1):
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
    
    # Ordenando as regressoras
    regressoras = np.flip(regressoras, axis=1)
    
    return regressoras, target

