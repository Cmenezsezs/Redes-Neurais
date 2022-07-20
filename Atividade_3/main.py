import os
import pandas as pd
import numpy as np
import plotly.express as px
import optuna
import tensorflow as tf
import pmdarima as pm

from time import time
from Utils import *
from datetime import datetime
from math import floor
from sklearn.preprocessing import StandardScaler


SEED = 0 
tf.random.set_seed(SEED)
optuna.logging.set_verbosity(0)

# # ARIMA
# def obj_arima(trial):
    


# MLP
def obj_mlp(trial):
    # Número de camadas:
    funcoes = ["relu", "sigmoid", "tanh"]
    ativacao = trial.suggest_categorical('ativacao', funcoes)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    
    
    # Criação do modelo
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.layers.Dense(
        num_in, activation=ativacao, kernel_initializer=ini, 
        bias_initializer=ini))
    
    for i in range(n_layers):
        modelo.add(tf.keras.layers.Dense(
            trial.suggest_int(f'n_units_{i}', 32, 128),
            kernel_initializer=ini, bias_initializer=ini,
            activation=ativacao))
    
    modelo.add(tf.keras.layers.Dense(
        num_out, kernel_initializer=ini, bias_initializer=ini))
    
    otimizador = tf.keras.optimizers.Adam(
        learning_rate=trial.suggest_categorical(
            "lr", [0.1, 0.01, 0.001, 0.0001, 0.00001]))
    modelo.compile(optimizer=otimizador, loss='mse')
    
    modelo.fit(X_train, y_train, epochs=100, verbose=0, batch_size=96)
    
    mse = modelo.evaluate(X_valid, y_valid, verbose=0)
    
    return mse


def obj_lstm_vanilla(trial):
    # Número de camadas:
    funcoes = ["relu", "sigmoid", "tanh"]
    ativacao = trial.suggest_categorical('ativacao', funcoes)
        
    # Criação do modelo
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.layers.LSTM(
        trial.suggest_int(f'n_units_0', 32, 128),
        activation=ativacao,
        input_shape=(num_in, num_feat),
        kernel_initializer=ini, bias_initializer=ini))
    
    modelo.add(tf.keras.layers.Dense
               (num_out, kernel_initializer=ini, bias_initializer=ini))
    otimizador = tf.keras.optimizers.Adam(
        learning_rate=trial.suggest_categorical(
            "lr", [0.1, 0.01, 0.001, 0.0001, 0.00001]))
    modelo.compile(optimizer=otimizador, loss='mse')
    
    modelo.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], num_feat)), 
               y_train, epochs=100, verbose=0, batch_size=96)
    
    mse = modelo.evaluate(X_valid.reshape((X_valid.shape[0], X_valid.shape[1], num_feat)), 
                          y_valid, verbose=0)
    
    return mse


def obj_lstm_stacked(trial):
    # Número de camadas:
    n_layers = trial.suggest_int('n_layers', 0, 2)
    funcoes = ["relu", "sigmoid", "tanh"]
    ativacao = trial.suggest_categorical('ativacao', funcoes)
    
    # Criação do modelo
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.layers.LSTM(
        trial.suggest_int('n_units_0', 32, 128),
        activation=ativacao, kernel_initializer=ini, 
        bias_initializer=ini, input_shape=(num_in, num_feat), 
        return_sequences=True))
    
    for i in range(n_layers):
        modelo.add(tf.keras.layers.LSTM(
            trial.suggest_int(f'n_units_{i+1}', 32, 128),
            activation=ativacao, kernel_initializer=ini, 
            bias_initializer=ini, return_sequences=True))
    
    modelo.add(tf.keras.layers.LSTM(
        trial.suggest_int(f'n_units_{n_layers+1}', 32, 128),
        activation=ativacao, kernel_initializer=ini, 
        bias_initializer=ini))
    
    modelo.add(tf.keras.layers.Dense(
        num_out, kernel_initializer=ini, bias_initializer=ini))
    
    otimizador = tf.keras.optimizers.Adam(
        learning_rate=trial.suggest_categorical(
            "lr", [0.1, 0.01, 0.001, 0.0001, 0.00001]))
    modelo.compile(optimizer=otimizador, loss='mse')
        
    modelo.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], num_feat)), 
               y_train, epochs=100, verbose=0, batch_size=96)
    
    mse = modelo.evaluate(X_valid.reshape((
        X_valid.shape[0], X_valid.shape[1], num_feat)), y_valid, verbose=0)
    
    return mse


def obj_lstm_bidirecional(trial):
    # Número de camadas:
    funcoes = ["relu", "sigmoid", "tanh"]
    ativacao = trial.suggest_categorical('ativacao', funcoes)
    
    # Criação do modelo
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(trial.suggest_int('n_units_0', 32, 128),
        activation=ativacao, kernel_initializer=ini, 
        bias_initializer=ini, input_shape=(num_in, num_feat))))
    modelo.add(tf.keras.layers.Dense(
        num_out, kernel_initializer=ini, bias_initializer=ini))
    
    otimizador = tf.keras.optimizers.Adam(
        learning_rate=trial.suggest_categorical(
            "lr", [0.1, 0.01, 0.001, 0.0001, 0.00001]))
    modelo.compile(optimizer=otimizador, loss='mse')
        
    modelo.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], num_feat)), 
               y_train, epochs=100, verbose=0, batch_size=96)
    
    mse = modelo.evaluate(X_valid.reshape((
        X_valid.shape[0], X_valid.shape[1], num_feat)), y_valid, verbose=0)
    
    return mse
    
# Separando datasets em estudo:
dataset_path = os.path.join(os.getcwd(), "Dados")
centrais = ("GUNNING1",)

escala = "Horaria"
plot = False

# Parâmetros de treinamento
num_in = 5
num_out = 12
num_feat = 1
num_trials = 50

# Treinamentos
train_arima = True
train_mlp = True
train_lstm_vanilla = True
train_lstm_stacked = True
train_lstm_bidirecional = True
train_persistencia = True

for central in centrais:
    # Abrindo dataset
    df = pd.read_csv(os.path.join(dataset_path, 
                                  f"Potência_{escala}_{central}.csv"),
                     parse_dates=True, index_col=0)
    
    # Ajustando datasets em sazonalidade
    data1 = pd.date_range(start=df.index[0], 
                          end=pd.to_datetime("2018-04-01"),
                          freq="30T", inclusive="left")
    data2 = pd.date_range(start=pd.to_datetime("2018-04-01"), 
                          end=pd.to_datetime("2018-07-01"),
                          freq="30T", inclusive="left")
    data3 = pd.date_range(start=pd.to_datetime("2018-07-01"), 
                          end=pd.to_datetime("2018-10-01"),
                          freq="30T", inclusive="left")
    data4 = pd.date_range(start=pd.to_datetime("2018-10-01"), 
                          end=df.index[-1],
                          freq="30T", inclusive="both")

    df1 = df[central][data1].to_frame().rename(
        columns={central: "Jan-Mar"})
    df2 = df[central][data2].to_frame().rename(
        columns={central:"Abr-Jun"})
    df3 = df[central][data3].to_frame().rename(
        columns={central: "Jul-Set"})
    df4 = df[central][data4].to_frame().rename(
        columns={central: "Out-Dez"})
    
    df = pd.concat([df1, df2, df3, df4], axis=1)
    
    if plot:
        fig = px.line(df, x=df.index, y=df.columns, 
                      labels={"index":"Meses", "value":"Potência", 
                              "variable": "Intervalos"},
                      title=f"Potência da central {central} no ano de 2018")
        fig.show()
        
    for interv in df.columns:
        df_estudo = df[interv].copy()
        df_estudo.dropna(inplace=True)
        
        # Definição dos tamanhos das séries de treinamento, validação e teste
        # 2/3 para treinamento, 1/3 para teste
        size_train = floor((1 / 2) * len(df_estudo))
        size_valid = floor((1 / 3) * (len(df_estudo) - size_train))
        
        df_train = df_estudo.iloc[:size_train]
        df_test = df_estudo.iloc[size_train:]
        df_valid = df_test.iloc[:size_valid]
                
        # Salvando série de teste para avaliação futura:
        df_test.last("M").to_csv(f"Resultados/Teste_{central}_{escala}_{interv}.csv")
        
        # Conjuntos de variáveis regressoras
        X_train, y_train = split_regr_targ(df_train.values, num_in, num_out)
        X_valid, y_valid = split_regr_targ(df_valid.values, num_in, num_out)
        X_test, y_test = split_regr_targ(df_test.values, num_in, num_out)
        
        # Normalizando base de dados
        escala_regr = StandardScaler()
        escala_targ = StandardScaler()
        
        # Variáveis Regressoras e Targets:
        X_train = escala_regr.fit_transform(X_train)
        y_train = escala_targ.fit_transform(y_train)
        X_valid = escala_regr.transform(X_valid)
        y_valid = escala_targ.transform(y_valid)
        X_test = escala_regr.transform(X_test)
        y_test = escala_targ.transform(y_test)
        
        
        # Inicializador das Redes Neurais
        ini = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.01, seed=SEED)
        
        if train_persistencia:
            aux = pd.concat([df_valid, df_test], axis=0)
            previsoes = pd.DataFrame([])
            for i in range(num_out):
                previsoes[f"Persistência-{i+1}"] = aux.shift(i+1)
            
            previsoes.last("M").to_csv(f"Resultados/Teste_{central}_{escala}_{interv}Persistência.csv")
            
        # ARIMA - BASELINE
        if train_arima:
            print("Tempo - ARIMA")
            t1 = time()
            
            # Treinamento
            modelo = pm.auto_arima(
                df_train, start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5,
                seasonal=False, error_action='warn', trace=True, stepwise=False, 
                supress_warnings=True, random_state=SEED)
            
            # Previsão
            previsao = modelo.predict(len(df_valid) + len(df_test))  
            previsao = pd.DataFrame(
                previsao, index=pd.date_range(
                    start=df_valid.index[0], freq="30T", 
                    periods=len(df_valid) + len(df_test)))
            
            previsoes = pd.DataFrame([])
            for ii in range(num_out):
                previsoes[f"ARIMA-{ii+1}"] = previsao.shift(ii)
            
            aux = pd.concat([df_test, previsoes], axis=1)
            fig=px.line(aux, x=aux.index, y=aux.columns)
            fig.show() 
            breakpoint()          
            forecasts = model.predict(test.shape[0])
            print((time() - t1))
            print()
            parametros = study.best_params
        
        # MLP
        if train_mlp:
            print("Tempo - MLP")
            t1 = time()
            study = optuna.create_study(study_name='MLP', direction='minimize')
            study.optimize(obj_mlp, n_trials=num_trials)
            print((time() - t1))
            print(central, interv, study.best_params)
            print()
            parametros = study.best_params
            
            modelo = tf.keras.Sequential()
            modelo.add(tf.keras.layers.Dense(
                num_in, activation=parametros['ativacao']))
            
            for i in range(parametros['n_layers']):
                modelo.add(tf.keras.layers.Dense(
                    parametros[f'n_units_{i}'],
                    activation=parametros['ativacao']))
            
            modelo.add(tf.keras.layers.Dense(num_out))
            otimizador = tf.keras.optimizers.Adam(
                learning_rate=parametros['lr'])
            modelo.compile(optimizer=otimizador, loss='mse')
            
            modelo.fit(X_train, y_train, epochs=100, verbose=0, batch_size=96)
            previsoes = escala_targ.inverse_transform(modelo.predict(X_test))
            previsoes = pd.DataFrame(previsoes, 
                                     columns=[f"MLP-{i+1}" 
                                              for i in range(num_out)])
            
            for i in range(num_out):
                previsoes[f"MLP-{i+1}"] = previsoes[f"MLP-{i+1}"].shift(i)
            if escala == "Horaria":
                previsoes.index = pd.date_range(end=df_test.index[-1], 
                                                freq="30T", 
                                                periods=len(previsoes))
            else:
                previsoes.index = pd.date_range(end=df_test.index[-1], 
                                                freq="D", 
                                                periods=len(previsoes))
            
            previsoes.last("M").to_csv(
                f"Resultados/Teste_{central}_{escala}_{interv}MLPs.csv")
    
        # LSTM-Vanilla
        if train_lstm_vanilla:
            print("Tempo - LSTM-Vanilla")
            t1 = time()
            study = optuna.create_study(study_name='LSTM-Vanilla', 
                                        direction='minimize')
            study.optimize(obj_lstm_vanilla, n_trials=num_trials)
            print((time() - t1))
            print(central, interv, study.best_params)
            print()
            parametros = study.best_params
        
            # Criação do modelo
            modelo = tf.keras.Sequential()
            modelo.add(tf.keras.layers.LSTM(
                parametros['n_units_0'],
                activation=parametros['ativacao'],
                input_shape=(num_in, num_feat)))
            
            modelo.add(tf.keras.layers.Dense(num_out))
            otimizador = tf.keras.optimizers.Adam(
                learning_rate=parametros['lr'])
            modelo.compile(optimizer=otimizador, loss='mse')
            
            modelo.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], num_feat)), 
                    y_train, epochs=100, verbose=0, batch_size=96)
            
            previsoes = escala_targ.inverse_transform(
                modelo.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], num_feat))))
            previsoes = pd.DataFrame(previsoes, 
                                     columns=[f"LSTM-Vanilla-{i+1}" 
                                              for i in range(num_out)])
            
            for i in range(num_out):
                previsoes[f"LSTM-Vanilla-{i+1}"] = previsoes[f"LSTM-Vanilla-{i+1}"].shift(i)
            if escala == "Horaria":
                previsoes.index = pd.date_range(end=df_test.index[-1], 
                                                freq="30T", 
                                                periods=len(previsoes))
            else:
                previsoes.index = pd.date_range(end=df_test.index[-1], 
                                                freq="D", 
                                                periods=len(previsoes))
            
            previsoes.last("M").to_csv(
                f"Resultados/Teste_{central}_{escala}_{interv}LSTM-Vanillas.csv")
            
        # LSTM-Stacked
        if train_lstm_stacked:
            print("Tempo - LSTM-Stacked")
            t1 = time()
            study = optuna.create_study(study_name='LSTM-Stacked', 
                                        direction='minimize')
            study.optimize(obj_lstm_stacked, n_trials=num_trials)
            print((time() - t1))
            print(central, interv, study.best_params)
            print()
            parametros = study.best_params
            
            ativacao = parametros['ativacao']
            n_layers = parametros['n_layers']
            # Criação do modelo
            modelo = tf.keras.Sequential()
            modelo.add(tf.keras.layers.LSTM(
                parametros['n_units_0'],
                activation=ativacao, kernel_initializer=ini, 
                bias_initializer=ini, input_shape=(num_in, num_feat), 
                return_sequences=True))
            
            for i in range(n_layers):
                modelo.add(tf.keras.layers.LSTM(
                    parametros[f'n_units_{i+1}'],
                    activation=ativacao, kernel_initializer=ini, 
                    bias_initializer=ini, return_sequences=True))
            
            modelo.add(tf.keras.layers.LSTM(
                parametros[f'n_units_{n_layers+1}'],
                activation=ativacao, kernel_initializer=ini, 
                bias_initializer=ini))
            
            modelo.add(tf.keras.layers.Dense(
                num_out, kernel_initializer=ini, bias_initializer=ini))
            
            otimizador = tf.keras.optimizers.Adam(
                learning_rate=parametros["lr"])
            modelo.compile(optimizer=otimizador, loss='mse')
                
            modelo.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], num_feat)), 
                    y_train, epochs=100, verbose=0, batch_size=96)
            previsoes = escala_targ.inverse_transform(
                modelo.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], num_feat))))
            previsoes = pd.DataFrame(previsoes, 
                                     columns=[f"LSTM-Stacked-{i+1}" 
                                              for i in range(num_out)])
            
            for i in range(num_out):
                previsoes[f"LSTM-Stacked-{i+1}"] = previsoes[f"LSTM-Stacked-{i+1}"].shift(i)
            if escala == "Horaria":
                previsoes.index = pd.date_range(end=df_test.index[-1], 
                                                freq="30T", 
                                                periods=len(previsoes))
            else:
                previsoes.index = pd.date_range(end=df_test.index[-1], 
                                                freq="D", 
                                                periods=len(previsoes))
            
            previsoes.last("M").to_csv(
                f"Resultados/Teste_{central}_{escala}_{interv}LSTM-Stackeds.csv")

        # LSTM-Bidirecional
        if train_lstm_bidirecional:
            print("Tempo - LSTM-Bidirecional")
            t1 = time()
            study = optuna.create_study(study_name='LSTM-Bidirecional', 
                                        direction='minimize')
            study.optimize(obj_lstm_bidirecional, n_trials=num_trials)
            print((time() - t1))
            print(central, interv, study.best_params)
            print()
            parametros = study.best_params
            
            ativacao = parametros['ativacao']
            # Número de camadas:
            
            # Criação do modelo
            modelo = tf.keras.Sequential()
            modelo.add(tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(parametros['n_units_0'],
                activation=ativacao, kernel_initializer=ini, 
                bias_initializer=ini, input_shape=(num_in, num_feat))))
            modelo.add(tf.keras.layers.Dense(
                num_out, kernel_initializer=ini, bias_initializer=ini))
            
            otimizador = tf.keras.optimizers.Adam(
                learning_rate=parametros["lr"])
            modelo.compile(optimizer=otimizador, loss='mse')
            
            modelo.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], num_feat)), 
                    y_train, epochs=100, verbose=0, batch_size=96)
            previsoes = escala_targ.inverse_transform(
                modelo.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], num_feat))))
            previsoes = pd.DataFrame(previsoes, 
                                     columns=[f"LSTM-Bidirecional-{i+1}" 
                                              for i in range(num_out)])
            
            for i in range(num_out):
                previsoes[f"LSTM-Bidirecional-{i+1}"] = previsoes[f"LSTM-Bidirecional-{i+1}"].shift(i)
            if escala == "Horaria":
                previsoes.index = pd.date_range(end=df_test.index[-1], 
                                                freq="30T", 
                                                periods=len(previsoes))
            else:
                previsoes.index = pd.date_range(end=df_test.index[-1], 
                                                freq="D", 
                                                periods=len(previsoes))
            
            previsoes.last("M").to_csv(
                f"Resultados/Teste_{central}_{escala}_{interv}LSTM-Bidirecionals.csv")

