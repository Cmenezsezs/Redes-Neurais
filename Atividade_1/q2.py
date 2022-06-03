import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import dist
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def funcao_1(x):
    return (1 - x - 2 * (x ** 2)) * np.exp(- (x ** 2) / 2)
    

def funcao_2(x):
    return 2 * np.sin(x[:,0] / 2) * np.cos(x[:,1] / 2)


def datasets(dados, funcao):
    y = funcao(dados)
    return y.reshape((-1, 1))


def kmeans_parametros(num_clusters, X, seed):
    # Construindo modelo K-Means
    kmeans = KMeans(num_clusters, random_state=seed)
    kmeans.fit(X)
    
    # Definindo os parâmetros
    centroids = kmeans.cluster_centers_
    
    distancia = np.zeros((X.shape[0], num_clusters))
    
    for cent in range(centroids.shape[0]):
        distancia[:,cent] = np.array([dist(X[i,:], centroids[cent,:])
                                      for i in range(X.shape[0])])
    
    min_distancia = np.argmin(distancia, axis=1)
    
    covariancia = np.array([np.cov(X[min_distancia==i,:].T)
                            for i in range(len(centroids))])
    
    return centroids, covariancia

class RBFnetwork:
    
    def __init__(self, num_clusters, lr=1e-2, tol=1e-5, seed=0):
        np.random.seed(seed)
        self.num_clusters = num_clusters
        self.lr = lr
        self.tol = tol
        self.seed = seed
        self.pesos = np.random.randn(num_clusters + 1, 1)
        self.scaler = StandardScaler()
        
        
    def fit(self, X, y, epocas):
        self.X_train = X
        self.y_train = y
        
        self.centroids, self.covariancia = kmeans_parametros(self.num_clusters, 
                                                             self.X_train,
                                                             self.seed)
        
        convergiu = False
        loss_ant = np.random.randn()
        epc = 0
        
        self.y_train_scaler = self.scaler.fit_transform(self.y_train)
        self.historico = []
        pesos_anterior = self.pesos
        
        while(epc < epocas) and not convergiu:
            epc += 1
            loss = []
            
            for instancia in range(self.X_train.shape[0]):
                # Vetor auxiliar de pesos da RBF
                auxiliar = [1]
                for cent, covar in zip(self.centroids, self.covariancia):
                    gauss = multivariate_normal(mean=cent, cov=covar)
                    auxiliar.append(gauss.pdf(self.X_train[instancia, :]))
                    
                auxiliar = np.array(auxiliar).reshape((-1, 1))
                
                # Previsao
                y_pred = auxiliar.T.dot(self.pesos)
                erro = self.y_train_scaler[instancia] - y_pred[0]
                
                loss.append(erro ** 2)
                
                lr = self.lr / (1 + instancia/self.X_train.shape[0])
                
                self.pesos = self.pesos + lr * auxiliar * erro
            
            loss_epoca = np.mean(loss)
            
            print(f"Época: {epc} --- Loss: {loss_epoca}")
            # print(np.max(np.abs(self.pesos - pesos_anterior)))
            # if np.abs(np.max(self.pesos - pesos_anterior)) < self.tol:
            #     convergiu = True
            
            # pesos_anterior = self.pesos
            
            self.historico.append([epc, loss_epoca])
            
            
    def predict(self, X):
        self.X_test = X
        self.y_pred = np.zeros((self.X_test.shape[0], 1))
        
        for instancia in range(self.X_test.shape[0]):
            # Vetor auxiliar de pesos da RBF
            auxiliar = [1]
            for cent, covar in zip(self.centroids, self.covariancia):
                gauss = multivariate_normal(mean=cent, cov=covar)
                auxiliar.append(gauss.pdf(self.X_test[instancia, :]))
                
            auxiliar = np.array(auxiliar).reshape((-1, 1))
            
            # Previsao
            self.y_pred[instancia] = auxiliar.T.dot(self.pesos)
        
        return self.scaler.inverse_transform(self.y_pred)
        

# Criando dataset

questaoA = False
questaoB = False
# A
if questaoA:
    seed = 0
    
    x = np.linspace(-10, 10, 100)

    X = x.reshape((-1, 1))
    y = datasets(X, funcao_1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,
                                                        test_size=0.2, 
                                                        random_state=seed)
    parametros = []
    
    for num_clusters in range(2, 10):
        for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            modelo = RBFnetwork(num_clusters, lr=lr)
            modelo.fit(X_train, y_train, epocas=100)
            
            previsao = modelo.predict(X_test)
            erro = mean_squared_error(y_test, previsao)
            
            parametros.append([num_clusters, lr, erro])
    
    np.savetxt("parametrosA.txt", parametros)
    
# B 
if questaoB:
    seed = 0
    
    x = np.linspace(0, 10, 50)

    X = []
    for i in x:
        for j in x:
            X.append([i, j])

    X = np.array(X)
    y = datasets(X, funcao_2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,
                                                        test_size=0.2, 
                                                        random_state=seed)
    parametros = []
    
    for num_clusters in range(2, 10):
        for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            modelo = RBFnetwork(num_clusters, lr=lr)
            modelo.fit(X_train, y_train, epocas=100)
            
            previsao = modelo.predict(X_test)
            erro = mean_squared_error(y_test, previsao)
            
            parametros.append([num_clusters, lr, erro])
            print(erro)
    
    
    np.savetxt("parametrosB.txt", parametros)


resultadoA = False
resultadoB = False

if resultadoA:
    parametros = np.loadtxt("parametrosA")
    idx = np.argmin(parametros[:,-1])
    best_params = parametros[idx,:]
    
    
    seed = 0
    
    x = np.linspace(-10, 10, 100)

    X = x.reshape((-1, 1))
    y = datasets(X, funcao_1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,
                                                        test_size=0.2, 
                                                        random_state=seed)
    
    modelo = RBFnetwork(int(best_params[0]), lr=best_params[1])
    modelo.fit(X_train, y_train, epocas=100)
    
    previsao = modelo.predict(X_test)
    erro = mean_squared_error(y_test, previsao)
    print(f"Nº cluster: {int(best_params[0])} - LR: {best_params[1]} - MSE: {erro}")
    
    
if resultadoB:
    parametros = np.loadtxt("parametrosB")
    idx = np.argmin(parametros[:,-1])
    best_params = parametros[idx,:]
    
    seed = 0
    
    x = np.linspace(0, 10, 50)
    X = []
    
    for i in x:
        for j in x:
            X.append([i, j])

    X = np.array(X)
    y = datasets(X, funcao_2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,
                                                        test_size=0.2, 
                                                        random_state=seed)
    
    modelo = RBFnetwork(int(best_params[0]), lr=best_params[1])
    modelo.fit(X_train, y_train, epocas=100)
    
    previsao = modelo.predict(X_test)
    erro = mean_squared_error(y_test, previsao)
    print(f"Nº cluster: {int(best_params[0])} - LR: {best_params[1]} - MSE: {erro}")
