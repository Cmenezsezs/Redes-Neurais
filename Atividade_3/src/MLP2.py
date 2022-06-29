import numpy as np
from Utils import *


class MLP:
    "Multilayer Perceptron"
    
    def __init__(self, num_inputs, hiddens, num_outputs, act_funcs):
        self.num_inputs = num_inputs
        self.hiddens = hiddens
        self.num_outputs = num_outputs
        self.act_funcs = act_funcs
        
        if len(hiddens) + 1 != len(act_funcs):
            print(f"O número de act_fns precisa ser igual a len(hidden) + 1")
            print(f"A ordem das act_fns se referem a ordem dos hiddens layers")
            exit()
            
        layers = [self.num_inputs] + self.hiddens + [self.num_outputs]
        
        # Criação dos pesos iniciais e bias (Xavier):
        self.bias = []
        self.weights = []
        self.derivadas_bias = []
        self.derivadas_weig = []
        
        for i in range(len(layers) - 1):
            # pesos por camadas
            w = np.random.normal(
                loc=0.0, scale=np.sqrt(6) / (layers[i] + layers[i + 1]),
                size=(layers[i] + 1, layers[i + 1]))
            
            # derivadas por camadas
            d = np.zeros((layers[i] + 1, layers[i + 1]))
            
            self.bias.append(w[0])
            self.weights.append(w[1:])
            self.derivadas_bias.append(d[0])
            self.derivadas_weig.append(d[1:])
            
            # Ativação por camadas
            self.ativacao = []
            for i in range(len(layers)):
                a = np.zeros(layers[i])
                self.ativacao.append(a)
                
    def forward(self, inputs):

        self.ativacao[0] = inputs
        
        # Iterar pelas camadas
        for i, w in enumerate(self.weights):
            # Multiplicação da matriz
            calc_rede = np.dot(self.ativacao[i], w) + self.bias[i]
            
            # Aplicar a função de ativação
            if self.act_funcs[i] == 'relu':
                self.ativacao[i + 1] = relu(calc_rede)
            # Tanh
            elif self.act_funcs[i] == 'sigmoid':
                self.ativacao[i + 1] = sigmoid(calc_rede)
            # Sigmoid
            elif self.act_funcs[i] == 'tanh':
                self.ativacao[i + 1] = tanh(calc_rede)

        # retorna as ativações em cada camada.
            print(self.ativacao[i].shape)

        return self.ativacao[-1]
    
    def backpropagation(self, error):
        for i in reversed(range(len(self.derivadas_weig))):
             # Aplicar a função de ativação
             # ReLU
            if self.act_funcs[i] == 'relu':
                delta = error * relu(self.ativacao[i + 1], gradient=True)
            # Sigmoid
            elif self.act_funcs[i] == 'sigmoid':
                delta = error * sigmoid(self.ativacao[i + 1], gradient=True)
            # Tanh
            elif self.act_funcs[i] == 'tanh':
                delta = error * tanh(self.ativacao[i + 1], gradient=True)


            # Backpropagação do erro

            breakpoint()
            self.derivadas_weig[i] = np.dot(
                self.ativacao[i].reshape(self.ativacao[i].shape[0],-1),
                delta.reshape(delta.shape[0], -1).T)
            self.derivadas_bias[i] = np.dot(
                self.ativacao[i].reshape(self.ativacao[i].shape[0],-1),
                delta.reshape(delta.shape[0], -1).T)
            error = np.dot(delta, self.weights[i].T) + \
                np.dot(delta, self.bias[i].T)
            
            
    def gradient_descendent(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] += self.derivadas[i] * lr
            self.bias[i] += self.derivadas[i] * lr
        
    def fit(self, X_train, y_train, epochs, learning_rate):
        for epoc in range(epochs):
            sum_errors = 0
            for inputs, targets in zip(X_train, y_train):
                outputs = self.forward(inputs)
                error = targets - outputs
                self.backpropagation(error)
                self.gradient_descendent(learning_rate)
                
                sum_errors += self._mse(target, output)
                breakpoint()
                
                
    def _mse(self, target, output):
        return np.average((target - output) ** 2)
      
            
    
    