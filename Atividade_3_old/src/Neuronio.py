import numpy as np
from Utils import *
from random import gauss

class Neuronio:

    def __init__(self, tam_input, act_fn = None):
        # self.bias_pesos = (np.random.rand(tam_input + 1) - 0.5) / 10
        self.bias_pesos = np.random.normal(loc=0.0, 
                                           scale=(np.sqrt(6) / \
                                               (np.sqrt(tam_input)))/5,
                                           size=tam_input+1)
        self.tam_input = tam_input
        self.act_fn = act_fn.lower()
        self.tmp_inputs = None

    def activate(self, x, gradient=False):
        if self.act_fn   == "sigmoid": return sigmoid(x, gradient)
        elif self.act_fn == "relu":    return relu(x, gradient)
        elif self.act_fn == "tanh":    return tanh(x, gradient)
        elif self.act_fn != "none":
            print(f"sct_fn {self.act_fn} not implemented"); exit()
        return x

    def forward(self, inputs, training=False):
        # Mesma coisa de: np.sum(inputs * self.bias_pesos[1:]) + self.bias_pesos[0]
        # Mesma coisa de: np.sum(np.concatenate((np.array([1]), inputs)) * self.bias_pesos)
        out = np.dot(np.concatenate((np.array([1]), inputs)), self.bias_pesos)
        if training: self.tmp_inputs = inputs
        return self.activate(out, gradient=False)
    
    def backpropagation(self, output_error_scalar, lr):
        # Calcula o gradiente da função de ativação caso tenha (backward) (dE/Dactfn)
        output_error_scalar = self.activate(output_error_scalar, gradient=True)

        # Atualização dos pesos com base no erro e dados de entrada (dE/dX)
        weights_error = np.dot(self.tmp_inputs.T, output_error_scalar)
        self.bias_pesos[1:] -= lr * weights_error
        self.bias_pesos[0] -= lr * output_error_scalar

        # TODO: só pra testar
        self.bias_pesos[self.bias_pesos > 1] = 1
        self.bias_pesos[self.bias_pesos < -1] = -1

        # Gradiente dos pesos para propagação do erros nos layers (backward) (dE/dw)
        # print(output_error_scalar, self.bias_pesos)
        inputs_error = np.array([output_error_scalar * w for w in self.bias_pesos[1:]])

        # TODO: só pra testar
        # inputs_error[inputs_error > 1] = 1
        # inputs_error[inputs_error < -1] = -1

        return inputs_error



if __name__ == '__main__':
    np.random.seed(1)

    n = Neuronio(10, act_fn='none')
    inputs = np.random.rand(10)
    
    print(n.forward(inputs))