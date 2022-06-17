import numpy as np
from Layer import Layer

class MLP:
    def __init__(self, num_inputs, hiddens, num_outputs, act_fns):
        self.num_inputs = num_inputs
        self.hiddens = hiddens
        self.num_outputs = num_outputs
        self.act_fns = act_fns

        if len(hiddens) + 1 != len(act_fns):
            print(f"O número de act_fns precisa ser igual a len(hidden) + 1")
            print(f"A ordem das act_fns se referem a ordem dos hiddens layers")
            exit()

        self.layers = [Layer(num_inputs, hiddens[0], act_fns[0])]
        for i in range(len(hiddens) - 1):
            self.layers.append(Layer(hiddens[i], hiddens[i+1], act_fns[i]))
        self.layers.append(Layer(hiddens[-1], num_outputs, act_fns[-1]))

    def forward(self, inputs, training=False):
        out = inputs
        for layer in self.layers:
            out = layer.forward(out, training)
        return out[0]

    def backpropagation(self, error, lr):
        for layer in reversed(self.layers):
            error = layer.backpropagation(error, lr)

    def mostrar_rede(self):
        print(f"Arquitetura do Modelo:")
        for layer in self.layers:
            print(f"{layer.tam_input}->{layer.num_neuronios} ({layer.act_fn})")
        print("")

if __name__ == '__main__':
    from Utils import *
    np.random.seed(1)

    num_inputs = 2
    hiddens = [50, 100, 50]
    num_outputs = 1
    act_fns = ["none", "none", "none", "none"] # "relu", "tanh", "sigmoid", "none"
    mlp = MLP(num_inputs, hiddens, num_outputs, act_fns)
    mlp.mostrar_rede()

    vet_mean_loss = []
    tam_max_vet_mean_loss = 10
    lr = 0.01
    for i in range(10000):
        inputs = np.random.rand(num_inputs)
        gt = 0.0 if np.mean(inputs) <= 0.5 else 1.0

        pred = mlp.forward(inputs, training=True)
        loss = mse(gt, pred, gradient=False)
        vet_mean_loss.append((pred <= 0.5 and gt == 0.0)) or (pred > 0.5 and gt == 1.0)
        if len(vet_mean_loss) > tam_max_vet_mean_loss: vet_mean_loss.pop(0)
        if i % 10 == 0:
            print(f"-> {i}, loss: {loss}, acc: {(vet_mean_loss.count(True)/tam_max_vet_mean_loss)*100}%")
        
        mlp.backpropagation(mse(gt, pred, gradient=True), lr)

    # print(f"acc: {vet_mean_loss.count(True)}")