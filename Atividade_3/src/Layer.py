import numpy as np
from Neuronio import Neuronio

class Layer:
    def __init__(self, tam_input, num_neuronios, act_fn=None):
        self.tam_input = tam_input
        self.num_neuronios = num_neuronios
        self.act_fn = act_fn

        self.layer = [Neuronio(tam_input, act_fn) for _ in range(num_neuronios)]

    def forward(self, inputs, training=False):
        return np.array(list(map(lambda n: n.forward(inputs, training), self.layer)))

    def backpropagation(self, output_error, lr):
        # Vetor com derivadas para propagar o erro para is próximos layers
        inputs_error = [0.0] * self.tam_input
        for i, n in enumerate(self.layer):
            inputs_error += n.backpropagation(output_error[i], lr)
        return inputs_error


if __name__ == '__main__':
    np.random.seed(1)
    tam_input_layer = 5
    num_neuronios = 2
    l = Layer(tam_input_layer, num_neuronios, act_fn='sigmoid')
    
    inputs = np.random.rand(tam_input_layer)
    out = l.forward(inputs)

    print(out)