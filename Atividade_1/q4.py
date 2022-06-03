from scipy.optimize import minimize,Bounds
import numpy as np

N = 4
# Possibilidades de entrada do XOR
x_xor = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
# Resultados das operações do XOR
y = np.array([       0,        1,      1,       0]) 
alpha = np.array([.4065, .4065, .4065, .4065]) # Inicialização do alpha

# Kernel definido no Exemplo 20.1
def kernel(x, xi):
    aux_1 = (x[0] ** 2) * (xi[0] ** 2)
    aux_2 = 2 * x[0] * x[1] * xi[0] * xi[1]
    aux_3 = (x[1] ** 2) * (xi[1] ** 2)
    aux_4 = 2 * x[0] * xi[0]
    aux_5 = 2 * x[1] * xi[1]
    return 1 + aux_1 + aux_2 + aux_3 + aux_4 + aux_5

# Função objetivo 21.7
def obj(a):
    aux_1, aux_2 = 0, 0
    for p in range(N):
        for i in range(N):
            k = kernel(x_xor[p], x_xor[i])
            aux_1 += 1/2 * y[p] * y[i] * k * a[p] * a[i]
        aux_2 += a[p]
    return aux_1 - aux_2

# Derivada parcial da obj (21.7) em a[p]
def jac_obj(a):
    matriz = np.zeros(N)
    aux_1, aux_2 = 0, 0
    for p in range(N):
        for i in range(N):
            k = kernel(x_xor[p], x_xor[i])
            aux_1 += y[p] * y[i] * k * a[i]
        aux_2 += 1
    return aux_1 - aux_2

# Restrição 21.8
def resticao_1(a):
    aux = 0
    for p in range(N):
        aux += y[p] * a[p]
    return aux

# Restrição 21.9 para intervalos de Alpha
restic_alpha = (0, 1)

# Constrói a função de otimização
res = minimize(
    fun = lambda aux_lambda: obj(alpha),
    jac = lambda aux_lambda: jac_obj(alpha),
    x0 = np.ones(N),
    method = "Powell",
    constraints = {'type' : 'eq', 'fun' : resticao_1},
    bounds = (restic_alpha, restic_alpha, restic_alpha, restic_alpha))

print("\n\nValor da função após otimizada",res.fun)
print("Valor das variáveis após otimizada", res.x, '\n')    
