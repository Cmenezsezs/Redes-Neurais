import os
import numpy as np
from DataLoader import DataLoader
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

seed = 1
np.random.seed(seed)

n_features = 5
n_output = 1
dir_base = os.path.join(os.getcwd(), 'base_dados')

dt = DataLoader(n_features, n_output, dir_base, seed=seed)
X_train, y_train, X_val, y_val, X_test, y_test = dt.get_dataset(
        "OAKLAND1", size_train=0.8, size_val=0.5, seed=seed)

# Normalização da base de dados N~(média=0, desv_pad=1)
escala_regr = StandardScaler() 
X_train = escala_regr.fit_transform(X_train)
X_val = escala_regr.transform(X_val)
X_test = escala_regr.transform(X_test)

escala_targ = StandardScaler() 
y_train = np.array([aux[0] for aux in escala_targ.fit_transform(y_train)])
y_val = np.array([aux[0] for aux in escala_targ.transform(y_val)])
y_test = np.array([aux[0] for aux in escala_targ.transform(y_test)])

tam_base_train = y_train.shape[0]
tam_base_val = y_val.shape[0]
tam_base_test = y_test.shape[0]

modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(64), 
    tf.keras.layers.Dense(32), 
    tf.keras.layers.Dense(16), 
    tf.keras.layers.Dense(1, activation='sigmoid'), 
])

modelo.compile(optimizer='adam', loss='mse')
modelo.fit(X_train, y_train, epochs=5, verbose=1)

print(modelo.evaluate(X_test, y_test, verbose=1))
print(X_test[0], modelo.predict(X_test[0]), y_test[0])
