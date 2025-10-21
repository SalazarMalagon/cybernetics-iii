import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
import keras
from keras import layers

# Reproducibilidad
np.random.seed(7)
tf.random.set_seed(7)

# Cargar datos .mat
print("Cargando datos...")
here = os.path.dirname(os.path.abspath(__file__))
print(here)
dog_mat = loadmat(os.path.join(here, "dogData_w.mat"))
cat_mat = loadmat(os.path.join(here, "catData_w.mat"))

# Asumiendo variables 'dog_wave' y 'cat_wave' dentro de los .mat
dog_wave = np.array(dog_mat["dog_wave"])
cat_wave = np.array(cat_mat["cat_wave"])

# x train y test (columnas = muestras en MATLAB -> transponer para Keras)
X_train = np.hstack([dog_wave[:, :40], cat_wave[:, :40]]).T  # (80, n_features)
X_test  = np.hstack([dog_wave[:, 40:80], cat_wave[:, 40:80]]).T  # (80, n_features)

# One-hot labels: 1ra mitad perros [1,0], 2da mitad gatos [0,1]
y_train = np.vstack([np.tile([1, 0], (40, 1)), np.tile([0, 1], (40, 1))])  # (80,2)
y_test  = y_train.copy()

# Modelo: equivalente a patternnet(10) con tansig -> Dense(10, tanh) + Dense(2, softmax)
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(10, activation="tanh"),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=8,
    verbose=0  # cambia a 1 si quieres ver el progreso
)

# Evaluación
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Train - loss: {train_loss:.4f}, acc: {train_acc:.4f}")
print(f"Test  - loss: {test_loss:.4f}, acc: {test_acc:.4f}")

# Predicciones (análogas a y y y2 en MATLAB)
y_train_pred = model.predict(X_train, verbose=0)  # (80,2)
y_test_pred  = model.predict(X_test, verbose=0)   # (80,2)

# Clases (1 o 2 para imitar vec2ind 1-based)
classes_train = np.argmax(y_train_pred, axis=1) + 1
classes_test  = np.argmax(y_test_pred, axis=1) + 1

# Gráficas tipo MATLAB
plt.figure(figsize=(10, 8))
plt.subplot(4,1,1); plt.bar(np.arange(len(y_train_pred)), y_train_pred[:,0], color=[0.6,0.6,0.6], edgecolor='k'); plt.title("Train - salida clase 1")
plt.subplot(4,1,2); plt.bar(np.arange(len(y_train_pred)), y_train_pred[:,1], color=[0.6,0.6,0.6], edgecolor='k'); plt.title("Train - salida clase 2")
plt.subplot(4,1,3); plt.bar(np.arange(len(y_test_pred)),  y_test_pred[:,0],  color=[0.6,0.6,0.6], edgecolor='k'); plt.title("Test - salida clase 1")
plt.subplot(4,1,4); plt.bar(np.arange(len(y_test_pred)),  y_test_pred[:,1],  color=[0.6,0.6,0.6], edgecolor='k'); plt.title("Test - salida clase 2")
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.subplot(2,1,1); plt.bar(np.arange(len(classes_train)), classes_train, color=[0.6,0.6,0.6], edgecolor='k'); plt.title("Train - clases predichas (1=perro, 2=gato)")
plt.subplot(2,1,2); plt.bar(np.arange(len(classes_test)),  classes_test,  color=[0.6,0.6,0.6], edgecolor='k'); plt.title("Test - clases predichas (1=perro, 2=gato)")
plt.tight_layout()
plt.show()