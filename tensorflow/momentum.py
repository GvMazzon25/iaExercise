import numpy as np
import tensorflow as tf
import tensorflow.keras as k

mnist = tf.keras.datasets.mnist #fotografie di numeri e lettere sottoforma di numeri che vanno da 0 a 255: intensità colore del pixel
(X,y), _= mnist.load_data()

import matplotlib.pyplot as plt

X = X / 255.0

f = plt.figure(figsize=(10,5))
nrows, ncols = 2, 5
axs = f.subplots(nrows, ncols)
for i in range(10):
    ax = axs[i // ncols, i % ncols]  # Corretto l'indice
    ax.imshow(X[i, ...], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

print(f'Y: {np.unique(y)}')

print(X.shape)

X = X.reshape(-1, X.shape[1] * X.shape[2]) #appiattiamo la matrice in vettori 28x28

print(X.shape)

#Funziona che plotta la loss e l'accuracy
def plot_loss_acc(history):
    f = plt.figure(figsize=(15,5))
    axs = f.subplots(1,2)
    axs[0].plot(history.history['loss'])
    axs[1].plot(history.history['sparse_categorical_accuracy'])
    plt.tight_layout()
    for ax in axs.ravel():
        ax.grid(True)
    plt.show()


k.backend.clear_session()
model = k.models.Sequential([
    tf.keras.layers.Dense(8,activation=tf.keras.activations.relu,input_shape=[X.shape[1]]), #785 features
    tf.keras.layers.Dense(16,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(np.unique(y).shape[0], activation=k.activations.softmax),
])
model.save_weights('init_weights.h5')

model.compile(optimizer=k.optimizers.SGD(),
              loss=k.losses.sparse_categorical_crossentropy, metrics=[k.metrics.sparse_categorical_accuracy])
history = model.fit(X,y,epochs=10,batch_size=32)

plot_loss_acc(history)

model.compile(optimizer=k.optimizers.SGD(momentum=0.9),
              loss=k.losses.sparse_categorical_crossentropy, metrics=[k.metrics.sparse_categorical_accuracy])
history = model.fit(X,y,epochs=10,batch_size=32)

plot_loss_acc(history)