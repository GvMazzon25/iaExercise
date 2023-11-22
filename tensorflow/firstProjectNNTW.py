import numpy as np
import tensorflow as tf
from tensorflow import keras as k
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

#print(data)

X,y = data['data'], data['target']

#print(X.shape)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

model = k.Sequential([
    k.layers.Dense(16,activation=k.activations.relu, input_shape=[30]),#30, numero features/numero colonne. 16= numero neuroni
    k.layers.Dense(16,activation=k.activations.relu),
    k.layers.Dense(1,activation=k.activations.sigmoid)
])
model.summary() #print sommario, somme del modello e dei suoi parametri: 30 features, 16 neuroni (1 layer)= 16*30+16= 496P. Nel secondo: 16*16+16 = 272P. Nel terzo: 16*1+1= 17. Totale parametri: 496+272+17=785
#k.utils.plot_model(model, show_shapes=True)

model.compile(optimizer=k.optimizers.Adam(), loss=k.losses.binary_crossentropy,metrics=[k.metrics.binary_accuracy])  #loss: allena il modello, metrics: numero classi predette giuste diviso il totale

epochs = 150
history = model.fit(X,y, epochs=epochs)
print(history)

plt.figure(figsize=(10,5))
plt.plot(range(epochs), model.history.history['loss'])
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(epochs), model.history.history['binary_accuracy'])
plt.grid(True)
plt.show()

y_pred = model.predict(X)
print(y_pred)

y_pred_class = np.where(y_pred[:,0] >= 0.5, 1,0)
accuracy = k.metrics.binary_accuracy(y,y_pred_class)
print(accuracy)