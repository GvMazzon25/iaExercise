import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.standard_normal((200,1))
#print(X)
y = 2 * X + 3

plt.plot(X,y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Relazione Lineare: y = 2X + 3')
plt.grid(True)
# Mostra il grafico
plt.show()

#y è uguale a 2*x+3. Allenando il modello w e b dovrebbero avvicinarsi a 2 e 3

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])  #ha un solo neurone/input_shape: dimensione dei dati, il numero di colonne
])
model.summary()

#tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=tf.keras.optimizers.Adam())
history = model.fit(x=X, y=y, epochs=1000)

print(history)

y_pred = model.predict(X)

table = np.c_[y,y_pred]

print(table)

loss_hist = history.history['loss']
plt.plot(range(1000), loss_hist)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Relazione Lineare: y = 2X + 3')
plt.grid(True)
# Mostra il grafico
plt.show()

w1 = model.layers[0].weights
print(w1)