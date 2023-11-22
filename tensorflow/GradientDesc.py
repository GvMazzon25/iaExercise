import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer

X,y = load_breast_cancer(return_X_y=True)

print(X.shape) #569(righe), 30(features o colonne)

#y = valori di 0 e 1

tf.keras.backend.clear_session() #resetta tutti i grafi che tensorflow ha generato prima: creerà un nuovo grafo pulito
tf.random.set_seed(10) #seed randomico->esperimento riproducibile
model = tf.keras.models.Sequential([ #creazione modello sequenziale
    tf.keras.layers.Dense(16,activation=tf.keras.activations.relu,input_shape=[30]), #30: numero features
    tf.keras.layers.Dense(32,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid),
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy)
sum = model.summary()
print(sum)

model.save_weights('init_weights.h5')

# Importa il modulo pyplot di Matplotlib con l'alias 'plt'
import matplotlib.pyplot as plt

# Definisci una funzione 'plot_model_loss' che prende in input un modello, dimensione del batch e il numero di epoche
def plot_model_loss(model, batch_size, epochs):

    # Definisci una classe 'LossHistory' che estende la classe Callback di Keras
    # Callback: classe che definisce delle operazioni da fare durante la fase di training
    class LossHistory(tf.keras.callbacks.Callback):
        # Metodo chiamato all'inizio dell'addestramento
        def on_train_begin(self, logs={}):
            # Inizializza una lista vuota per registrare le perdite durante l'addestramento
            self.losses = []

        # Metodo chiamato alla fine di ogni batch durante l'addestramento
        def on_batch_end(self, batch, logs={}):
            # Aggiungi la perdita del batch corrente alla lista delle perdite
            self.losses.append(logs.get('loss'))

        # Metodo chiamato alla fine dell'addestramento
        def on_train_end(self, logs={}):
            # Aggiungi la lista delle perdite alla storia dell'addestramento
            self.model.history.history['batch_loss'] = self.losses
        #L'obbiettivo è quello di prendere il valore della loss alla fine del training di ogni batch in cui è diviso il datasets

    # Addestra il modello utilizzando i dati di input 'X' e le etichette 'y'
    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=epochs, verbose=0, #quanto output deve avere durante il training
                        shuffle=True,
                        # Utilizza il callback LossHistory durante l'addestramento
                        callbacks=[LossHistory()])

    # Crea un oggetto figura di Matplotlib con dimensioni specificate
    fig = plt.figure(figsize=(10, 5))

    # Scegli quale tipo di perdita plottare in base alla dimensione del batch
    to_plot = 'loss' if batch_size >= X.shape[0] else 'batch_loss'

    # Plotta la curva della perdita nel corso delle epoche
    plt.plot(range(len(history.history[to_plot])), history.history[to_plot])

    # Aggiungi un titolo al grafico specificando la dimensione del batch
    plt.title(label=f'Batch size: {batch_size}')

    # Aggiungi una griglia al grafico
    plt.grid(True)

    # Mostra il grafico
    plt.show()

    # Restituisci la storia dell'addestramento
    return history


X.shape

#BGD
h = plot_model_loss(model,1000,20)


#MBGD
model.load_weights('init_weights.h5')
h = plot_model_loss(model,30,20) #derivata più grande, passo più grande e loss su meno parametri
plt.plot(range(50), h.history['batch_loss'][50:100])
plt.grid(True)
plt.show()


#SGD
model.load_weights('init_weights.h5') #modello lento
h = plot_model_loss(model,1,20) #Il batch è costituito da un elemento alla volta
plt.plot(range(50), h.history['batch_loss'][50:100])#Pià
plt.grid(True)
plt.show()