import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image

# Definizione della struttura della rete neurale
model = Sequential()

# Aggiunta dei layer convoluzionali e di pooling
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Collegamento dei layer densamente connessi
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))  # Output binario (0 o 1)

# Compilazione del modello
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Preparazione dei dati di training
data_directory = 'Man_Woman'  # Cartella contenente 'man' e 'woman'
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    data_directory,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    classes=['man', 'woman'],
    subset='training'

)

test_generator = train_data_gen.flow_from_directory(
    data_directory,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    classes=['man', 'woman'],
    subset='validation'
)

# Addestramento del modello
model.fit(train_generator, epochs=25, validation_data=test_generator)

# Funzione per prevedere il genere in base all'immagine
def predict_gender(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return "uomo" if prediction < 0.5 else "donna"

# Inserimento e previsione di un'immagine fornita dall'utente
user_image_path1 = 'user_image/ale-foto.jpeg'  # Inserisci il percorso dell'immagine 1
user_image_path2 = 'user_image/myphoto.jpeg'  # Inserisci il percorso dell'immagine 2

result1 = predict_gender(user_image_path1)
print(f"L'immagine 1 è di un {result1}.")

result2 = predict_gender(user_image_path2)
print(f"L'immagine 2 è di un {result2}.")