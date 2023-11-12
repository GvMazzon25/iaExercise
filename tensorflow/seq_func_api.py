import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 1:
    # Set memory growth for the second GPU
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Not enough GPUs available.")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
y_train = y_train.reshape(-1, 28*28).astype("float32") / 255.0

#Sequential API (Very convinient, not very flexible)
model = keras.Sequential(
    [
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)
    ]
)

model.compile(
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)