import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 1:
    # Set memory growth for the second GPU
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Not enough GPUs available.")

#Initialization of Tensors
x = tf.constant(4.0, shape=(1,1), dtype=tf.float32)
y = tf.constant([[1,2,3],[4,5,6]])
print(x)
print(y)

x = tf.ones((3,3)) #Matrix of one
print(x)
x = tf.zeros((2,3)) #matrix of zero
print(x)
x = tf.eye(3)  #I for the identity matrix(eye)
print(x)
x = tf.random.normal((3,3), mean =0, stddev=1)
print(x)
x = tf.random.uniform((1,3), minval=0, maxval=1)
print(x)
x = tf.range(start=1, limit=10, delta=2)
print(x)
x = tf.cast(x, dtype=tf.float64)
print(x)

#Mathematical Operation
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

z = tf.add(x,y)
#z = x+y
print(z)

z = tf.subtract(x,y)
#z = x-y
print(z)

z = tf.divide(x,y)
#z = x/y
print(z)

z = tf.multiply(x,y)
#z = x*y
print(z)

z = tf.tensordot(x, y, axes=1)
print(z)
z = tf.reduce_sum(x*y, axis=0)
print(z)

z = x ** 5 #esponente
print(z)

# Genera una matrice 2x3 di valori casuali distribuiti secondo una distribuzione normale
x = tf.random.normal((2,3))

# Genera una matrice 3x4 di valori casuali distribuiti secondo una distribuzione normale
y = tf.random.normal((3,4))

# Esegue il prodotto matriciale tra le matrici x e y
z = tf.matmul(x,y)
print(z)
z = x @ y
print(z)

#Indexing
x = tf.constant([0,1,1,2,3,1,2,3])
print(x[:])
print(x[1:]) #lo 0 sarà escluso
print(x[1:3]) #inclusi dall'elemento in uno al 3 con 3 escluso
print(x[::2]) #salta ogni due elementi
print(x[::-1]) #printa al contrario

indices = tf.constant([0,3])
x_int = tf.gather(x, indices)
x = tf.constant([[1,2],
                 [3,4],
                 [5,6]])
print(x[0,:]) #tutti gli elementi nell'indice 0: [1,2]
print(x[0:2,:]) #elementi da 0 fino a 2 con il 2 escluso


#Reshaping
x = tf.range(9)
print(x)

x = tf.reshape(x, (3,3)) #prende x e li inserisce in una matrice 3x3
print(x)

x = tf.transpose(x, perm=[1,0])
print(x)

