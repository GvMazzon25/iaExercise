import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

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

#Indexing

#Reshaping

