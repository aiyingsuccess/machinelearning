import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


train, test = tf.keras.datasets.mnist.load_data()

x_train, y_train = train
x_test, y_test = test

train_in = np.reshape( x_train, (-1, 28, 28) ) / 255
test_in = np.reshape( x_test, (-1, 28, 28) ) / 255
train_out = tf.keras.utils.to_categorical( y_train, 10 )
test_out = tf.keras.utils.to_categorical( y_test, 10 )

def generate_training_batch( batch_size, timesteps ):
    data_values = np.zeros( [batch_size, timesteps, 28] )
    indices = np.random.choice( np.arange(60000), batch_size, replace=False )
    
    for i in range( batch_size ):
        data_values[ i, 0:timesteps, :] = train_in[ indices[i], 0:timesteps, :]
        
    tag_vectors = train_out[ indices ]
    tag_values = y_train[ indices ]
    
    return data_values, tag_vectors, tag_values

def generate_testing_batch( batch_size, timesteps ):
    data_values = np.zeros( [batch_size, timesteps, 28] )
    indices = np.random.choice( np.arange(10000), batch_size, replace=False )
    
    for i in range( batch_size ):
        data_values[ i, 0:timesteps, :] = test_in[ indices[i], 0:timesteps, :]
        
    tag_vectors = test_out[ indices ]
    tag_values = y_test[ indices ]
    
    return data_values, tag_vectors, tag_values


layer_init = tf.variance_scaling_initializer()

#tf.nn.rnn_cell.BasicRNNCell
cell = tf.nn.rnn_cell.LSTMCell( num_units = 50, activation = tf.nn.tanh, initializer = layer_init )
data = tf.placeholder(tf.float32, [None, None, 28])
output, state = tf.nn.dynamic_rnn(cell, data,  dtype=tf.float32)

classification_output = tf.layers.dense( inputs = output, units = 10, activation = tf.nn.softmax,
    kernel_initializer = layer_init, bias_initializer = layer_init )

final_classification = classification_output[:,-1,:]

true_classification = tf.placeholder( tf.float32, [None, 10] )

loss = tf.reduce_mean( (-1)*(true_classification * tf.log( final_classification ) + ( 1 - true_classification ) * tf.log( 1 - final_classification )) )

predicted_digit = tf.argmax( final_classification, axis = 1 )

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize( loss )

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    def train_on_diverse( batch_size ):
        for i in range( 60000 // batch_size ):
            timesteps = np.random.randint( 28 ) + 1
            data_batch, tag_batch, tag_values = generate_training_batch( batch_size, timesteps )
            sess.run( train, { data : data_batch, true_classification : tag_batch } )
    
    def test_accuracy( timesteps ):
        data_batch, tag_batch, tag_values = generate_testing_batch( 10000, timesteps )
        
        predicted_digits = sess.run( predicted_digit, feed_dict = { data: data_batch } )
        
        wrong_count = np.sum(predicted_digits != tag_values)
        
        return 1 - wrong_count / 10000
    
    print("Prior To Training")
    accuracies = [ test_accuracy( 7 ), test_accuracy( 14 ), test_accuracy( 21 ), test_accuracy( 28 ) ]
    print("Accuracy:", accuracies )
    input()
    
    for i in range( 10 ):
        print("Round",i)
        train_on_diverse( batch_size = 100 )
        accuracies = [ test_accuracy( 7 ), test_accuracy( 14 ), test_accuracy( 21 ), test_accuracy( 28 ) ]
        print("Accuracy:", accuracies )