import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


train, test = tf.keras.datasets.mnist.load_data()

x_train, y_train = train
x_test, y_test = test

train_in = np.reshape( x_train, (-1, 28, 28, 1) ) / 255
test_in = np.reshape( x_test, (-1, 28, 28, 1) ) / 255
train_out = tf.keras.utils.to_categorical( y_train, 10 )
test_out = tf.keras.utils.to_categorical( y_test, 10 )

def generate_plot( images, kcount ):
    images = np.reshape( images, (-1, 28, 28 )) * 255
    
    fig, axarr = plt.subplots(kcount, kcount)
    
    k = 0
    for i in range(kcount):
        for j in range(kcount):
            axarr[i,j].imshow( images[k] )
            axarr[i,j].axis('off')
            k += 1
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
    return

## Display Initial Digits ####################################################
k_count = 20
indices = np.random.choice( np.arange(60000), k_count*k_count, replace=False )
generate_plot(train_in[indices], k_count)
input()
##############################################################################

latent_dimension = 2
layer_width = 256

layer_init = tf.variance_scaling_initializer()
activation_function = tf.nn.elu

## Build the Encoder Layers #########################################################################################
encoder_inputs = tf.placeholder( shape=[None, 28, 28, 1], dtype=tf.float32 )

encoder_flatten = tf.layers.flatten( encoder_inputs )

encoder_layer_1 = tf.layers.dense( inputs = encoder_flatten, units = layer_width, activation = activation_function,
    kernel_initializer = layer_init, bias_initializer = layer_init )

encoder_layer_2 = tf.layers.dense( inputs = encoder_layer_1, units = layer_width, activation = activation_function,
    kernel_initializer = layer_init, bias_initializer = layer_init )

encoder_layer_3 = tf.layers.dense( inputs = encoder_layer_2, units = layer_width, activation = activation_function,
    kernel_initializer = layer_init, bias_initializer = layer_init )
#####################################################################################################################

## Compute Embedding ################################################################################################
z_means = tf.layers.dense( inputs = encoder_layer_3, units = latent_dimension,
    kernel_initializer = layer_init, bias_initializer = layer_init )

z_logvar = tf.layers.dense( inputs = encoder_layer_3, units = latent_dimension,
    kernel_initializer = layer_init, bias_initializer = layer_init )

raw_noise = tf.random_normal( shape = [ tf.shape( encoder_inputs )[0], latent_dimension ], dtype=tf.float32 )

z = z_means + tf.exp( 0.5 * z_logvar ) * raw_noise 
#####################################################################################################################


## Build Decoder ####################################################################################################
decoder_inputs = z

decoder_layer_1 = tf.layers.dense( inputs = decoder_inputs, units = layer_width, activation = activation_function,
    kernel_initializer = layer_init, bias_initializer = layer_init )

decoder_layer_2 = tf.layers.dense( inputs = decoder_layer_1, units = layer_width, activation = activation_function,
    kernel_initializer = layer_init, bias_initializer = layer_init )

decoder_layer_3 = tf.layers.dense( inputs = decoder_layer_2, units = layer_width, activation = activation_function,
    kernel_initializer = layer_init, bias_initializer = layer_init )

decoder_layer_4 = tf.layers.dense( inputs = decoder_layer_3, units = 28*28, activation = tf.nn.sigmoid,
    kernel_initializer = layer_init, bias_initializer = layer_init )

decoder_out = tf.reshape( decoder_layer_4, shape = [-1, 28, 28, 1] )
#####################################################################################################################

## Build Losses #####################################################################################################
reconstruction_loss = (-1) * tf.reduce_mean( ( encoder_inputs * tf.log( decoder_out ) ) + 
    ( 1 - encoder_inputs ) * tf.log( 1 - decoder_out ) )

mean_kl_cost = tf.reduce_mean( 0.5 * tf.reduce_sum( tf.pow( z_means, 2 ), axis = 1 ) )
sigma_kl_cost = tf.reduce_mean( 0.5 * tf.reduce_sum( tf.exp( z_logvar ) - z_logvar - 1, axis = 1 ))

average_mean_distance = tf.reduce_mean( tf.reduce_sum( tf.pow( z_means, 2 ), axis = 1 ) )
average_sigma_value = tf.reduce_mean( tf.exp( z_logvar ) )
#####################################################################################################################

## Build Optimizer/Trainer ##########################################################################################
kl_weight = tf.placeholder( shape = None, dtype = tf.float32 )

total_loss = reconstruction_loss + kl_weight * ( mean_kl_cost + sigma_kl_cost )


optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize( total_loss )
#####################################################################################################################

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    def generate_interpolation( bound, k ):
        delta = 2*bound / (k-1)
        z_values = [ -bound + i*delta for i in range( k ) ]
        
        values = np.zeros( [k*k, 2] )
        for i in range(k):
            for j in range(k):
                values[i*k + j, 0] = z_values[i]
                values[i*k + j, 1] = z_values[j]
        
        generated_images = sess.run( decoder_out, feed_dict = { decoder_inputs : values } )
        generate_plot( generated_images, k )
    
    print("Generating Initial Latent Realization - No Training")
    generate_interpolation( 3.0, 20 )
    input()
    
    batch_size = 100
    replications = 2
    
    def train_one_round( w, verbose = True ):
        for t in range( 60000 // ( batch_size // replications ) ):
            input_images = np.zeros( [ batch_size, 28, 28, 1 ] )
            
            idx = np.random.choice( np.arange(60000), batch_size // replications, replace=False )
            
            for k in range(replications):
                input_images[ (k*batch_size//replications):((k+1)*batch_size//replications), :, :, : ] = train_in[ idx ]
            
            input_dict = { encoder_inputs : input_images, kl_weight : w }
            
            if verbose:
                values_to_compute = [ total_loss, reconstruction_loss, average_mean_distance, average_sigma_value ]
                total_loss_value, loss_value, avg_mean, avg_sig = sess.run( values_to_compute, feed_dict = input_dict )
                print("Total Loss", total_loss_value, "Reconstruction Loss", loss_value, "AvgMean", avg_mean, "AvgVar", avg_sig )
            
            sess.run(train, feed_dict = input_dict )
        values_to_compute = [ total_loss, reconstruction_loss, average_mean_distance, average_sigma_value ]
        total_loss_value, loss_value, avg_mean, avg_sig = sess.run( values_to_compute, feed_dict = input_dict )
        print("Total Loss", total_loss_value, "Reconstruction Loss", loss_value, "AvgMean", avg_mean, "AvgVar", avg_sig )
            
    
    train_one_round( 0.01 )
    print("Generating Latent Realization - One Training Round")
    generate_interpolation( 3.0, 20 )
    input()
    
    
    for r in range( 10 ):
        print("New Round", r)
        train_one_round( 0.01, verbose = False )
    
    print("Generating Latent Realization - Training Round ", r)
    generate_interpolation( 3.0, 20 )
    