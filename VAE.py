#!/usr/bin/python3 
import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from nn_helper_functions import sigmoid_cross_entropy_with_logits, display_autoencoder_output

tf.enable_eager_execution()


# Requires data to be of form (N, M), where N is size of batch and M is length of input data vector
class BasicVAE():

    def __init__(self, input_shape, NumLatentVars, 
                 encoder_hidden_units = 200, decoder_hidden_units = 200,
                 num_samples=1):

        self.num_samples = num_samples

        self.input_size = input_size = np.prod(input_shape)
        w1 = tf.Variable(tf.random_normal([input_size, encoder_hidden_units], 0.0, 0.001))
        w2 = tf.Variable(tf.random_normal([encoder_hidden_units, NumLatentVars*2], 0.0, 0.001))

        w3 = tf.Variable(tf.random_normal([NumLatentVars, decoder_hidden_units], 0.0, 0.001))
        w4 = tf.Variable(tf.random_normal([decoder_hidden_units, input_size], 0.0, 0.001))

        self.weights = [w1,w2,w3,w4]

    def encoder(self, x):
        h = tf.nn.relu(x@self.weights[0])
        logit = h@self.weights[1]
        return logit

    def decoder(self, z):
        """ 
        This function expects an input shape of [batch, vector]
        """
        h = tf.nn.relu(z@self.weights[2])
        logit = h@self.weights[3]
        return logit

    def predict(self, X, l=1, return_logit_and_parameters=False):
        """
        This function assumes data in the range [0,1]
        If l=1 (default), then returns shape (batch_size, input_size)
        If l!=1, then returns shape (l, batch_size, input_size)
        """
        logit, _ = self._predict(X, l)
        return tf.nn.sigmoid(logit)

    def _predict(self, X, l=1):
        """
        Function for internal use, returns logit and parameters
        """
        parameters_layer = self.encoder(X)
        z_sample, z_parameters = self.g(parameters_layer,num_samples=l)
        #reshape so new batch size is l*original_batch size
        z_sample = tf.reshape(z_sample, (-1, z_sample.shape[-1]))
        logit = self.decoder(z_sample)
        if(l!=1):
            logit = tf.reshape(logit, (l,-1, self.input_size))

        return logit, z_parameters

    def train(self, X, num_epochs=10, batch_size=100):
        """
        Yields control back every epoch, returning the average ELBO across batches
        """
        opt = tf.train.AdagradOptimizer(0.01)
        for i in range(num_epochs):
            loss_history = []
            for batch in range(X.shape[0]//batch_size):
                start = batch*batch_size
                end = (batch+1)*batch_size

                with tf.GradientTape() as tape:
                    loss = -self.ELBO(X[start:end])

                loss_history.append(loss.numpy())

                gradients = tape.gradient(loss, self.weights)
                opt.apply_gradients(zip(gradients, self.weights))

            yield -np.mean(loss_history)

    def ELBO(self, X):
        logit, z_parameters = self._predict(X, l=self.num_samples)
        z_mu, z_sigma = z_parameters
        # Separate KL divergence and Expected reconstruction Loss
        logp = self.reconstructionLoss(logit, X)
        kl = self.KLdivergence((z_mu, z_sigma))
        elbo = logp - kl
        return elbo

    def reconstructionLoss(self, logits, labels):
        # uses my version of sigmoid cross entropy which supports broadcasting
        cross_entropy = sigmoid_cross_entropy_with_logits(labels=labels,logits=logits)

        # Account for multiple samples
        if(tf.rank(cross_entropy).numpy() == 3):
            cross_entropy = tf.reduce_mean(cross_entropy,axis=0)

        # Average across
        logp = -tf.reduce_sum(
                tf.reduce_mean(cross_entropy,axis=0),
                axis=-1)

        assert(tf.rank(logp).numpy() == 0)

        return logp

    def KLdivergence(self, z_parameters):
        z_mu, z_sigma = z_parameters

        kl = 0.5*tf.reduce_sum(z_mu**2 + z_sigma**2 - tf.log(z_sigma) - 1.0,axis=-1)
        return kl

    def g(self, parameters_layer, num_samples=1):
        z_mu, z_log_sigma = tf.split(parameters_layer,2,axis=-1)
        z_sigma = tf.exp(z_log_sigma)

        z_parameters = (z_mu, z_sigma)

        size = (num_samples, *tf.shape(z_mu))
        
        z_sample = z_mu + tf.random_normal(size)*z_sigma
        return z_sample, z_parameters

class BasicVAE2(BasicVAE):

    ''' KL Divergence from a prior whose parameters can be changed '''
    def KLdivergence(self, z_parameters):
        z_mu, z_sigma = z_parameters
        p_mu, p_sigma = 0.0, 1.0
        z_sigma_squared = z_sigma**2
        p_sigma_squared = p_sigma**2
        kl = 0.5*tf.reduce_sum(-2*tf.log(z_sigma)-1.0 + 2*tf.log(p_sigma)
                               + ((z_mu - p_mu)**2)/p_sigma_squared 
                               + z_sigma_squared/p_sigma_squared,axis=-1)
        return kl



(X, Y), (X_test, Y_test) = mnist.load_data()
X = X.astype('float32')/255.
X_test = X_test.astype('float32')/255.
X = tf.reshape(X, shape=[-1, 28*28])
X_test = tf.reshape(X_test, shape=[-1, 28*28])

# Init
vae = BasicVAE(input_shape=(28,28), NumLatentVars=30,
                encoder_hidden_units = 400, decoder_hidden_units = 400, num_samples=1)

# Train
for elbo in vae.train(X,num_epochs=10,batch_size=200):
    print(f"Estimated ELBO: {elbo}")

# Predict
predicted_images = vae.predict(X_test)
pred_images = np.reshape(predicted_images.numpy(),(-1,28,28))
X_test = np.reshape(X_test,(-1,28,28))


display_autoencoder_output(X_test, pred_images)

