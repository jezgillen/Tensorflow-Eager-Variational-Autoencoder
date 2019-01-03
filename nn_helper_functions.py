import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# See https://arxiv.org/abs/1606.08415
def gelu(x):
    return x*tf.nn.sigmoid(1.702*x)

# A replacement for the tensorflow function of the same name,
# but this one can handle broadcasting
def sigmoid_cross_entropy_with_logits(logits, labels):
    x = logits
    z = labels
    return tf.maximum(x,0) - x*z +tf.log(1+tf.exp(-tf.abs(x)))

def display_autoencoder_output(originals, prediction=None):
    indicies = np.random.randint(0, len(originals), size=(50))
    plt.figure(1)
    plt.title('Original|Prediction')
    for i in range(50):
        orig = originals[indicies[i]]

        plt.subplot(10, 10, i*2+1)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.imshow(orig, cmap='gray')

        if(prediction is not None):
            pred = prediction[indicies[i]]
            plt.subplot(10, 10, i*2+2)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.imshow(pred, cmap='gray')
    plt.show()
