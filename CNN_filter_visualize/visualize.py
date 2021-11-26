import numpy as np
import tensorflow as tf
from tensorflow import keras


class Feature_Extract():
    def __init__(self, model, layer_name, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        
        self.model = model
        self.layer = self.model.get_layer(name=layer_name)
        self.channels = self.layer.output.shape[3]
        self.feature_extractor = keras.Model(inputs=self.model.inputs, outputs=self.layer.output)

    def initialize_image(self):
        img = tf.random.uniform((1, self.img_height, self.img_width, 3))
        return (img-0.5)*0.25

    def compute_loss(self, img, filter_index):
        activation = self.feature_extractor(img)
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)
      
    @tf.function
    def gradient_ascent_step(self, img, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self.compute_loss(img, filter_index)
        grads = tape.gradient(loss, img)
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return loss, img
  
    def visualize_filter(self, filter_index):
        # We run gradient ascent for 20 steps
        iterations = 30
        learning_rate = 10.0
        img = self.initialize_image()
        for iteration in range(iterations):
            loss, img = self.gradient_ascent_step(img, filter_index, learning_rate)

        # Decode the resulting input image
        img = self.deprocess_image(img[0].numpy())
        return loss, img
    
    def deprocess_image(self, img):
        # Normalize array: center on 0., ensure std is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Center crop
        img = img[25:-25, 25:-25, :]

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")
        return img

