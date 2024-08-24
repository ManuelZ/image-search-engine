"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# External imports
import tensorflow as tf
from tensorflow.keras.applications import resnet, densenet
from tensorflow.keras import layers


def get_embedding_module(image_size):
    """ """

    inputs = tf.keras.Input(image_size + (3,))

    baseCnn = densenet.DenseNet201(weights="imagenet", include_top=False)
    baseCnn.trainable = False

    x = densenet.preprocess_input(inputs)
    x = baseCnn(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(units=128)(x)
    embedding = tf.keras.Model(inputs, outputs, name="embedding")

    return embedding


def get_siamese_network(image_size, embedding_model):
    """ """

    anchor_input = tf.keras.Input(name="anchor", shape=image_size + (3,))
    positive_input = tf.keras.Input(name="positive", shape=image_size + (3,))
    negative_input = tf.keras.Input(name="negative", shape=image_size + (3,))

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    siamese_network = tf.keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_embedding, positive_embedding, negative_embedding],
    )
    return siamese_network


class SiameseModel(tf.keras.Model):
    def __init__(self, siamese_net, margin, **kwargs):
        super().__init__()
        self.siamese_net = siamese_net
        self.margin = margin
        self.lossTracker = tf.keras.metrics.Mean(name="loss")

    def _compute_distance(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """

        (anchor_embedding, positive_embedding, negative_embedding) = self.siamese_net(
            inputs
        )

        apDistance = tf.reduce_sum(
            tf.square(anchor_embedding - positive_embedding), axis=-1
        )

        anDistance = tf.reduce_sum(
            tf.square(anchor_embedding - negative_embedding), axis=-1
        )
        return (apDistance, anDistance)

    def get_config(self):
        """
        See:
        https://www.tensorflow.org/guide/keras/serialization_and_saving#custom_objects
        https://www.tensorflow.org/guide/keras/serialization_and_saving#config_methods
        """
        base_config = super().get_config()
        config = {
            "siamese_net": self.siamese_net,
            "margin": self.margin,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config["siamese_net"] = tf.keras.layers.deserialize(config["siamese_net"])
        return cls(**config)

    def _compute_loss(self, apDistance, anDistance):
        loss = apDistance - anDistance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def call(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """
        return self._compute_distance(inputs)

    def train_step(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """

        with tf.GradientTape() as tape:
            (apDistance, anDistance) = self._compute_distance(inputs)
            loss = self._compute_loss(apDistance, anDistance)

        gradients = tape.gradient(loss, self.siamese_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_net.trainable_variables)
        )
        self.lossTracker.update_state(loss)
        return {"loss": self.lossTracker.result()}

    def test_step(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """

        (apDistance, anDistance) = self._compute_distance(inputs)
        loss = self._compute_loss(apDistance, anDistance)
        self.lossTracker.update_state(loss)
        return {"loss": self.lossTracker.result()}

    @property
    def metrics(self):
        return [self.lossTracker]
