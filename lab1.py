import tensorflow as tf


def mcculloch_pitts_neuron(inputs, weights, threshold):
    weighted_sum = tf.reduce_sum(tf.multiply(inputs, weights))

    output = tf.cond(weighted_sum >= threshold, lambda: 1.0, lambda: 0.0)

    return output


inputs = tf.constant([0.5, 0.3, 0.8], dtype=tf.float32)
weights = tf.constant([0.1, 0.4, 0.6], dtype=tf.float32)
threshold = tf.constant(0.7, dtype=tf.float32)

result = mcculloch_pitts_neuron(inputs, weights, threshold)
print(result)
