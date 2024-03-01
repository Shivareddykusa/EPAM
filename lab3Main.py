
import tensorflow as tf
import numpy as np
import LinearRegression as ml

np.random.seed(42)
x_train = np.random.rand(100, 1).astype(np.float32)
y_train = 2 * x_train + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

x_train_tensor = tf.constant(x_train)
y_train_tensor = tf.constant(y_train)

model = ml.LinearRegressionModel()

num_epochs = 100
learning_rate = 0.01

for epoch in range(num_epochs):
    loss = ml.train_step(model, x_train_tensor, y_train_tensor, learning_rate)

ml.plot_LR_graph(model, x_train, y_train)
