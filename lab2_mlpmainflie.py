import numpy as np

import LAB2MLP as ml

x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[1],[0]])
n_x = 2
n_y = 1
n_h = 2
model = ml.MLP_TF(n_x,n_h,n_y)
print("Weighta before training:", model.get_weights())
model.train(x_train.T,y_train.T,iterations=10,lr=0.1)
print("Weights after training:",model.get_weights())
newdata = np.array([[1,1]])
prediction = model.predict(newdata)
print(f"Input: {newdata}, Predicted Output: {prediction}")