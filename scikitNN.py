import numpy as np
from sknn.mlp import Regressor


# Create some random data
X = np.matrix(np.random.sample([80, 5]))
theta = np.matrix([1.0, 2.0, 3.0, 4.0, 5.0]).T
y = X * theta + 6

# Create the neural network
layers = (5, 5, 1)
network = Regressor(layers)

network.fit(X, y)

X = np.matrix(np.random.sample([5, 5]))
for sample in X:
    sample = np.matrix(sample)
    pred1 = network.predict(sample)
    pred2 = sample * Theta
    print pred1, pred2
    
