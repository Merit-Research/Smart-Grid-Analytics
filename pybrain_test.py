import numpy as np

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer, LinearLayer
from pybrain.supervised.trainers import BackpropTrainer

print "Initializing network..."
net = buildNetwork(5, 5, 1, bias=True, hiddenclass=LinearLayer)

print net['in'].dim

print "Adding data..."
ds = SupervisedDataSet(5, 1)

for i in xrange(10):
    X = np.random.sample((5))
    y = np.sum((5 * X) + 3)
    ds.addSample(X, [y])

print "Training..."
trainer = BackpropTrainer(net, ds)
#trainer.trainUntilConvergence()

stop = False
while trainer.train() > 1e-10:
    pass
    #stop = int(raw_input('stop?'))

X1 = np.asarray([1, 2, 3, 4, 5])
X2 = np.asarray([2, 2, 2, 2, 2])
X3 = np.asarray([1, 0.5, 0.2, 1.4, 0.1])
print net.activate(X1)
print np.sum((5 * X1) + 3)
print net.activate(X2)
print np.sum((5 * X2) + 3)
print net.activate(X3)
print np.sum((5 * X3) + 3)


