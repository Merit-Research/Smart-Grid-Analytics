import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from severity import Severity
from algoFunctions import f1_scores


# Set up prediction and anomaly classes
severity = Severity(0.5)

# Generate random target-prediction data
t = np.array(range(5000))
targets = np.cos(t / 10.0)
predictions = targets + 0.2 * np.random.normal(0.0, 0.5, len(t))
ground_truth = np.random.randint(len(t), size=10)
for i in ground_truth:
    predictions[i] += 0.5
    print i
    
error = severity.update(targets, predictions)
anomalies, pvalues = severity.check(targets, predictions)

detected = [i for i, x in enumerate(anomalies) if x]
ground_truth = ground_truth.tolist()
ground_truth.sort()
print detected
print ground_truth
f1_scores(set(detected), set(ground_truth))

plt.figure()

plt.subplot(221)
plt.plot(t, targets, t, predictions)
plt.title("Targets and Predictions")
plt.show(block=False)

plt.subplot(222)
plt.plot(t, error)
plt.title("Error")
plt.show(block=False)

plt.subplot(223)
plt.plot(t, pvalues)
plt.title("P-Values")
plt.show(block=False)

plt.subplot(224)
n, bins, patches = plt.hist(error, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
#y = mlab.normpdf( bins, mu, sigma)
#y = mlab.normpdf( bins, 0, 1)
#l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Standard Deviations from Mean')
plt.ylabel('Probability')
plt.title('Gaussian Distribution')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.tight_layout()
plt.show()

