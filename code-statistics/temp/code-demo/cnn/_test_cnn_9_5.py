#!/usr/bin/python3
"""
test support vector machine 
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm

from __init__ import write_log

log_file = "_test_cnn_9_5.log"

iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target


def my_kernel(x, y):
    """
    We create a custom kernel:

                 (2  0)
    k(x, y) = x  (    ) y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(x, M), y.T)


h = 0.02  # step size in the mesh

# we create an instance of SVM and fit out data.
clf = svm.SVC(kernel=my_kernel)
result = clf.fit(x, y)

write_log(str(type(clf)), file=log_file)
write_log(str(result), file=log_file)

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1


xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
z = z.reshape(xx.shape)
plt.pcolormesh(xx, yy, z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")
plt.title("3-Class classification using Support Vector Machine with custom" " kernel")
plt.axis("tight")
plt.show()
