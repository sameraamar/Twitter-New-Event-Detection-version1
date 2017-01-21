import time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import random

X_train = [[5, 5, 2], [21, 5, 5], [1, 1, 1], [8, 9, 1], [6, 10, 2]]
X_test = [[9, 1, 6], [3, 1, 10], [7, 10, 3]]

dim=3000

lshf = LSHForest(random_state=42, n_estimators=65, n_candidates=200, n_neighbors=10)



X_train = [ [random.randint(0, 4) for k in range(dim)] for i in range(50)]

for j in range(1000):
    X_test = [ [random.randint(0, 4) for k in range(dim)] for i in range(1)]
    lshf.partial_fit(X_test)
    if j % 50==0:
        print (j)

distances, indices = lshf.kneighbors(X_test, n_neighbors=33)
print(distances, indices)
