# Non-sklearn packages
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Sklearn modules & functions
from sklearn import datasets
from sklearn.model_selection import train_test_split


def data_load():
    # Loading the MNIST dataset
    mnist = datasets.fetch_openml('mnist_784', as_frame=True)
    # Checking the shape of the data
    X, y = np.array(mnist.data), np.array(mnist.target) 
    y = y.astype(int)

    image = X[1].reshape(28, 28)
    for X_, y_ in zip(X, y):
        image = X.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.show()
        print(y)
        break

class filter:
    def __init(self, size, num_filters):
        self.size = size
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, size, size) / size * size



class cnn:
    def __init__(self):
        self.layers = []
        self.pool_size = None
    def rerlu(x):
        return np.maximum(0, x)
    
def main():
    data_load()
    print("Hello world")
    

if __name__ == "__main__":
    main()