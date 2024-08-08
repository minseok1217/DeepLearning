import numpy as np
# Checking the shape of the data
# Loading the MNIST dataset
# Non-sklearn packages
# import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Sklearn modules & functions
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Filter:
    def __init__(self, size, num_filters):
        self.size = size
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, size, size, 1) / np.sqrt(size * size)
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, height, width, channels = inputs.shape
        self.output = np.zeros((batch_size, height - self.size + 1, width - self.size + 1, self.num_filters))

        for i in range(self.output.shape[1]):
            for j in range(self.output.shape[2]):
                region = inputs[:, i:i+self.size, j:j+self.size, :]
                # 여기서 필터 차원을 맞추기 위해 np.expand_dims 사용
                region = np.expand_dims(region, axis=-1)  # (batch_size, 3, 3, 1, 1)
                filters_expanded = np.expand_dims(self.filters, axis=0)  # (1, 8, 3, 3, 1)
                self.output[:, i, j] = np.sum(region * filters_expanded, axis=(1, 2, 3))

        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros_like(self.inputs)
        filter_gradient = np.zeros_like(self.filters)

        for i in range(self.output.shape[1]):
            for j in range(self.output.shape[2]):
                region = self.inputs[:, i:i+self.size, j:j+self.size, :]
                for k in range(self.num_filters):
                    input_gradient[:, i:i+self.size, j:j+self.size, :] += output_gradient[:, i, j, k][:, np.newaxis, np.newaxis, np.newaxis] * self.filters[k]
                    filter_gradient[k] += np.sum(output_gradient[:, i, j, k][:, np.newaxis, np.newaxis, np.newaxis] * region, axis=0)

        # Update filters
        self.filters -= learning_rate * filter_gradient

        return input_gradient

class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient):
        return output_gradient * (self.inputs > 0)

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def backward(self, output_gradient):
        return output_gradient  # Backpropagation for Softmax combined with Cross-Entropy is handled differently in practice

class CrossEntropyLoss:
    def forward(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels
        return -np.mean(np.log(predictions[range(len(predictions)), labels]))

    def backward(self):
        n = len(self.predictions)
        grad = self.predictions
        grad[range(n), self.labels] -= 1
        return grad / n

class CNN:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filter = Filter(3, num_filters)
        self.relu = Relu()
        self.softmax = Softmax()
        self.loss = CrossEntropyLoss()
    
    def conv(self, images):
        return self.filter.forward(images)

    def max_pooling(self, image, pool_size=2, stride=2):
        batch_size, height, width, depth = image.shape
        output_height = (height - pool_size) // stride + 1
        output_width = (width - pool_size) // stride + 1
        pooled = np.zeros((batch_size, output_height, output_width, depth))

        for i in range(0, height - pool_size + 1, stride):
            for j in range(0, width - pool_size + 1, stride):
                region = image[:, i:i + pool_size, j:j + pool_size, :]
                pooled[:, i // stride, j // stride, :] = np.max(region, axis=(1, 2))

        return pooled

    def forward(self, images, labels=None):
        conv_output = self.conv(images)
        relu_output = self.relu.forward(conv_output)
        pooled_output = self.max_pooling(relu_output)
        
        # Flatten the output for fully connected layers (if necessary)
        flattened_output = pooled_output.reshape(pooled_output.shape[0], -1)
        
        # Here, you would typically have a fully connected layer before the softmax.
        # For simplicity, let's assume this CNN directly goes to the softmax.
        
        softmax_output = self.softmax.forward(flattened_output)

        if labels is not None:
            loss = self.loss.forward(softmax_output, labels)
            return softmax_output, loss
        else:
            return softmax_output

    def backward(self, learning_rate):
        gradient = self.loss.backward()
        gradient = self.filter.backward(gradient, learning_rate)
        return gradient

def main():
    mnist = datasets.fetch_openml('mnist_784', as_frame=True)
    X, y = mnist.data, mnist.target
    X.shape, y.shape
    X, y = np.array(mnist.data), np.array(mnist.target) 
    y = y.astype(int)
    X_reshaped = []

    for i in range(len(X)):
        X_reshaped.append(X[i].reshape(28, 28))

    # X_reshaped는 이제 2차원 이미지 데이터를 포함하는 리스트입니다.
    X_reshaped = np.array(X_reshaped)
    X_reshaped.shape
    # 데이터 전처리
    X_reshaped = np.expand_dims(X_reshaped, axis=-1)  # (70000, 28, 28, 1)로 변환

    # CNN 모델 생성
    cnn_model = CNN(num_filters=8)

    # Forward pass for a batch of data
    output, loss = cnn_model.forward(X_reshaped, y)

    print("Output shape:", output.shape)
    print("Final output:\n", output)
    print("Loss:", loss)

if __name__ == "__main__":
    main()