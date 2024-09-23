import numpy as np

class Perceptron:
  def __init__(self, n_inputs, epochs=100, eta=0.5):
    self.n_inputs = n_inputs
    self.epochs = epochs
    self.eta = eta
    self.bias = 0
    self.weights = np.zeros(n_inputs)

  def activation(self, weighted_sum):
    return 1 if weighted_sum >= 0 else 0

  def predict(self, X):
    z = np.dot(X, self.weights) + self.bias
    return self.activation(z)

  def fit(self, X, y):
    for _ in range(self.epochs):
      error_total = 0
      for i in range(len(X)):
        # predecir
        prediction = self.predict(X[i])
        # error
        error = y[i] - prediction

        # actualizar pesos
        for w in range(len(self.weights)):
          self.weights[w] += self.eta * error * X[i][w]

        self.bias += self.eta * error
        error_total += error ** 2
        
        
      print(f"EPOCHS {_+1} ------------> error: {error_total}")

perceptron = Perceptron(3, epochs=125)
        
      
