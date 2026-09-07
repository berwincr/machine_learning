import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    
    def fit(self, X, y):
        samples, features = X.shape

    
        self.weights = np.zeros(features)
        self.bias = 0

        
        for i in range(self.iterations):


            z = np.dot(X, self.weights) + self.bias

            
            y_pred = sigmoid(z)
            dw = (1 / samples) * np.dot(X.T, (y_pred - y))
            db = (1 / samples) * np.sum(y_pred - y)


            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db



    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        probability = sigmoid(z)

        return np.where(probability >= 0.5, 1, 0)



X = np.array([
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 4],
    [6, 5]
])


# Labels:
# 0 = Fail
# 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])


model = LogisticRegression(
    learning_rate=0.1,
    iterations=1000
)

model.fit(X, y)
predictions = model.predict(X)


print("Weights:", model.weights)
print("Bias:", model.bias)

print("Predictions:", predictions)
print("Actual:", y)