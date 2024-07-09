import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Creating Linear Regression scratch
class LinearRegression:
    '''
    Implementation of Linear Regression from scratch using python.
    Attributes:
    learning_rate : This attribute is used for gradient descent as learning rate to train the model.
    iterations : Used for training the models, it's also known as epochs.
    formula : f(x) = M.X + C
    X : This variable consists of independent variables data from the dataset.
    y : This consists of dependent variable data from the dataset.
    rows : Number of rows in the dataset.
    cols : Number of columns in the dataset.
    weights : Slope i.e: "M" in formula.
    bias : Intercept i.e: "C" in formula.
    losses : A list to keep track of the loss during training.
    '''
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.losses = []
    
    # Passing dataset to model
    def fit(self, X, y):
        # Number of rows and columns in the dataset.
        self.rows, self.cols = X.shape
        # Initializing weights to zero to train model from origin
        self.weights = np.zeros(self.cols)
        # Initializing bias "C" in formula
        self.bias = 0

        self.X = X
        self.y = y

        # Implementing gradient descent
        for i in range(self.iterations):
            self.update_weights()
            loss = self.calculate_loss()
            self.losses.append(loss)
            if i % 100 == 0:  
                print(f"Iteration {i}: Weights: {self.weights}, Bias: {self.bias}, Loss: {loss}")

        return self
     
    def update_weights(self):
        Y_pred = self.predict(self.X)
        # Implementing derivatives to weight and bias
        derivative_weight = - (2 * (self.X.T).dot(self.y - Y_pred)) / self.rows
        derivative_bias = - 2 * np.sum(self.y - Y_pred) / self.rows

        # Updating weights
        self.weights = self.weights - self.learning_rate * derivative_weight
        self.bias = self.bias - self.learning_rate * derivative_bias

        return self
    
    # Implementing 'f(x)' line formula 
    def predict(self, X):
        return X.dot(self.weights) + self.bias
    
    # Calculate mean squared error loss
    def calculate_loss(self):
        Y_pred = self.predict(self.X)
        loss = np.mean((self.y - Y_pred) ** 2)
        return loss


# loading iris dataset for testing
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Using required features
X = df[['petal length (cm)']].values
y = df['petal width (cm)'].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"""
X_train: {len(X_train)}
X_test: {len(X_test)}
y_train: {len(y_train)}
y_test: {len(y_test)}
""")

model = LinearRegression(iterations=1000, learning_rate=0.01)
model.fit(X, y)

y_preds = model.predict(X_test)
print(np.ravel([np.argmax(i) for i in y_preds]))