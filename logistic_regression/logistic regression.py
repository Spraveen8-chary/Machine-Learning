import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# Creating Logistic Regression from scratch 
class LogisticRegression:
    '''
    Implementation of Logistic Regression from scratch using Python.
    Attributes:
    learning_rate : This attribute is used for gradient descent as learning rate to train the model.
    iterations : Used for training the models, it's also known as epochs.
    formula : Sigmoid function and binary cross-entropy loss.
    X : This variable consists of independent variables data from the dataset.
    y : This consists of dependent variable data.
    rows : Number of rows in the dataset.
    cols : Number of columns in the dataset.
    weights : Coefficients for the independent variables.
    bias : Intercept term.
    losses : List to store the loss values for each iteration.
    '''
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.losses = []

    # Passing dataset to model
    def fit(self, X, y):
        # Number of rows and columns in the dataset.
        rows, cols = X.shape
        # Initializing weights to zero to train model from origin
        self.weights = np.zeros(cols)
        # Initializing bias
        self.bias = 0

        # Implementing gradient descent
        for i in range(self.iterations):
            value = self.update(X)
            loss = self.binary_cross_entropy(y, value)
            self.losses.append(loss)
            derivative_z = value - y
            derivative_weight = (1 / rows) * np.dot(X.T, derivative_z)
            derivative_bias = (1 / rows) * np.sum(derivative_z)
            # Updating weights
            self.weights -= self.learning_rate * derivative_weight
            self.bias -= self.learning_rate * derivative_bias
            if i % 200 == 0:  
                print(f"Iteration {i}: Weights: {self.weights}, Bias: {self.bias}, Loss: {loss}")

    # Sigmoid function
    # Formula: sigmoid(x) = 1 / (1 + exp(-x))
    # The sigmoid function maps any real-valued number into the range (0, 1),
    # which is useful for binary classification as it can be interpreted as a probability.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Implementing Binary Cross Entropy
    # Formula: BCE = -1/N * Σ [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    # where N is the number of samples, y_true is the true label, and y_pred is the predicted probability.
    # This loss function measures the performance of a classification model whose output is a probability value between 0 and 1.
    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-9  # Adding a small constant to avoid log(0)
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    # Updating weights
    def update(self, X):
        z = np.dot(X, self.weights) + self.bias
        value = self.sigmoid(z)
        return value

    # Predict function
    # It returns the predicted class (0 or 1) based on the threshold of 0.5.
    def predict(self, X):
        threshold = 0.5
        y_prediction = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(y_prediction)
        predicted_class = [1 if i > threshold else 0 for i in y_predicted]
        return np.array(predicted_class)

# Using breast cancer datsaet for testing
dataset = load_breast_cancer()
df = pd.DataFrame(data= dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

X = dataset.data
y = dataset.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=0)

model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

y_preds = model.predict(X_test)
mapping = {0 : "malignant", 1 : "benign"}
predictions = [mapping[pred] for pred in y_preds]
print(predictions)

matrix = confusion_matrix(y_preds, y_test)

print("Confusion Matrix : \n", confusion_matrix(y_preds, y_test))
print("Accuracy of our model : ",accuracy_score(y_preds, y_test))

from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(dataset.data ,dataset.target,train_size=0.8, random_state=0)

lmodel = LogisticRegression().fit(X_train, y_train)

y_predtions = lmodel.predict(X_test)
predictions = [mapping[pred] for pred in y_preds]

print("Predictions from in-built model : \n",predictions)
print("\n\nConfusion Matrix : \n",confusion_matrix(y_predtions, y_test))
print("\n\nAccuracy : ",accuracy_score(y_predtions, y_test))



