# -*- coding: utf-8 -*-
"""
CIS6550 Project

Model Extraction
"""

# Imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from typing import Callable
from sklearn.linear_model import LogisticRegression
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Data
print("----- IMPORT DATA -----")
data = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
print(data.DESCR)

# Train Model
print("----- TRAIN MODEL -----")
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Model")
plt.show()

print("----- MODEL PARAMS-----")
print(model.coef_)       # the true model weights
print(model.intercept_)  # the true model biases

# Demo: Extract Model Weights
def logit(p: float, eps: float = 1e-12) -> float:
    """
    Given a probability p, return the logit of p.
    :param p: probability
    :param eps: small number to avoid log(0)
    :return: logit of p
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def estimate_derivative(model, x, i):
    """
    Estimate the derivative of the model with respect to the i-th coefficient.
    :param model: sklearn model
    :param x: input
    :param i: index of the coefficient
    :return: derivative of the model with respect to the i-th coefficient
    """
    e = np.zeros(x.shape)
    e[i-1] = 1                      # i-th standard basis vector
    return model(x + e) - model(x)  # finite difference

def predict(W, b, X):
    """
    Predictions for input X. For demonastration/comparison of accuracy.
    :param W: weights
    :param b: bias
    :param X: input
    :return: predictions
    """
    logits = W.dot(X.T) + b              # raw logits
    sigmoid = 1 / (1 + np.exp(-logits))  # sigmoid activation
    return (sigmoid >= 0.5).astype(int)  # binary output

def predict_model(model, X):
    """
    Predictions for input X.
    :param model: model takes single row of X as input
    :param X: input
    :return: predictions
    """
    logits = np.array([[model(X[i])] for i in range(X.shape[0])])
    sigmoid = 1 / (1 + np.exp(-logits))
    return (sigmoid >= 0.5).astype(int)

logit_model = lambda x: logit(model.predict_proba(x.reshape(1,-1))[0][1])

print("----- EXTRACTED PARAMS -----")
W = np.array([estimate_derivative(logit_model, X_test[0], i) for i in range(1,31)])
print(W)  # extracted weights

b = logit_model(X_test[0]) - W.dot(X_test[0])
print(b)  # extracted bias

print("----- EXTRACTED MODEL -----")
y_extracted = predict(W, b, X_test)  # make prediction based on extracted values
print(classification_report(y_test, y_extracted))
cm = confusion_matrix(y_test, y_extracted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Extracted")
plt.show()

# Defense: DP
# This defense technique involves adding Gaussian (normal) noise to the model output. The model prediction is not very sensitive to small amounts of noise, but the method of estimating derivatives using finite differences is very sensitive to this noise.

# The first method of defense: add Gaussian (Normal) noise to model output
dp_model = lambda x: logit(model.predict_proba(x.reshape(1,-1))[0][1]) + np.random.normal(0, 1)

print("----- DP MODEL -----")
y_dp = predict_model(dp_model, X_test)  # use DP model
print(classification_report(y_test, y_dp))
cm = confusion_matrix(y_test, y_dp)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("DP Model")
plt.show()

print("----- DP MODEL EXTRACTED WEIGHTS -----")
W_dp = np.array([estimate_derivative(dp_model, X_test[0], i) for i in range(1,31)])
print(W_dp)  # extracted weights

b_dp = dp_model(X_test[0]) - W_dp.dot(X_test[0])
print(b_dp)  # extracted bias

print("----- EXTRACTED DP MODEL -----")
y_dp_extracted = predict(W_dp, b, X_test)  # make prediction based on extracted values
print(classification_report(y_test, y_dp_extracted))
cm = confusion_matrix(y_test, y_dp_extracted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("DP Extracted")
plt.show()

# As seen above, after applying differential privacy to the model by adding Gaussian noise to the model output, the model performs identically or slightly worse (±0.01), but the extracted model using the previous method is much worse.

# Defense: Limit model output
# This defense technique involves removing access to logits/probability outputs of the model (e.g. remove/overwrite model.predict_proba() function from sklearn model). This makes it much more difficult to extract model parameters. For example the weight extraction method demonstrated above using finite differences would no longer work.
print("----- LIMITED MODEL -----")
limited_model = copy.deepcopy(model)
print(limited_model.predict_proba(X_test[0].reshape(1, -1)))

limited_model.predict_proba = lambda x: None  # Remove probability outputs
print(limited_model.predict_proba(X_test[0].reshape(1, -1)))
