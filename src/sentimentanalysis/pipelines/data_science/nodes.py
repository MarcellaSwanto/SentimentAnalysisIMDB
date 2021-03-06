import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


def train_model(
        train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
) -> np.ndarray:
    model = KNeighborsClassifier(n_neighbors=13)
    model.fit(train_x, train_y)
    return model


######
# Whatis the model type for inputting it for predict node?? --> can use pd dataframe
######

# def predict(model: pd.DataFrame, test_x: pd.DataFrame) -> np.ndarray:
def predict(train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame) -> np.ndarray:
    """Node for making predictions given a model and a test set.
    """
    # To be deleted when model is added as an input
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(train_x, train_y)
    ###
    y_pred = model.predict(test_x)

    # print(classification_report(y_test, y_pred, output_dict=True))

    print(y_pred)
    # Return numpy array of predictions
    return y_pred


def report_accuracy(y_pred: np.ndarray, y_test: pd.DataFrame) -> None:
    """Node for reporting the accuracy of the predictions performed by the
    previous node. Notice that this function has no outputs.
    """
    print(classification_report(y_test, y_pred, output_dict=True))

##############################
# IGNORE
##############################
# def train_model(
#     train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
# ) -> np.ndarray:
#     """Node for training a simple multi-class logistic regression model. The
#     number of training iterations as well as the learning rate are taken from
#     conf/project/parameters.yml. All of the data as well as the parameters
#     will be provided to this function at the time of execution.
#     """
#     num_iter = parameters["example_num_train_iter"]
#     lr = parameters["example_learning_rate"]
#     X = train_x.to_numpy()
#     Y = train_y.to_numpy()
#
#     # Add bias to the features
#     bias = np.ones((X.shape[0], 1))
#     X = np.concatenate((bias, X), axis=1)
#
#     weights = []
#     # Train one model for each class in Y
#     for k in range(Y.shape[1]):
#         # Initialise weights
#         theta = np.zeros(X.shape[1])
#         y = Y[:, k]
#         for _ in range(num_iter):
#             z = np.dot(X, theta)
#             h = _sigmoid(z)
#             gradient = np.dot(X.T, (h - y)) / y.size
#             theta -= lr * gradient
#         # Save the weights for each model
#         weights.append(theta)
#
#     # Return a joint multi-class model with weights for all classes
#     return np.vstack(weights).transpose()
#
#
# def predict(model: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
#     """Node for making predictions given a pre-trained model and a test set.
#     """
#     X = test_x.to_numpy()
#
#     # Add bias to the features
#     bias = np.ones((X.shape[0], 1))
#     X = np.concatenate((bias, X), axis=1)
#
#     # Predict "probabilities" for each class
#     result = _sigmoid(np.dot(X, model))
#
#     # Return the index of the class with max probability for all samples
#     return np.argmax(result, axis=1)
#
#
# def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
#     """Node for reporting the accuracy of the predictions performed by the
#     previous node. Notice that this function has no outputs, except logging.
#     """
#     # Get true class index
#     target = np.argmax(test_y.to_numpy(), axis=1)
#     # Calculate accuracy of predictions
#     accuracy = np.sum(predictions == target) / target.shape[0]
#     # Log the accuracy of the model
#     log = logging.getLogger(__name__)
#     log.info("Model accuracy on test set: %0.2f%%", accuracy * 100)
#
#
# def _sigmoid(z):
#     """A helper sigmoid function used by the training and the scoring nodes."""
#     return 1 / (1 + np.exp(-z))
