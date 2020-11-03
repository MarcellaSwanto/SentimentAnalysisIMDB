# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier


def split_data(train: pd.CSVDataSet) -> Dict[str, Any]:
    df = pd.read_excel(train)
    df = df.head(10)#error if not: memory allocation

    tfidf = TfidfVectorizer(max_features=5000)
    X = df['Reviews']
    y = df['Sentiment']

    X = [str(item) for item in X]

    X = tfidf.fit_transform(X)

    X_train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = KNeighborsClassifier(n_neighbors=1)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_Test)

    print(classification_report(y_test, y_pred, output_dict=True))

    # When returning many variables, it is a good practice to give them names:
    return dict(
        train_x=X_train,
        train_y=y_train,
        test_x=X_Test,
        test_y=y_test,
    )


# def split_data(data: pd.DataFrame, example_test_data_ratio: float) -> Dict[str, Any]:
#     """Node for splitting the classical Iris data set into training and test
#     sets, each split into features and labels.
#     The split ratio parameter is taken from conf/project/parameters.yml.
#     The data and the parameters will be loaded and provided to your function
#     automatically when the pipeline is executed and it is time to run this node.
#     """
#     data.columns = [
#         "sepal_length",
#         "sepal_width",
#         "petal_length",
#         "petal_width",
#         "target",
#     ]
#     classes = sorted(data["target"].unique())
#     # One-hot encoding for the target variable
#     data = pd.get_dummies(data, columns=["target"], prefix="", prefix_sep="")
#
#     # Shuffle all the data
#     data = data.sample(frac=1).reset_index(drop=True)
#
#     # Split to training and testing data
#     n = data.shape[0]
#     n_test = int(n * example_test_data_ratio)
#     training_data = data.iloc[n_test:, :].reset_index(drop=True)
#     test_data = data.iloc[:n_test, :].reset_index(drop=True)
#
#     # Split the data to features and labels
#     train_data_x = training_data.loc[:, "sepal_length":"petal_width"]
#     train_data_y = training_data[classes]
#     test_data_x = test_data.loc[:, "sepal_length":"petal_width"]
#     test_data_y = test_data[classes]
#
#     # When returning many variables, it is a good practice to give them names:
#     return dict(
#         train_x=train_data_x,
#         train_y=train_data_y,
#         test_x=test_data_x,
#         test_y=test_data_y,
#     )
