from typing import Any, Dict
import pandas as pd
from kedro.extras.datasets.pandas import ExcelDataSet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def split_data(train: ExcelDataSet) -> Dict[str, Any]:
    # print(train.head(5))
    # df = pd.read_excel(train)
    # reading train as ExcelDataSet from kedro's extras pandas datasets
    # transforms it directly into a pandas dataframe
    # so no need to do pd.read_excel
    df = train
    # df = df.head(10)  # error to be solved: memory allocation- try in batches?

    tfidf = TfidfVectorizer(stop_words='english')  # , max_features=5000)
    X = df['Reviews']
    y = df['Sentiment']

    X = [str(item) for item in X]

    # Generate document-term matrix
    X = tfidf.fit_transform(X)

    X_train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return dict(
        train_x=X_train,
        train_y=y_train,
        test_x=X_Test,
        test_y=y_test,
    )

##############################
# IGNORE
##############################
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
