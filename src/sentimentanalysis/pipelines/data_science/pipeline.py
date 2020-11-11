from kedro.pipeline import Pipeline, node

from .nodes import predict, report_accuracy, train_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["example_train_x", "example_train_y", "parameters"],  # parameters to be dealt with
                "knn_model",
            ),
            node(
                predict,
                ["example_train_x", "example_train_y", "example_test_x"],  # model to be inputted
                "example_predictions",
            ),
            node(
                report_accuracy,
                ["example_predictions", "example_test_y"],
                None),
        ]
    )
