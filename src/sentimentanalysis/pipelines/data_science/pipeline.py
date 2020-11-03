from kedro.pipeline import Pipeline, node

from .nodes import predict, report_accuracy, train_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["example_train_x", "example_train_y", "parameters"],
                "knn_model",
            ),
            node(
                predict,
                dict(model="knn_model", test_x="example_test_x"),
                "example_predictions",
            ),
            node(
                report_accuracy,
                ["example_predictions", "example_test_y"],
                None),
        ]
    )
