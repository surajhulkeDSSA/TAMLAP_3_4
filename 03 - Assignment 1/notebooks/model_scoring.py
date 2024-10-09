import mlflow


def score_model(model, X_test, y_test):
    with mlflow.start_run(nested=True) as run:
        predictions = model.predict(X_test)
        mlflow.log_param("model_scoring", "completed")
        return predictions
