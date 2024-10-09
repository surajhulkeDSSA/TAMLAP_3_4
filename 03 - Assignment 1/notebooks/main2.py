import mlflow
from data_preparation import prepare_data
from model_training import train_model
from model_scoring import score_model

with mlflow.start_run() as parent_run:
    # Data preparation
    X_train, X_test, y_train, y_test = prepare_data()

    # Model training
    model, mse = train_model(X_train, y_train, X_test, y_test)

    # Model scoring
    predictions = score_model(model, X_test, y_test)
