import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_model(X_train, y_train, X_test, y_test):
    with mlflow.start_run(nested=True) as run:
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("mse", mse)
        return model, mse
