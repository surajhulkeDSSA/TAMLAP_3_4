import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def prepare_data():
    with mlflow.start_run(nested=True) as run:
        housing = fetch_california_housing()
        housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
        housing_df["target"] = housing.target

        X_train, X_test, y_train, y_test = train_test_split(
            housing_df.drop("target", axis=1),
            housing_df["target"],
            test_size=0.2,
            random_state=42,
        )

        mlflow.log_param("data_preparation", "completed")
        return X_train, X_test, y_train, y_test
