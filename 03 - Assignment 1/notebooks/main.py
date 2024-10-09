import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from custom_transformer import CustomTransformer

# Load the dataset
housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df["target"] = housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    housing_df.drop("target", axis=1),
    housing_df["target"],
    test_size=0.2,
    random_state=42,
)

# Create the pipeline
pipeline = Pipeline(
    [
        ("custom_transformer", CustomTransformer()),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline
pipeline_predictions = pipeline.predict(X_test)
pipeline_mse = mean_squared_error(y_test, pipeline_predictions)
print(f"Pipeline Mean Squared Error: {pipeline_mse}")
