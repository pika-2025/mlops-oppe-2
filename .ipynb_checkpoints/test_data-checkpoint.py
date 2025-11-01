import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# =================================================================
# == Pytest Fixtures: Setup for loading data and model once.
# =================================================================

@pytest.fixture(scope='session')
def data():
    """Fixture to load the training data."""
    try:
        return pd.read_csv('data/iris.csv')
    except FileNotFoundError:
        pytest.fail("The training data file 'data/iris.csv' was not found.")

@pytest.fixture(scope='session')
def model():
    """Fixture to load the trained model."""
    try:
        return joblib.load('artifacts/model/model.pkl')
    except FileNotFoundError:
        pytest.fail("The model file 'models/model.joblib' was not found. Please train the model first.")

@pytest.fixture(scope='session')
def test_data():
    """Fixture to load the test data."""
    try:
        return pd.read_csv('data/iris.csv')
    except FileNotFoundError:
        pytest.fail("The test data file 'data/iris.csv' was not found.")


# =================================================================
# == Data Validation Tests (Pre-Training)
# =================================================================

def test_column_presence(data):
    """Check if required columns are present in the training data."""
    required_columns = ['sepal_length','sepal_width','petal_length','petal_width'] # Example columns
    assert all(col in data.columns for col in required_columns), "One or more required columns are missing."

def test_no_null_values(data):
    """Check for any null values in the training dataset."""
    assert data.isnull().sum().sum() == 0, "Null values found in the training dataset."

def test_feature_data_types(data):
    """Check if features have the expected data types."""
    # Example: Assuming all features should be floats
    feature_cols = data.drop('species', axis=1)
    assert all(feature_cols.dtypes == 'float64'), "Feature columns have incorrect data types."

def test_target_column_values(data):
    """Check if the target column has expected values (e.g., for classification)."""
    # Example for a binary classification task
    expected_values = {"setosa", "virginica", "versicolor"}
    actual_values = set(data['species'].unique())
    assert actual_values.issubset(expected_values), f"Target column contains unexpected values: {actual_values - expected_values}"

# =================================================================
# == Model Evaluation Tests (Post-Training)
# =================================================================

def test_model_performance(model, test_data):
    """Check if model accuracy is above a certain threshold."""
    MIN_ACCURACY = 0.85 # Your performance threshold

    X_test = test_data.drop('species', axis=1)
    y_test = test_data['species']

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= MIN_ACCURACY, f"Model accuracy {accuracy:.2f} is below the threshold {MIN_ACCURACY}."

def test_prediction_output_shape(model, test_data):
    """Check if the prediction output has the correct shape."""
    X_test = test_data.drop('species', axis=1)
    predictions = model.predict(X_test)
    assert predictions.shape == (len(X_test),), "Prediction output shape is incorrect."
