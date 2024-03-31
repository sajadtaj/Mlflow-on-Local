## 1-install packages

```bash
#bash
pip install requirements.txt
```

## 2-Call Packages

```python
#Python Script

#Pandas
import pandas as pd

#sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Ml flow
import mlflow
from mlflow.models import infer_signature
```

## 3- Run MlFlow Server

**NOTE** : review the links mentioned above for
guidance on connecting to a managed tracking
server, such as the free Databricks Community Edition

in bash:

```bash
#bash
 mlflow server --host 127.0.0.1 --port 8080
```

## 4- Set our tracking server uri for logging

>> mlflow.set_tracking_uri(uri="http://<host>:<port>")
>>

```python
#Python Script
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
```

## 5-Continue By __sklearn__

Train a model and prepare metadata for logging:

```python
#Python

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto", "random_state": 8888,
      }

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate accuracy as a target loss metric
accuracy = accuracy_score(y_test, y_pred)

```

## 6-Create a new MLflow Experiment

```python
#Python

mlflow.set_experiment("My Experiment")
```

## 7-Start an MLflow run

```Python
#Python

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
```

## 8- Load the model back for predictions as a generic Python Function model

```python
# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions  = loaded_model.predict(X_test)
iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

result[:4]
```

output:


| sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | actual\_class | predicted\_class |
| ----------------- | ---------------- | ----------------- | ---------------- | ------------- | ---------------- |
| 6.1               | 2.8              | 4.7               | 1.2              | 1             | 1                |
| 5.7               | 3.8              | 1.7               | 0.3              | 0             | 0                |
| 7.7               | 2.6              | 6.9               | 2.3              | 2             | 2                |
| 6.0               | 2.9              | 4.5               | 1.5              | 1             | 1                |


## 9-End:
>> See: http://127.0.0.1:8080/