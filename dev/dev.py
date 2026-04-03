import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

# Data
X = np.random.randn(100, 2)
y = X[:, 0] * 10 + np.random.randn(100)

# MLflow setup
mlflow.set_tracking_uri("http://34.133.160.231:5000")
mlflow.set_experiment('demo-experiment')

with mlflow.start_run(run_name='lasso-demo'):
    mlflow.set_tags({"model": "lasso", "dataset": "synthetic"})

    lambda_hp = 0.1
    mlflow.log_params({"lambda": lambda_hp})

    model = Lasso(alpha=lambda_hp)
    model.fit(X, y)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    mlflow.log_metrics({"mse": mse, "r2": r2})

    # Tests artifact logging to GCS
    mlflow.sklearn.log_model(model, "lasso-model")

    print(f"MSE: {mse:.4f}")
    print(f"R2:  {r2:.4f}")
    print("Run ID:", mlflow.active_run().info.run_id)

mlflow.stop_run()
