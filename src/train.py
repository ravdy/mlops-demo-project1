import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_estimators = 200
random_state = 50

mlflow.set_experiment("breast-cancer-classifier")
# Everything inside this block is grouped into one trackable "run".
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"accuracy: {accuracy:.4f}")
    print(classification_report(y_test, preds))

    # Log inputs (params) and output (metric) so you can compare runs later.
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("accuracy", accuracy)

    # MLflow's own model logging replaces the joblib.dump from before -
    # it saves the model plus metadata (library version, input/output shape)
    # in a self-describing folder instead of a bare file.
    mlflow.sklearn.log_model(model, "model")
