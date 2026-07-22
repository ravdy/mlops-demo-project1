import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Built-in dataset so you don't need to download anything yet.
# Later we'll swap this for a real CSV in data/.
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# Never evaluate a model on the data it was trained on — split it first.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# random_state=42 makes the split/training reproducible across runs.
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Check performance on data the model has never seen.
preds = model.predict(X_test)
print(f"accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds))

# This file is the "artifact" — the thing you'll later serve via an API.
joblib.dump(model, "models/model.joblib")
print("saved model to models/model.joblib")