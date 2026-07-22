# MLOps Learning Project — Step 1: Train & Save a Model

Follow these steps yourself in the terminal, in this folder
(`/Users/ravdsun/Documents/MLOps`). Type the commands rather than copy-pasting
in bulk — it sticks better when your fingers do it.

## 0. Project layout

Create the folders we'll use:

```bash
mkdir src models data
```

- `src/` — your Python scripts
- `models/` — saved model files (the "artifacts")
- `data/` — datasets (empty for now, we'll use a built-in one first)

## 1. Git init

```bash
git init
```

Why: you already know git from DevOps. Every step from here on (code, later
Dockerfile, later CI/CD config) should be tracked from the start.

## 2. Virtual environment

Python projects isolate their dependencies per-project instead of installing
globally — this avoids version conflicts between projects.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You'll know it worked because your shell prompt gets a `(.venv)` prefix.
You need to run the `source` command in every new terminal session/tab you use
for this project.

## 3. Dependencies

Create a file named `requirements.txt` in the project root with this content:

```
pandas
numpy
scikit-learn
joblib
```

- `pandas`/`numpy` — data handling
- `scikit-learn` — the ML library (models, train/test split, metrics)
- `joblib` — saves/loads a trained model to/from disk

Install them:

```bash
pip install -r requirements.txt
```

## 4. Write `src/train.py`

Create this file yourself, line by line, and read the comments as you type —
they explain what each block does and why it's there in an ML pipeline
(not just what the Python syntax means).

```python
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
```

## 5. Run it

```bash
python src/train.py
```

Expected: it prints an accuracy (should be ~0.95+) and a classification
report, and creates `models/model.joblib`.

## 6. Commit your work

```bash
git add .
git commit -m "Step 1: train and save a baseline model"
```

## What's next

Once this runs cleanly and you understand every line (ask me about any of
it!), the next milestone is **experiment tracking with MLflow** — logging
params/metrics instead of just printing them. Come back and say "let's do
step 3" when you're ready.
