# MLOps Learning Project — Step 1: Train & Save a Model

Follow these steps yourself in the terminal, in this folder
(`/Users/ravdsun/Documents/MLOps`). Type the commands rather than copy-pasting
in bulk — it sticks better when your fingers do it.

## 0. Project layout

Create the folders we'll use:

```bash
mkdir src models data
```

`mkdir` just creates empty folders — nothing Python- or ML-specific here,
same command you've always used.

- `src/` — your Python scripts
- `models/` — saved model files (the "artifacts")
- `data/` — datasets (empty for now, we'll use a built-in one first)

## 1. Git init

```bash
git init
```

Turns this folder into a git repository (creates a hidden `.git/` folder that
tracks history). Same command, same meaning as in any DevOps repo — nothing
ML-specific about it. Every step from here on (code, later Dockerfile, later
CI/CD config) should be tracked from the start.

## 2. Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

This is **not** like Docker, even though "isolated environment" sounds
similar — there's no separate filesystem or process. Here's what actually
happens:

- `python3 -m venv .venv` creates a folder named `.venv/` containing a private
  copy of the Python interpreter plus a `bin/` directory with `python`, `pip`,
  and an `activate` script inside it. This step just creates files — nothing
  runs or activates yet.
- `source .venv/bin/activate` runs that script **in your current shell**, and
  the only thing it does is change your shell's `PATH` variable so that typing
  `python` or `pip` now resolves to the copies inside `.venv/bin/` instead of
  your system-wide Python. You are still in the same folder, same shell, same
  everything else — nothing is isolated except *which python/pip binary gets
  used* and *where pip installs packages* (into `.venv/lib/...` instead of
  system-wide).
- Why `source` and not just running the script? Environment variable changes
  only stick in the shell that sets them. Running the script normally would
  execute it in a disposable subshell, and the `PATH` change would vanish the
  instant it finished. `source` runs it in your actual terminal session so the
  change persists.
- You'll know it worked because your prompt gets a `(.venv)` prefix.
- To undo it later: run `deactivate` — restores your original `PATH`.
- You need to re-run the `source` command in every *new* terminal tab/window
  you open for this project (activation doesn't persist across sessions).

Why bother at all: without this, `pip install` would install packages
system-wide, and different projects on your machine could end up needing
conflicting versions of the same package. The venv keeps this project's
packages in `.venv/lib/`, separate from every other project's.

## 3. Dependencies

Create a file named `requirements.txt` in the project root with this content
(a plain text file, one package name per line — not a script, doesn't need to
be executable):

```
pandas
numpy
scikit-learn
joblib
```

- `pandas`/`numpy` — data handling
- `scikit-learn` — the ML library (models, train/test split, metrics)
- `joblib` — saves/loads a trained model to/from disk

Install them (make sure your `(.venv)` prompt prefix is showing first, so
these install into the venv and not system-wide):

```bash
pip install -r requirements.txt
```

`-r requirements.txt` tells `pip` to read that file and install every package
listed in it, rather than naming packages one at a time on the command line.

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

Make sure `(.venv)` is still showing in your prompt (re-activate with the
`source` command from step 2 if you opened a new terminal), then:

```bash
python src/train.py
```

This just tells the Python interpreter to execute that file top to bottom —
same as running any script. Since you activated the venv, this runs
`.venv/bin/python`, which has access to the packages you installed in step 3.

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
