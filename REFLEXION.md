# Reflection

## Lab 3 — Experiment tracking with MLflow

**1. Which model is the best? Can you prove it?**
Yes. Every trial is recorded as an MLflow run with its parameters and metrics. Sorting the runs by f1 (in the UI or with `read_runs.py`) shows that trial-1 and trial-3 are tied (accuracy 0.99, f1 0.889), while trial-2 is clearly weaker (f1 0.333) even though its accuracy (0.96) looks good. With equal results we keep trial-3, the simpler model that does not use the salary column. The proof is dated and kept in MLflow.

**2. Can you give back exactly the model of Trial 1?**
Yes. Each run keeps its own model as an artifact, so nothing is overwritten. `mlflow.sklearn.load_model("runs:/<run_id>/model")` reloads the exact model of a given run without retraining (see `reload_model.py`).

**3. Can a classmate reproduce your work on another machine?**
Partly. The code is on GitHub, and each run records its parameters and the library versions (`requirements.txt`, `python_env.yaml`). But `mlruns/` and `mlflow.db` stay on my machine, so a classmate has to rerun the scripts to get their own runs, and getting an identical environment is still not guaranteed. Docker (Labs 5–6) will solve that.

## Lab 4 — Model Registry and FastAPI

**1. How do we officially designate the model to use?**
With the `champion` alias in the MLflow Model Registry. `models:/churn-model@champion` always points to the chosen version, so nobody has to remember a run id. Moving the alias to another version takes one line (`move_alias.py`).

**2. How can an application use the model?**
Through the API (`app.py`). Any application sends a `POST /predict` request with the customer as JSON, and the response contains the prediction (`churn` 0 or 1 plus a message). The application never loads the model itself, and invalid inputs are rejected with a 422 error.

**3. What happens when we change champion?**
We move the alias to the new version and restart the API, which loads the champion at startup. Neither the API code nor the client applications change, because they only refer to `@champion` and `/predict`.
