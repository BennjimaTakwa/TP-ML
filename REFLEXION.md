# Reflection

## Lab 3 — Experiment tracking with MLflow

**1. Which model is the best? Can you prove it?**
Yes. Every trial is recorded as an MLflow run with its parameters and metrics. Sorting the runs by f1 (in the UI or with `read_runs.py`) shows that trial-1 and trial-3 are tied (accuracy 0.99, f1 0.889), while trial-2 is clearly weaker (f1 0.333) even though its accuracy (0.96) looks good. With equal results we keep trial-3, the simpler model that does not use the salary column. The proof is dated and kept in MLflow.

**2. Can you give back exactly the model of Trial 1?**
Yes. Each run keeps its own model as an artifact, so nothing is overwritten. `mlflow.sklearn.load_model("runs:/<run_id>/model")` reloads the exact model of a given run without retraining (see `reload_model.py`).

**3. Can a classmate reproduce your work on another machine?**
Partly. The code is on GitHub, and each run records its parameters and the library versions (`requirements.txt`, `python_env.yaml`). But `mlruns/` and `mlflow.db` stay on my machine, so a classmate has to rerun the scripts to get their own runs, and getting an identical environment is still not guaranteed. Docker (Labs 5–6) will solve that.
