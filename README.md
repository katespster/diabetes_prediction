Run with Docker Build image:
docker build -t diabetes-api .


Compared Logistic Regression and Random Forest in MLflow
Logged parameters, metrics, reports, and trained models
Observed better accuracy / f1 for Random Forest and slightly better roc_auc for Logistic Regression

Можно добавить короткий раздел:

## CI

This project uses GitHub Actions to automatically run tests on every push and pull request.

Run tests locally:

```bash
python -m pytest -v