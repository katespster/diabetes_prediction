
Run with Docker Build image:
docker build -t diabetes-api .

Compared Logistic Regression and Random Forest in MLflow
Logged parameters, metrics, reports, and trained models
Observed better accuracy / f1 for Random Forest and slightly better roc_auc for Logistic Regression

This project uses GitHub Actions to automatically run tests on every push and pull request.

Run tests locally:
python -m pytest -v


Allure test reporting

The project uses Allure Report for structured pytest reporting.

Run tests and collect Allure results:

python -m pytest --alluredir=allure-results --clean-alluredir

Open the report locally:

allure serve allure-results

The report includes:

preprocessing tests;
FastAPI endpoint tests;
ML prediction tests;
smoke and regression markers;
environment metadata.
