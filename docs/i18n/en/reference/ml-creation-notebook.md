# 0_ml-model-creation.ipynb

Full Jupyter notebook for the Azure ML model creation workflow. Covers loading the registered dataset, preprocessing, training a `RandomForestRegressor`, evaluating with MAE/RMSE/R², registering the model, and deploying an ACI endpoint.

[⬇ Download 0_ml-model-creation.ipynb](../assets/files/0_ml-model-creation.ipynb){ .md-button .md-button--primary download="0_ml-model-creation.ipynb" }

---

## Notebook Contents

| Cell | Description |
|---|---|
| 1 | Import libraries and configure MLClient |
| 2 | Load `employee_data` asset from Azure ML datastore |
| 3 | Preprocessing — encode `Department`, drop `Name`, scale features |
| 4 | Train/test split and `RandomForestRegressor` training |
| 5 | Evaluate — MAE, MSE, RMSE, R² |
| 6 | Plot prediction errors and Predicted vs Actual |
| 7 | `joblib.dump` and `Model.register` |
| 8 | Create scoring script, environment, and ACI deployment |
| 9 | Smoke-test the deployed endpoint |

> Open the downloaded notebook in Azure ML Studio → **Notebooks** or in VS Code with the Jupyter extension.
