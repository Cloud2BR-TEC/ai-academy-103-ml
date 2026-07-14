# 0_ml-model-creation.ipynb

Notebook completo de Jupyter para el flujo de creación de modelos en Azure ML. Cubre la carga del dataset registrado, preprocesamiento, entrenamiento de un `RandomForestRegressor`, evaluación con MAE/RMSE/R², registro del modelo y despliegue de un endpoint ACI.

[⬇ Descargar 0_ml-model-creation.ipynb](../assets/files/0_ml-model-creation.ipynb){ .md-button .md-button--primary download="0_ml-model-creation.ipynb" }

---

## Contenido del Notebook

| Celda | Descripción |
|---|---|
| 1 | Importar librerías y configurar MLClient |
| 2 | Cargar el asset `employee_data` desde el datastore de Azure ML |
| 3 | Preprocesamiento — codificar `Department`, eliminar `Name`, escalar features |
| 4 | División train/test y entrenamiento de `RandomForestRegressor` |
| 5 | Evaluación — MAE, MSE, RMSE, R² |
| 6 | Graficar distribución de errores y Predicho vs Real |
| 7 | `joblib.dump` y `Model.register` |
| 8 | Crear script de scoring, entorno y despliegue ACI |
| 9 | Smoke-test del endpoint desplegado |

> Abre el notebook descargado en Azure ML Studio → **Notebooks** o en VS Code con la extensión de Jupyter.
