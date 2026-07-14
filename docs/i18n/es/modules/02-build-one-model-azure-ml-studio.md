# 02. Crear un Modelo en Azure ML Studio

## Fuente Utilizada

- `azML-modelcreation/src/0_ml-model-creation.ipynb`
- `azML-modelcreation/data/sample_data.csv`

## Pasos

1. Subir y registrar dataset en la sección Data de Azure ML Studio.
2. Abrir notebook en compute instance.
3. Cargar referencia del data asset y validar esquema.
4. Entrenar modelo baseline (como en notebook fuente).
5. Evaluar con MAE, RMSE y R2.
6. Guardar artefacto del modelo (`model.pkl`).

## Salida

- Un artefacto de modelo entrenado
- Un reporte de métricas
- Una ejecución reproducible de notebook
