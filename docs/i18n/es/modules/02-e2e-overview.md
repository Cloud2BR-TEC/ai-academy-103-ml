# Pipeline de Creación de Modelos (Ruta Azure ML Studio)

Usa este módulo si tu objetivo es: **crear un modelo en Azure ML Studio y dejarlo listo para despliegue**.

## Assets Fuente Utilizados

- `azML-modelcreation/data/sample_data.csv`
- `azML-modelcreation/src/0_ml-model-creation.ipynb`
- `azML-modelcreation/src/score.py`

## Paso 1 - Registrar Datos

1. Azure ML Studio -> **Data** -> **Create** -> From local files.
2. Subir `sample_data.csv` (o tu dataset).
3. Confirmar esquema y columna objetivo.
4. Guardar como data asset versionado.

## Paso 2 - Preparar Runtime de Notebook

1. Abrir notebook en compute instance.
2. Instalar/importar librerías requeridas.
3. Cargar dataset desde referencia de data asset, no desde ruta temporal local.

## Paso 3 - Entrenar un Modelo Base

1. Hacer split train/test.
2. Entrenar baseline (ej., RandomForestRegressor).
3. Registrar métricas: MAE, RMSE, R2.
4. Guardar artefacto (`model.pkl`).

## Paso 4 - Evaluar y Decidir

Ejemplo de puerta mínima de calidad:

- RMSE por debajo del umbral acordado
- Distribución de error estable
- Sin fugas de features evidentes

Si falla: iterar features/parámetros y reentrenar.

## Paso 5 - Registrar y Preparar Inferencia

1. Registrar modelo en model registry de Azure ML.
2. Reutilizar patrón de `score.py`:
   - validación de entrada
   - esquema de salida determinístico
   - manejo de errores + logs

## Paso 6 - Desplegar un Endpoint (Opcional recomendado)

1. Crear environment/inference config.
2. Desplegar a endpoint online.
3. Probar con payload de muestra.
4. Confirmar contrato de respuesta + latencia.

## Paso 7 - Checklist de Entrega

- [ ] Versión de dataset documentada
- [ ] Notebook de entrenamiento reproducible
- [ ] Métricas guardadas con metadata de run
- [ ] Modelo registrado
- [ ] Contrato de scoring probado
- [ ] Smoke test de endpoint aprobado
