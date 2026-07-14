# 08. Validación End-to-End y Entrega

## Validación de la Ruta Azure ML

**1. Dataset registrado**
- Dataset `employee_data` registrado como Tabular desde `sample_data.csv` en Azure ML.

**2. Modelo entrenado y evaluado**

Ejecuta predicciones sobre el conjunto de prueba e imprime las métricas clave de regresión para confirmar que el modelo rinde correctamente antes del registro.

```python
predictions = model.predict(X_test)
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.2f}")
```

**3. Modelo registrado**

Serializa el modelo entrenado en disco y regrístralo en el workspace de Azure ML con un nombre versionado. Esto lo hace disponible para despliegue y recuperable de forma reproducible por nombre.

```python
Model.register(workspace=ws, model_path="model.pkl", model_name="my_model_RegressionModel")
```

**4. Endpoint desplegado y respondiendo**

```python
service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")
```

## Validación de la Ruta LLM en Fabric

**5. Capacidad de Fabric activa y clúster creado**

![Capacidad Fabric activa en Azure Portal](https://github.com/user-attachments/assets/117d6902-cc8b-45b8-a104-9b54180565f2)

**6. SynapseML y LangChain instalados**

```python
%pip install openai langchain_community
%pip show synapseml
```

**7. Azure OpenAI conectado y respondiendo**

```python
ai_msg = llm.invoke(messages)
print(ai_msg)
```

**8. LangChain Transformer generando definiciones**

```python
result.select("technology", "definition").show(truncate=False)
```

**9. Modelo MLflow registrado en Fabric**

```python
print(f"Model Name: {model_version.name}")
print(f"Model Version: {model_version.version}")
```

## Paquete de Entrega

1. `model.pkl` — modelo de regresión serializado desde Azure ML.
2. `score.py` — script de scoring con validación de entrada y logs.
3. `env.yaml` — definición de entorno con versiones fijadas.
4. Notebook de Fabric (`fabric-llms-overview_sample.ipynb`) con salida completa del flujo LLM.
5. Run ID de MLflow y versión del modelo registrado para trazabilidad.
6. Scoring URI documentado para integración por parte del consumidor.
