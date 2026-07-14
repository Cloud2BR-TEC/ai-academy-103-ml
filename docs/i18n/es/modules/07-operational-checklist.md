# 07. Checklist Operativo (Seguridad, Costo, Confiabilidad)

## Seguridad

- Almacenar todas las API keys en variables de entorno — nunca hardcodear en notebooks o scripts.

```python
# Correcto: leer desde el entorno
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
```

- Usar Azure Key Vault o gestión de secretos de Fabric en producción.
- Fijar todas las versiones de dependencias explícitamente.

```yml
dependencies:
  - pip:
      - synapseml==1.0.8
      - langchain==0.3.4
      - openai==1.53.0
```

- Aplicar RBAC de mínimo privilegio al workspace de Azure ML y a la capacidad de Fabric.
- No escribir API keys en bruto ni salidas del modelo con PII en logs.

## Costo

- **Pausar la capacidad de Fabric** en Azure Portal cuando no esté en uso.

![Pausar cómputo Fabric](https://github.com/user-attachments/assets/117d6902-cc8b-45b8-a104-9b54180565f2)

- Configurar auto-stop en las compute instances de Azure ML.
- Configurar alertas de presupuesto de tokens en los despliegues de Azure OpenAI.
- Usar `AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)` para endpoints de dev/test.

## Confiabilidad

- Versionar todos los datasets (`employee_data`, versión `"1"`) antes del entrenamiento.
- Registrar cada modelo entrenado con nombre y etiquetas consistentes.

```python
Model.register(workspace=ws, model_path="model.pkl", model_name="my_model_RegressionModel")
```

- Usar `mlflow.log_params(params)` y `mlflow.sklearn.log_model(...)` para reproducibilidad.
- Ejecutar `%pip show synapseml` después de la configuración del clúster para confirmar la instalación.
- Hacer smoke test del endpoint desplegado antes de la entrega.

```python
service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")
```
