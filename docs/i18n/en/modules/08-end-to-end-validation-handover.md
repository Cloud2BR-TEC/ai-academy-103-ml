# 08. End-to-End Validation and Handover

## Azure ML Path Validation

**1. Dataset registered**
- `employee_data` dataset registered as Tabular from `sample_data.csv` in Azure ML.

**2. Model trained and evaluated**

```python
predictions = model.predict(X_test)
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.2f}")
```

![Evaluation metrics](https://github.com/user-attachments/assets/6aa19680-cadb-4fe4-a419-a626942e15f9)

**3. Model registered**

```python
Model.register(workspace=ws, model_path="model.pkl", model_name="my_model_RegressionModel")
```

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/a82ff03e-437c-41bc-85fa-8b9903384a5b" type="video/mp4">
</video>

**4. Endpoint deployed and responding**

```python
service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")
```

## Fabric LLM Path Validation

**5. Fabric capacity active and cluster created**

![Fabric capacity running in Azure Portal](https://github.com/user-attachments/assets/117d6902-cc8b-45b8-a104-9b54180565f2)

**6. SynapseML and LangChain installed**

```python
%pip install openai langchain_community
%pip show synapseml
```

**7. Azure OpenAI connected and responding**

```python
ai_msg = llm.invoke(messages)
print(ai_msg)
```

**8. LangChain Transformer producing definitions**

```python
result.select("technology", "definition").show(truncate=False)
```

**9. MLflow model registered in Fabric**

```python
print(f"Model Name: {model_version.name}")
print(f"Model Version: {model_version.version}")
```

## Handover Package

1. `model.pkl` — serialized regression model from Azure ML.
2. `score.py` — scoring script with input validation and logs.
3. `env.yaml` — pinned environment definition.
4. Fabric notebook (`fabric-llms-overview_sample.ipynb`) with full LLM flow output.
5. MLflow run ID and registered model version for traceability.
6. Scoring URI documented for consumer integration.
