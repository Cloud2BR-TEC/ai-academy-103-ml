# 08. End-to-End Validation and Handover

## Azure ML Path Validation

**1. Dataset registered**
- `employee_data` dataset registered as Tabular from [`sample_data.csv`](../reference/sample-data-csv.md) in Azure ML.

**2. Model trained and evaluated**

> Run predictions on the test set and print the key regression metrics to confirm the model is performing as expected before registration.

```python
predictions = model.predict(X_test)
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.2f}")
```

**3. Model registered**

> Serialize the trained model to disk and register it in the Azure ML workspace with a versioned name. This makes it available for deployment and reproducible retrieval by name.

```python
Model.register(workspace=ws, model_path="model.pkl", model_name="my_model_RegressionModel")
```

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

1. **`model.pkl`** — serialized regression model produced by `joblib.dump(model, 'model.pkl')` during training.
2. [`score.py`](../reference/score-py.md) — scoring script with input validation and logs.
3. **`env.yaml`** — environment definition file created during the deployment setup step.
4. [`fabric-llms-overview_sample.ipynb`](../reference/fabric-llms-notebook.md) — full Fabric LLM flow notebook with outputs.
5. [`0_ml-model-creation.ipynb`](../reference/ml-creation-notebook.md) — full model training notebook.
6. [`sample_data.csv`](../reference/sample-data-csv.md) — training dataset.
7. MLflow run ID and registered model version for traceability.
8. Scoring URI documented for consumer integration.
