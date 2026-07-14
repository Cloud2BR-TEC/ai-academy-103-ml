# 07. Operational Checklist (Security, Cost, Reliability)

## Security

- Store all API keys in environment variables — never hardcode them in notebooks or scripts.

```python
# Correct: read from environment
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
```

- Use Azure Key Vault or Fabric secrets management for production.
- Pin all dependency versions explicitly.

```yml
dependencies:
  - pip:
      - synapseml==1.0.8
      - langchain==0.3.4
      - openai==1.53.0
```

- Apply least-privilege RBAC to the Azure ML workspace and Fabric capacity.
- Do not write raw API keys or model output with PII into logs.

## Cost

- **Pause Fabric capacity** in Azure Portal when not in use.

![Pause Fabric compute](https://github.com/user-attachments/assets/117d6902-cc8b-45b8-a104-9b54180565f2)

- Set compute auto-stop on Azure ML compute instances.
- Configure token budget alerts on Azure OpenAI deployments.
- Use `AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)` for dev/test endpoints to minimize cost.

## Reliability

- Version all datasets (`employee_data`, version `"1"`) before training.
- Register every trained model with a name and consistent tags.

```python
Model.register(workspace=ws, model_path="model.pkl", model_name="my_model_RegressionModel")
```

- Use `mlflow.log_params(params)` and `mlflow.sklearn.log_model(...)` to make every run reproducible.
- Run `%pip show synapseml` after cluster setup to confirm library installation.
- Smoke-test deployed endpoints with a real payload before handover.

```python
service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")
```
