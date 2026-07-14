# 03. Register and Deploy Endpoint in Azure ML

## Step 1 — Register the Model

```python
import joblib
joblib.dump(model, 'model.pkl')

from azureml.core import Workspace, Model
ws = Workspace.from_config()
Model.register(workspace=ws, model_path="model.pkl", model_name="my_model_RegressionModel")
```

![Register model](https://github.com/user-attachments/assets/a82ff03e-437c-41bc-85fa-8b9903384a5b)

## Step 2 — Create the Scoring Script

Use `score.py` with input validation, deterministic output, and logs.

![Scoring script](https://github.com/user-attachments/assets/cdc64857-3bde-4ec9-957d-5399d9447813)

## Step 3 — Create Environment File

Create `env.yaml` with all runtime dependencies.

![Environment file](https://github.com/user-attachments/assets/8e7c37a2-e32b-4630-8516-f95926c374c0)

## Step 4 — Deploy

```python
from azureml.core import Workspace
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice

ws = Workspace.from_config()
registered_model = Model(ws, name="my_model_RegressionModel")
env = Environment.from_pip_requirements(name="regression-env", file_path="requirements.txt")
inference_config = InferenceConfig(entry_script="score.py", environment=env)
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(
    workspace=ws,
    name="regression-model-service",
    models=[registered_model],
    inference_config=inference_config,
    deployment_config=deployment_config
)
service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")
```

![Deploy notebook](https://github.com/user-attachments/assets/1b3e5602-dc64-4c39-be72-ed1cbd74361e)
