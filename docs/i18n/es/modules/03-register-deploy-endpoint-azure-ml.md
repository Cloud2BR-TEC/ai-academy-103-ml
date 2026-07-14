# 03. Registrar y Desplegar Endpoint en Azure ML

## Paso 1 — Registrar el Modelo

```python
import joblib
joblib.dump(model, 'model.pkl')

from azureml.core import Workspace, Model
ws = Workspace.from_config()
Model.register(workspace=ws, model_path="model.pkl", model_name="my_model_RegressionModel")
```

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/a82ff03e-437c-41bc-85fa-8b9903384a5b" type="video/mp4">
</video>

## Paso 2 — Crear el Script de Scoring

Usar [`score.py`](https://github.com/Cloud2BR-MSFTLearningHub/Azure-ML-Overview/blob/main/azML-modelcreation/src/score.py) con validación de entrada, salida determinística y logs.

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/cdc64857-3bde-4ec9-957d-5399d9447813" type="video/mp4">
</video>

## Paso 3 — Crear el Archivo de Entorno

Crear `env.yaml` con todas las dependencias del runtime.

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/8e7c37a2-e32b-4630-8516-f95926c374c0" type="video/mp4">
</video>

## Paso 4 — Desplegar

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

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/1b3e5602-dc64-4c39-be72-ed1cbd74361e" type="video/mp4">
</video>
