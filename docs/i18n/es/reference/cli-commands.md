# Comandos CLI

Una referencia de los comandos usados a lo largo del curso, agrupados por etapa, con una breve nota sobre
qué hace cada uno y cuándo usarlo.

## Configuración del entorno (conda + pip)

```console
conda env create --name aml-env --file environment.yml
conda activate aml-env
pip install -r requirements.txt
```

- `conda env create` construye un entorno aislado a partir de una especificación fijada (dependencias del sistema + Python).
- `conda activate` cambia tu shell a ese entorno.
- `pip install -r` agrega paquetes fijados de Python puro encima.

Validación:

```console
conda env list            # confirm which environments exist / which is active
python --version          # confirm the interpreter version
pip show scikit-learn     # confirm a key package and its version
python -m pip install ... # always prefer this over bare pip to target the right interpreter
python -m ipykernel install --user --name aml-env --display-name "AML Env"
```

## CLI de Azure ML (v2)

```bash
# Authenticate and set defaults
az login
az account set --subscription "<sub-id>"
az configure --defaults group=<rg> workspace=<ws>

# Data and environments
az ml data create --name fraud-train --version 1 --path ./data --type uri_folder
az ml environment create --name fraud-infer --version 2 --file env.yml

# Submit a training/AutoML job
az ml job create --file job.yml
az ml job list --query "[].{name:name,status:status}" -o table
az ml job stream --name <run-id>          # live logs

# Register and inspect models
az ml model create --name fraud-model --version 3 --path ./model --type mlflow_model
az ml model list --name fraud-model -o table
```

## Despliegue (endpoint en línea administrado)

```bash
az ml online-endpoint create --name fraud-endpoint
az ml online-deployment create --file deployment.yml --all-traffic

# Canary: route a small slice of traffic to a new deployment
az ml online-endpoint update --name fraud-endpoint --traffic "blue=90 green=10"

# Test and inspect
az ml online-endpoint invoke --name fraud-endpoint --request-file sample.json
az ml online-deployment get-logs --name green --endpoint-name fraud-endpoint
```

## Servicio y depuración con Kubernetes

```bash
kubectl version --client
kubectl get pods -n <namespace>                 # high-level pod health
kubectl describe pod <pod-name> -n <namespace>  # events: scheduling, image pull, probes
kubectl logs <pod-name> -n <namespace>          # current container logs
kubectl logs <pod-name> -n <namespace> --previous  # crashed container logs (CrashLoopBackOff)
kubectl get events --sort-by=.lastTimestamp     # recent cluster events
kubectl get svc                                 # services and virtual IPs
kubectl get endpoints <service-name>            # ready pod IPs behind a service (empty = no targets)
kubectl port-forward svc/<service-name> 8080:80 # test the service directly
```

Kubernetes local para desarrollo:

```bash
kind create cluster        # Kubernetes in Docker, fast disposable clusters
minikube start             # single-node local cluster
```
