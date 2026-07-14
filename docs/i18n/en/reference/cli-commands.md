
# CLI Commands

A reference of the commands used across the course, grouped by stage, with a short note on
what each does and when to use it.

## Environment setup (conda + pip)

```console
conda env create --name aml-env --file environment.yml
conda activate aml-env
pip install -r requirements.txt
```

- `conda env create` builds an isolated environment from a pinned spec (system + Python deps).
- `conda activate` switches your shell into that environment.
- `pip install -r` adds pinned pure-Python packages on top.

Validation:

```console
conda env list            # confirm which environments exist / which is active
python --version          # confirm the interpreter version
pip show scikit-learn     # confirm a key package and its version
python -m pip install ... # always prefer this over bare pip to target the right interpreter
python -m ipykernel install --user --name aml-env --display-name "AML Env"
```

## Azure ML CLI (v2)

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

## Deployment (managed online endpoint)

```bash
az ml online-endpoint create --name fraud-endpoint
az ml online-deployment create --file deployment.yml --all-traffic

# Canary: route a small slice of traffic to a new deployment
az ml online-endpoint update --name fraud-endpoint --traffic "blue=90 green=10"

# Test and inspect
az ml online-endpoint invoke --name fraud-endpoint --request-file sample.json
az ml online-deployment get-logs --name green --endpoint-name fraud-endpoint
```

## Kubernetes serving and debugging

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

Local Kubernetes for development:

```bash
kind create cluster        # Kubernetes in Docker, fast disposable clusters
minikube start             # single-node local cluster
```

