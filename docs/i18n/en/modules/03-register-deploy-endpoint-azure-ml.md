# 03. Register and Deploy Endpoint in Azure ML

## Source Used

- `azML-modelcreation/src/score.py`

## Steps

1. Register model in Azure ML model registry.
2. Prepare scoring script based on `score.py`.
3. Create environment/inference configuration.
4. Deploy online endpoint.
5. Test endpoint with sample request payload.
6. Validate response schema and latency.

## Minimum Acceptance

- Endpoint returns valid prediction response.
- Errors are handled with clear status messages.
- Logs are visible for troubleshooting.
