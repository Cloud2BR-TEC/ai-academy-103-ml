# 04. Configure Microsoft Fabric for AI Workloads

## Source Used

- `msFabric-AI_integration/README.md`

## Steps

1. Register Fabric provider and confirm capacity.
2. Assign workspace to Fabric capacity.
3. Open Data Science notebook environment.
4. Install required dependencies (`synapseml`, `langchain`, `openai`).
5. Configure Azure OpenAI endpoint and key via secure secret approach.

## Configuration Snippet

```python
import os
os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "<from-secret-store>"
```
