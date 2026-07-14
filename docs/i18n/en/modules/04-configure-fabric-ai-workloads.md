# 04. Configure Microsoft Fabric for AI Workloads

## Goal

Leave Fabric ready to run the notebook flow and connect results into Azure ML operations.

## Step-by-step configuration

1. Register the Fabric provider in Azure subscription.
2. Create Fabric capacity with the right region/size.
3. Assign capacity to the target Fabric workspace.
4. Open Fabric notebook in Data Science experience.
5. Install pinned dependencies:

```python
%pip install synapseml==1.0.8 langchain==0.3.4 langchain_community==0.3.4 openai==1.53.0 langchain.openai==0.2.4
```

6. Configure Azure OpenAI settings via secure secret source.

```python
import os
os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "<from-secret-store>"
```

## Validation checks

- [ ] Notebook session can import SynapseML and LangChain
- [ ] Azure OpenAI client initializes successfully
- [ ] One test prompt executes successfully
- [ ] Output can be persisted for Azure ML downstream use

### Setup image

![Fabric capacity setup](../assets/img/azure-fabric-capacity-setup.png)
