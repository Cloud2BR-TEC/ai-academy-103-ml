# Fabric and Azure ML Integration (Advanced)

This module uses the implementation flow from your Fabric source material and turns it into a production-ready integration playbook.

## Source Material Used

- `msFabric-AI_integration/README.md`
- `msFabric-AI_integration/src/fabric-llms-overview_sample.ipynb`

## 1. Environment and Capacity Setup

### 1.1 Register provider and create Fabric capacity

Before running AI workloads, validate that Fabric capacity exists and is assigned to the correct workspace.

Checklist:

- Microsoft Fabric resource provider registered
- Fabric capacity created in the intended region
- Power BI/Fabric workspace linked to that capacity
- Cost pause/resume process documented

### 1.2 Prepare runtime dependencies

In Fabric notebooks, install only the packages you need and pin versions for repeatability.

```python
%pip install synapseml==1.0.8 langchain==0.3.4 langchain_community==0.3.4 openai==1.53.0 langchain.openai==0.2.4
```

## 2. Azure OpenAI Configuration Pattern

Use environment variables or secret scopes, never hardcoded secrets in notebook cells.

```python
import os

os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "<from-secret-store>"
```

Initialize the model client:

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.2,
    top_p=0.9,
    verbose=True,
)
```

## 3. LangChain + SynapseML Execution Flow

### 3.1 Prompt and chain definition

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["technology"],
    template="Define the following technology term for an ML engineering audience: {technology}",
)

chain = LLMChain(llm=llm, prompt=prompt)
```

### 3.2 Distributed transformer execution

```python
from synapse.ml.cognitive.langchain import LangchainTransformer

transformer = (
    LangchainTransformer()
    .setInputCol("technology")
    .setOutputCol("definition")
    .setChain(chain)
    .setSubscriptionKey(os.environ["AZURE_OPENAI_API_KEY"])
    .setUrl(os.environ["AZURE_OPENAI_ENDPOINT"])
)
```

### 3.3 Minimal data frame test

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("fabric-llm-demo").getOrCreate()

df = spark.createDataFrame(
    [(0, "docker"), (1, "spark"), (2, "feature store")],
    ["id", "technology"],
)

result_df = transformer.transform(df)
result_df.select("id", "technology", "definition").show(truncate=False)
```

## 4. Operational Bridge to Azure ML

Fabric should not be the final stop. Use it as upstream processing for governed Azure ML operations.

Recommended handoff pattern:

1. Persist transformed/feature-ready output with version labels.
2. Trigger Azure ML training or batch scoring job.
3. Register model + metrics in Azure ML model registry.
4. Promote only validated models to deployment stages.
5. Monitor endpoint quality and trigger retraining when drift appears.

## 5. Security, Cost, and Reliability Controls

### Security

- Keep API keys in managed secrets.
- Apply least-privilege workspace and service principal roles.
- Avoid logging raw sensitive payloads.

### Cost

- Cap notebook cluster size/time windows.
- Track token usage and define budget alarms.
- Pause Fabric capacity outside scheduled windows.

### Reliability

- Pin package versions.
- Validate input schema before LLM calls.
- Store run metadata (time, prompt version, model deployment, output location).

## 6. Implementation Checklist

- [ ] Fabric capacity assigned and validated
- [ ] SynapseML/LangChain dependencies pinned
- [ ] Azure OpenAI secrets externalized
- [ ] LLM transform tested on sample and batch data
- [ ] Handoff to Azure ML training/scoring in place
- [ ] Monitoring and cost guardrails active

## 7. Next Step

Continue to `07. Terraform for Azure ML` and `08. Terraform for Fabric` to operationalize this integration as reproducible infrastructure.
