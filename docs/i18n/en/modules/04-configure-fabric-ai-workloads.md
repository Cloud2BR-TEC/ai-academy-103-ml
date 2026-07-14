# 04. Configure Microsoft Fabric for AI Workloads

## Install Required Libraries

1. Open Fabric portal and sign in.
2. Go to **Data Science** and create a new cluster.

    ![Create cluster](https://github.com/user-attachments/assets/1763b8a0-e3ff-4ae7-adc9-413f0eb454f3)

    ![Cluster config](https://github.com/user-attachments/assets/6cf15794-27cb-4ee9-af66-ac236b14de1e)

3. Install SynapseML on the cluster.

    ![Install SynapseML](https://github.com/user-attachments/assets/a2243d3a-17f2-456b-829c-06c22f8ab7b7)

4. Install LangChain and dependencies.

```python
%pip install openai langchain_community
```

Or use a `.yml` environment file:

```yml
dependencies:
  - pip:
      - synapseml==1.0.8
      - langchain==0.3.4
      - langchain_community==0.3.4
      - openai==1.53.0
      - langchain.openai==0.2.4
```

![Environment config](https://github.com/user-attachments/assets/b61d180c-d7e5-4aec-a4b9-8133b3a92250)

## Configure Azure OpenAI

Set the required environment variables for your Azure OpenAI deployment. These values are found in the Azure Portal under your OpenAI resource → **Keys and Endpoint**.

```python
import os
os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
```

Create an `AzureChatOpenAI` instance that the LangChain chain will use to call your deployed model. `temperature` controls response randomness; `top_p` limits the token probability distribution.

```python
from langchain_openai import AzureChatOpenAI

api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.7,
    verbose=True,
    top_p=0.9
)
```
