# fabric-llms-overview_sample.ipynb

Full Jupyter notebook demonstrating LLM integration in Microsoft Fabric using LangChain and SynapseML. Covers configuring Azure OpenAI, building LLMChain and LangchainTransformer pipelines, running distributed inference on Spark DataFrames, and logging ML models with MLflow.

[⬇ Download fabric-llms-overview_sample.ipynb](../assets/files/fabric-llms-overview_sample.ipynb){ .md-button .md-button--primary download="fabric-llms-overview_sample.ipynb" }

---

## Notebook Contents

| Cell | Description |
|---|---|
| 1 | Install `openai`, `langchain_community` |
| 2 | Set Azure OpenAI environment variables |
| 3 | Initialize `AzureChatOpenAI` instance |
| 4 | Create `PromptTemplate` for technology definitions |
| 5 | Build `LLMChain` linking prompt to LLM |
| 6 | Configure `LangchainTransformer` for Spark |
| 7 | Run transformer on a sample Spark DataFrame |
| 8 | Define PDF extraction and prompt generation functions |
| 9 | Build sequential chain for literature review agent |
| 10 | Train `RandomForestRegressor` and log with MLflow |
| 11 | Register and compare models with `MlflowClient` |

> Open the downloaded notebook in Microsoft Fabric → **Data Science** → **Notebooks**, or in VS Code with the Jupyter extension.
