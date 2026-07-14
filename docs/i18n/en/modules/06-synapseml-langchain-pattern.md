# 06. SynapseML and LangChain Integration Pattern

## Tools Overview

| **Tool** | **Description** |
|---|---|
| **LangChain** | Framework for developing applications powered by language models — multi-step processing, preprocessing text, applying LLMs, and postprocessing results. |
| **SynapseML** | Open-source library for massively scalable ML pipelines — applies large language models at scale using distributed computing on Spark. |

## Large Scale Literature Review with LangChain

Automate extraction and summarization of academic papers using an agent that loads an online PDF and generates a structured prompt.

### Step 1 — Define Extraction and Prompt Functions

Two helper functions drive the pipeline. `paper_content_extraction` fetches an arXiv PDF via its URL and returns the first two pages of text. `prompt_generation` takes that extracted text and builds a structured prompt asking the model to identify the title, authors, and summary — then search for three recent papers by the first author.

```python
from langchain.document_loaders import OnlinePDFLoader

def paper_content_extraction(inputs: dict) -> dict:
    arxiv_link = inputs["arxiv_link"]
    loader = OnlinePDFLoader(arxiv_link)
    pages = loader.load_and_split()
    return {"paper_content": pages[0].page_content + pages[1].page_content}

def prompt_generation(inputs: dict) -> dict:
    output = inputs["Output"]
    prompt = (
        "find the paper title, author, summary in the paper description below, output them. "
        "After that, Use websearch to find out 3 recent papers of the first author in the author section below "
        "(first author is the first name separated by comma) and list the paper titles in bullet points: "
        "<Paper Description Start>\n" + output + "<Paper Description End>."
    )
    return {"prompt": prompt}
```

### Step 2 — Build Sequential Chain

Wrap the extraction function in a `TransformChain` so LangChain can treat it as a pipeline stage. The `paper_summarizer_template` defines the system instruction passed to the model at runtime.

```python
from langchain.chains import TransformChain, SimpleSequentialChain

paper_content_extraction_chain = TransformChain(
    input_variables=["arxiv_link"],
    output_variables=["paper_content"],
    transform=paper_content_extraction,
    verbose=False,
)

paper_summarizer_template = """
You are a paper summarizer, given the paper content, it is your job to summarize the paper into a short summary,
and extract authors and paper title from the paper content.
"""
```

## MLflow Model Registration and Comparison in Fabric

### Train and Register a Model with MLflow

Start an MLflow run, train the model, infer the input/output signature, then log parameters and register the model artifact with descriptive tags. The run ID and version number are printed for traceability.

```python
import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
params = {"n_estimators": 3, "random_state": 42}
model_tags = {
    "project_name": "grocery-forecasting",
    "store_dept": "produce",
    "team": "stores-ml",
    "project_quarter": "Q3-2023"
}

with mlflow.start_run() as run:
    model = RandomForestRegressor(**params).fit(X, y)
    signature = infer_signature(X, model.predict(X))
    mlflow.log_params(params)
    mlflow.sklearn.log_model(model, artifact_path="sklearn-model", signature=signature)
    model_uri = f"runs:/{run.info.run_id}/sklearn-model"
    model_version = mlflow.register_model(model_uri, "RandomForestRegressionModel", tags=model_tags)
    print(f"Model Name: {model_version.name}")
    print(f"Model Version: {model_version.version}")
```

### Compare and Filter Registered Models

Use `MlflowClient` to list all registered models in the workspace. This is useful for comparing multiple training runs and selecting the best candidate for promotion to production.

```python
from pprint import pprint
from mlflow.tracking import MlflowClient

client = MlflowClient()
for rm in client.search_registered_models():
    pprint(dict(rm), indent=4)
```
