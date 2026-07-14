# 05. Build One LLM Flow in Fabric Notebooks

## Create a Prompt Template

> Define the structure of the input that will be sent to the language model. The template accepts a `technology` variable and asks the model to produce a definition for it.

```python
from langchain.prompts import PromptTemplate

copy_prompt = PromptTemplate(
    input_variables=["technology"],
    template="Define the following word: {technology}",
)
```

## Set Up an LLMChain

> Link the prompt template to the Azure OpenAI instance (`llm`). The chain handles sending each filled-in prompt to the model and returning the generated text.

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=copy_prompt)
```

## Configure LangChain Transformer

> Wrap the chain in a SynapseML `LangchainTransformer` so it can run as a distributed Spark transformation. It reads from the `technology` column and writes generated definitions to a new `definition` column.

```python
from synapse.ml.cognitive.langchain import LangchainTransformer

transformer = (
    LangchainTransformer()
    .setInputCol("technology")
    .setOutputCol("definition")
    .setChain(chain)
    .setSubscriptionKey(os.environ["AZURE_OPENAI_API_KEY"])
    .setUrl(api_base)
)
```

## Run on a Spark DataFrame

> Create a small Spark DataFrame with sample technology names, apply the transformer, and display the results. Each row is processed in parallel by the distributed Spark engine.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("example").getOrCreate()
df = spark.createDataFrame(
    [(0, "docker"), (1, "spark"), (2, "python")],
    ["label", "technology"]
)
result = transformer.transform(df)
result.select("technology", "definition").show(truncate=False)
```

## Machine Learning Integration with Fabric

> Train a regression model and log it with MLflow directly inside Fabric. MLflow tracks parameters and registers the model artifact so it can be versioned and compared later.

```python
import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

X, y = make_regression(n_features=4, n_informative=2, random_state=0)
params = {"n_estimators": 3, "random_state": 42}
model = RandomForestRegressor(**params)
model.fit(X, y)

mlflow.log_params(params)
mlflow.sklearn.log_model(model, "model")
```
