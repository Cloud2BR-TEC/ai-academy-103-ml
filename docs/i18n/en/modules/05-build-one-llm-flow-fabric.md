# 05. Build One LLM Flow in Fabric Notebooks

## Create a Prompt Template

```python
from langchain.prompts import PromptTemplate

copy_prompt = PromptTemplate(
    input_variables=["technology"],
    template="Define the following word: {technology}",
)
```

![Prompt template](https://github.com/user-attachments/assets/f4a3dea8-d743-46e0-a6e9-279aae457bc8)

## Set Up an LLMChain

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=copy_prompt)
```

![LLMChain setup](https://github.com/user-attachments/assets/30a74226-7a02-4c81-a4b1-4039eb43fa9c)

## Configure LangChain Transformer

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

![Transformer config](https://github.com/user-attachments/assets/f7d6480a-b75e-449e-808d-ad5a51974af9)

## Run on a Spark DataFrame

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

![DataFrame result](https://github.com/user-attachments/assets/a2a8e208-6f1d-4cb0-9944-0d2457106b49)

## Machine Learning Integration with Fabric

Train and log models with MLflow inside Fabric:

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

![ML integration Fabric](https://github.com/user-attachments/assets/a6eebe61-bfde-48ce-88e7-9bd5dfb6d00a)
