# 05. Crear un Flujo LLM en Notebooks de Fabric

## Crear una Plantilla de Prompt

Define la estructura del input que se enviará al modelo de lenguaje. La plantilla acepta una variable `technology` y le pide al modelo que genere una definición.

```python
from langchain.prompts import PromptTemplate

copy_prompt = PromptTemplate(
    input_variables=["technology"],
    template="Define the following word: {technology}",
)
```

## Configurar un LLMChain

Vincula la plantilla de prompt a la instancia de Azure OpenAI (`llm`). La cadena se encarga de enviar cada prompt completo al modelo y devolver el texto generado.

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=copy_prompt)
```

## Configurar el LangChain Transformer

Encapsula la cadena en un `LangchainTransformer` de SynapseML para que pueda ejecutarse como transformación distribuida de Spark. Lee de la columna `technology` y escribe las definiciones generadas en una nueva columna `definition`.

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

## Ejecutar sobre un Spark DataFrame

Crea un DataFrame de Spark con nombres de tecnologías de ejemplo, aplica el transformer y muestra los resultados. Cada fila es procesada en paralelo por el motor distribuido de Spark.

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

## Integración de ML con Fabric

Entrena un modelo de regresión y lo registra con MLflow directamente dentro de Fabric. MLflow rastrea los parámetros y registra el artefacto del modelo para que pueda ser versionado y comparado más adelante.

```python
import mlflow
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

X, y = make_regression(n_features=4, n_informative=2, random_state=0)
params = {"n_estimators": 3, "random_state": 42}
model = RandomForestRegressor(**params)
model.fit(X, y)

mlflow.log_params(params)
mlflow.sklearn.log_model(model, "model")
```
