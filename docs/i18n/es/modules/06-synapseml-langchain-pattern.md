# 06. Patrón de Integración con SynapseML y LangChain

## Descripción de las Herramientas

| **Herramienta** | **Descripción** |
|---|---|
| **LangChain** | Framework para aplicaciones potenciadas por modelos de lenguaje — procesamiento multi-etapa, preprocesamiento de texto, aplicación de LLMs y postprocesamiento de resultados. |
| **SynapseML** | Biblioteca de código abierto para pipelines de ML masivamente escalables — aplica modelos de lenguaje grandes a escala usando cómputo distribuido en Spark. |

## Revisión Bibliográfica a Gran Escala con LangChain

Automatizar extracción y resumen de papers académicos usando un agente que carga un PDF en línea y genera un prompt estructurado.

### Paso 1 — Definir Funciones de Extracción y Prompt

Dos funciones auxiliares impulsan el pipeline. `paper_content_extraction` obtiene un PDF de arXiv a partir de su URL y devuelve las primeras dos páginas de texto. `prompt_generation` toma ese texto extraído y construye un prompt estructurado que le pide al modelo identificar título, autores y resumen, y buscar tres papers recientes del primer autor.

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

### Paso 2 — Construir Cadena Secuencial

Encapsula la función de extracción en un `TransformChain` para que LangChain pueda tratarla como una etapa del pipeline. El `paper_summarizer_template` define la instrucción de sistema que se pasa al modelo en tiempo de ejecución.

```python
from langchain.chains import TransformChain, SimpleSequentialChain

paper_content_extraction_chain = TransformChain(
    input_variables=["arxiv_link"],
    output_variables=["paper_content"],
    transform=paper_content_extraction,
    verbose=False,
)

paper_summarizer_template = """
Eres un resumidor de papers. Tu trabajo es resumir el contenido del paper
y extraer autores y título del mismo.
"""
```

## Registro y Comparación de Modelos con MLflow en Fabric

### Entrenar y Registrar un Modelo

Inicia un run de MLflow, entrena el modelo, infiere la firma de entrada/salida y registra los parámetros y el artefacto del modelo con etiquetas descriptivas. El run ID y el número de versión se imprimen para trazabilidad.

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

### Comparar y Filtrar Modelos Registrados

Usa `MlflowClient` para listar todos los modelos registrados en el workspace. Útil para comparar múltiples ejecuciones de entrenamiento y seleccionar el mejor candidato para promoción a producción.

```python
from pprint import pprint
from mlflow.tracking import MlflowClient

client = MlflowClient()
for rm in client.search_registered_models():
    pprint(dict(rm), indent=4)
```
