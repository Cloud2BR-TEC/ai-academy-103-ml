# fabric-llms-overview_sample.ipynb

Notebook completo de Jupyter que demuestra la integración de LLMs en Microsoft Fabric usando LangChain y SynapseML. Cubre la configuración de Azure OpenAI, construcción de pipelines con LLMChain y LangchainTransformer, inferencia distribuida en Spark DataFrames y registro de modelos ML con MLflow.

[⬇ Descargar fabric-llms-overview_sample.ipynb](../assets/files/fabric-llms-overview_sample.ipynb){ .md-button .md-button--primary download="fabric-llms-overview_sample.ipynb" }

---

## Contenido del Notebook

| Celda | Descripción |
|---|---|
| 1 | Instalar `openai`, `langchain_community` |
| 2 | Establecer variables de entorno de Azure OpenAI |
| 3 | Inicializar instancia de `AzureChatOpenAI` |
| 4 | Crear `PromptTemplate` para definiciones de tecnología |
| 5 | Construir `LLMChain` vinculando prompt al LLM |
| 6 | Configurar `LangchainTransformer` para Spark |
| 7 | Ejecutar transformer sobre un Spark DataFrame de ejemplo |
| 8 | Definir funciones de extracción de PDF y generación de prompts |
| 9 | Construir cadena secuencial para agente de revisión bibliográfica |
| 10 | Entrenar `RandomForestRegressor` y registrar con MLflow |
| 11 | Registrar y comparar modelos con `MlflowClient` |

> Abre el notebook descargado en Microsoft Fabric → **Data Science** → **Notebooks**, o en VS Code con la extensión de Jupyter.
