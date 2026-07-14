# 05. Crear un Flujo LLM en Notebooks de Fabric

## Fuente Utilizada

- `msFabric-AI_integration/src/fabric-llms-overview_sample.ipynb`

## Pasos

1. Crear plantilla de prompt.
2. Crear `LLMChain` con LangChain.
3. Configurar `LangchainTransformer` de SynapseML.
4. Ejecutar transformer sobre un DataFrame Spark pequeño.
5. Persistir salidas para uso downstream.

## Ejemplo

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(input_variables=["technology"], template="Define: {technology}")
chain = LLMChain(llm=llm, prompt=prompt)
```
