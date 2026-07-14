# 05. Build One LLM Flow in Fabric Notebooks

## Source Used

- `msFabric-AI_integration/src/fabric-llms-overview_sample.ipynb`

## Steps

1. Create prompt template.
2. Create LangChain `LLMChain`.
3. Configure `LangchainTransformer` from SynapseML.
4. Run transformer on a small Spark DataFrame.
5. Persist outputs for downstream use.

## Example

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(input_variables=["technology"], template="Define: {technology}")
chain = LLMChain(llm=llm, prompt=prompt)
```
