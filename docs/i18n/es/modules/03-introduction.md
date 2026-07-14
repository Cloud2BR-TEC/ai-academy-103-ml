# Integración de Fabric y Azure ML (Avanzado)

Este módulo usa el flujo de implementación de tu material fuente de Fabric y lo convierte en una guía operativa para producción.

## Material Fuente Utilizado

- `msFabric-AI_integration/README.md`
- `msFabric-AI_integration/src/fabric-llms-overview_sample.ipynb`

## 1. Setup de Entorno y Capacidad

### 1.1 Registrar proveedor y crear capacidad Fabric

Antes de ejecutar cargas de IA, valida que la capacidad de Fabric exista y esté asignada al workspace correcto.

Checklist:

- Proveedor de recursos Microsoft Fabric registrado
- Capacidad Fabric creada en la región objetivo
- Workspace Power BI/Fabric enlazado a esa capacidad
- Proceso de pausa/reanudación de costos documentado

### 1.2 Preparar dependencias del runtime

En notebooks de Fabric, instala solo lo necesario y fija versiones para reproducibilidad.

```python
%pip install synapseml==1.0.8 langchain==0.3.4 langchain_community==0.3.4 openai==1.53.0 langchain.openai==0.2.4
```

## 2. Patrón de Configuración de Azure OpenAI

Usa variables de entorno o secretos gestionados; nunca claves hardcodeadas en celdas.

```python
import os

os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "<from-secret-store>"
```

Inicializa el cliente del modelo:

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.2,
    top_p=0.9,
    verbose=True,
)
```

## 3. Flujo de Ejecución con LangChain + SynapseML

### 3.1 Definición de prompt y chain

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["technology"],
    template="Define el siguiente término tecnológico para una audiencia de ingeniería ML: {technology}",
)

chain = LLMChain(llm=llm, prompt=prompt)
```

### 3.2 Ejecución distribuida con transformer

```python
from synapse.ml.cognitive.langchain import LangchainTransformer

transformer = (
    LangchainTransformer()
    .setInputCol("technology")
    .setOutputCol("definition")
    .setChain(chain)
    .setSubscriptionKey(os.environ["AZURE_OPENAI_API_KEY"])
    .setUrl(os.environ["AZURE_OPENAI_ENDPOINT"])
)
```

### 3.3 Prueba mínima con DataFrame

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("fabric-llm-demo").getOrCreate()

df = spark.createDataFrame(
    [(0, "docker"), (1, "spark"), (2, "feature store")],
    ["id", "technology"],
)

result_df = transformer.transform(df)
result_df.select("id", "technology", "definition").show(truncate=False)
```

## 4. Puente Operativo hacia Azure ML

Fabric no debe ser el destino final. Úsalo como capa upstream para operaciones gobernadas en Azure ML.

Patrón recomendado de handoff:

1. Persistir salida transformada/lista para features con versionado.
2. Disparar job de entrenamiento o batch scoring en Azure ML.
3. Registrar modelo + métricas en el model registry de Azure ML.
4. Promover solo modelos validados a etapas de despliegue.
5. Monitorear calidad de endpoint y reentrenar cuando aparezca drift.

## 5. Controles de Seguridad, Costo y Confiabilidad

### Seguridad

- Mantener API keys en secretos gestionados.
- Aplicar permisos mínimos (least privilege) para workspace y service principals.
- Evitar logs con payload sensible en crudo.

### Costo

- Limitar tamaño/ventanas de cómputo en notebooks.
- Rastrear consumo de tokens y definir alertas de presupuesto.
- Pausar capacidad Fabric fuera de ventanas planificadas.

### Confiabilidad

- Fijar versiones de paquetes.
- Validar esquema de entrada antes de llamadas LLM.
- Guardar metadatos de ejecución (tiempo, versión de prompt, deployment del modelo, ubicación de salida).

## 6. Checklist de Implementación

- [ ] Capacidad Fabric asignada y validada
- [ ] Dependencias SynapseML/LangChain fijadas
- [ ] Secretos Azure OpenAI externalizados
- [ ] Transformación LLM validada en muestra y lote
- [ ] Handoff hacia entrenamiento/scoring en Azure ML activo
- [ ] Monitoreo y guardrails de costo activos

## 7. Siguiente Paso

Continúa con `07. Terraform para Azure ML` y `08. Terraform para Fabric` para convertir esta integración en infraestructura reproducible.
