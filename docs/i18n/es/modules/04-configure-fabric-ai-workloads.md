# 04. Configurar Microsoft Fabric para Cargas de IA

## Fuente Utilizada

- `msFabric-AI_integration/README.md`

## Pasos

1. Registrar proveedor Fabric y confirmar capacidad.
2. Asignar workspace a capacidad Fabric.
3. Abrir entorno de notebook de Data Science.
4. Instalar dependencias requeridas (`synapseml`, `langchain`, `openai`).
5. Configurar endpoint y key de Azure OpenAI con enfoque de secretos seguros.

## Fragmento de Configuración

```python
import os
os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "<from-secret-store>"
```
