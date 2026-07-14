# 04. Configurar Microsoft Fabric para Cargas de IA

## Objetivo

Dejar Fabric listo para ejecutar el flujo de notebook y conectar resultados con operaciones en Azure ML.

## Configuración paso a paso

1. Registrar el proveedor de Fabric en la suscripción Azure.
2. Crear capacidad Fabric con región/tamaño adecuados.
3. Asignar capacidad al workspace objetivo de Fabric.
4. Abrir notebook de Fabric en experiencia Data Science.
5. Instalar dependencias fijadas:

```python
%pip install synapseml==1.0.8 langchain==0.3.4 langchain_community==0.3.4 openai==1.53.0 langchain.openai==0.2.4
```

6. Configurar Azure OpenAI desde fuente segura de secretos.

```python
import os
os.environ["OPENAI_API_VERSION"] = "2023-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "<from-secret-store>"
```

## Validaciones

- [ ] La sesión de notebook importa SynapseML y LangChain
- [ ] El cliente de Azure OpenAI inicializa correctamente
- [ ] Un prompt de prueba ejecuta con éxito
- [ ] La salida puede persistirse para uso downstream en Azure ML

### Imagen de setup

![Configuración de capacidad Fabric](../assets/img/azure-fabric-capacity-setup.png)
