# Integración de Fabric y Azure ML (Avanzado)

Este módulo está orientado a implementación y se construye sobre tu material `msFabric-AI_integration`.

## Base de Implementación desde la Fuente

Fuentes reutilizadas:

- `msFabric-AI_integration/README.md`
- `msFabric-AI_integration/src/fabric-llms-overview_sample.ipynb`

## Objetivo de Integración

Definir límites claros entre sistemas:

- Fabric gestiona ingeniería de datos, experimentación en notebooks y transformaciones a escala.
- Azure ML gestiona gobernanza del ciclo de vida, despliegue y operación de endpoints.

## Secuencia de Setup (Orientada a Producción)

1. Validar capacidad de Fabric y asignación de workspace.
2. Configurar dependencias para SynapseML y componentes LangChain.
3. Configurar endpoint/keys de Azure OpenAI mediante secretos seguros.
4. Establecer estándares de ejecución de notebooks (entradas, salidas, logging).

## Patrón de Flujo LLM en Fabric

Desde el patrón del notebook fuente:

- Plantilla de prompt + composición de chain.
- Ejecución de transformadores sobre datos distribuidos.
- Flujo opcional de extracción de PDF/contenido.
- Persistencia de salidas en almacenamiento gobernado.

## Puente hacia Operaciones en Azure ML

Después del procesamiento en Fabric:

- Persistir assets procesados/listos para features con versionado.
- Disparar pipelines de entrenamiento o scoring en Azure ML.
- Registrar salidas de modelo y aplicar políticas de despliegue.
- Monitorear rendimiento online/batch y deriva en operación.

## Riesgos Operativos y Controles

- Fuga de prompts/keys -> usar almacenes de secretos y evitar claves hardcodeadas.
- Costo no acotado de LLM -> definir límites de presupuesto y cuotas de ejecución.
- Notebooks no reproducibles -> fijar dependencias e imágenes de runtime.
- Linaje débil -> etiquetar ejecuciones y assets entre Fabric y Azure ML.

## Checklist de Integración Avanzada

- [ ] Capacidad y workspace de Fabric validados
- [ ] Dependencias SynapseML/LangChain fijadas
- [ ] Secretos Azure OpenAI configurados de forma segura
- [ ] Linaje de artefactos definido entre plataformas
- [ ] Monitoreo + guardrails de costo activos

![Captura del portal Azure 3](../assets/img/azure-portal-photo-3.jpg)
