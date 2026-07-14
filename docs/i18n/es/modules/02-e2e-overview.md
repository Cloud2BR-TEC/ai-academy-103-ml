# Pipeline de Creación de Modelos (Avanzado)

Este módulo adapta tu flujo `azML-modelcreation` a un blueprint orientado a producción.

## Base de Implementación desde la Fuente

Fuentes reutilizadas:

- `azML-modelcreation/README.md`
- `azML-modelcreation/src/0_ml-model-creation.ipynb`
- `azML-modelcreation/src/score.py`
- `azML-modelcreation/data/sample_data.csv`

## Etapa 1 - Preparación de Plataforma

- Confirmar workspace, suscripción, región y modelo RBAC.
- Definir estrategia de cómputo (interactivo vs clúster de entrenamiento).
- Establecer convenciones de nombres/versiones para datos, jobs y modelos.

## Etapa 2 - Contrato de Registro de Datos

Desde el flujo fuente, aplica un contrato más estricto antes de entrenar:

- Registrar dataset con expectativas explícitas de esquema.
- Validar nulos, riesgos de fuga de target e identificadores de alta cardinalidad.
- Trazar versión de dataset y linaje en cada ejecución.

## Etapa 3 - Promoción de Notebook a Pipeline

Usa notebook para exploración y luego promueve a ejecución repetible:

1. Entrenar/evaluar en notebook.
2. Extraer lógica central a pasos reproducibles de script.
3. Parametrizar split ratio, semilla y controles de features.
4. Ejecutar como job rastreable para comparar resultados.

## Etapa 4 - Puerta de Calidad para Registro de Modelo

Antes de registrar:

- Definir umbrales mínimos de métricas.
- Guardar artefactos de evaluación (gráficos + métricas JSON/CSV).
- Registrar metadatos de entorno y versiones de dependencias.
- Registrar solo modelos que superen la puerta.

## Etapa 5 - Contrato de Scoring Listo para Despliegue

Aprovecha el estilo `score.py`, pero orientado a producción:

- Validación estricta del esquema de entrada.
- Envelope de salida determinístico.
- Logging estructurado para trazabilidad de requests.
- Códigos de error claros para payloads inválidos.

## Checklist Práctico

- [ ] Versión de data asset fijada
- [ ] Job de entrenamiento reproducible
- [ ] Umbral de métricas documentado
- [ ] Modelo registrado con linaje
- [ ] Contrato de scoring validado con payloads de prueba

![Captura del portal Azure 1](../assets/img/azure-portal-photo-1.jpg)
![Captura del portal Azure 2](../assets/img/azure-portal-photo-2.jpg)
