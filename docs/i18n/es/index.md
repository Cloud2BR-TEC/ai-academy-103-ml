# Machine Learning 103 - Guía de Setup y Construcción

Machine Learning 103 ahora es una guía práctica orientada a implementación.

## Qué Vas a Hacer

Vas a seguir una de dos rutas completas:

1. Crear un modelo en Azure ML Studio.
2. Crear un modelo en Microsoft Fabric.

Ambas rutas incluyen setup, configuración, validación y salidas listas para despliegue.

## Orden Recomendado

1. `01. Plataforma y Configuración de Azure`
2. `02. Pipeline de Creación de Modelos` (ruta Azure ML Studio)
3. `03. Integración de Fabric y Azure ML` (ruta Fabric)

Continúa con los módulos 04-14 después de completar al menos una ruta end-to-end.

## Guía Rápida de Decisión

- Elige **Azure ML Studio** si tu objetivo es ciclo de vida ML clásico + despliegue de endpoint.
- Elige **Fabric** si tu objetivo es flujo de datos + LLM en notebooks a escala.

## Entradas Requeridas

- Suscripción Azure con permisos
- Workspace y presupuesto de cómputo
- Dataset de ejemplo (`sample_data.csv`) o equivalente
- Gestión segura de claves para Azure OpenAI (ruta Fabric)
