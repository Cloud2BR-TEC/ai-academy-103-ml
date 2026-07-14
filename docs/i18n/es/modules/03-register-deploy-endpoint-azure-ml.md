# 03. Registrar y Desplegar Endpoint en Azure ML

## Fuente Utilizada

- `azML-modelcreation/src/score.py`

## Pasos

1. Registrar el modelo en el model registry de Azure ML.
2. Preparar script de scoring basado en `score.py`.
3. Crear configuración de environment/inference.
4. Desplegar endpoint online.
5. Probar endpoint con payload de ejemplo.
6. Validar esquema de respuesta y latencia.

## Aceptación Mínima

- Endpoint responde predicción válida.
- Errores con mensajes claros.
- Logs disponibles para troubleshooting.
