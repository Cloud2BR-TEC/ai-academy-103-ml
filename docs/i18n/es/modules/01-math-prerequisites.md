# Plataforma y Configuración de Azure

Este módulo es una guía directa de setup antes de crear tu primer modelo.

## Objetivo

Terminar este módulo con:

- Workspace de Azure ML configurado
- Cómputo listo para entrenamiento
- Base de registro de datos preparada
- Capacidad/workspace de Fabric listo (si usarás ruta Fabric)

## Paso 1 - Suscripción Azure y Permisos

1. Confirmar suscripción activa.
2. Confirmar permiso para crear recursos en el grupo objetivo.
3. Asignar roles necesarios para operaciones ML (workspace + storage + cómputo).

## Paso 2 - Crear Workspace de Azure ML

1. Ir a Azure Portal.
2. Crear **Machine Learning workspace**.
3. Definir: resource group, nombre de workspace, región.
4. Abrir Azure ML Studio al terminar el despliegue.

![Captura de setup Azure 1](../assets/img/azure-portal-photo-1.jpg)

## Paso 3 - Configuración de Cómputo

1. En Azure ML Studio, abrir **Compute**.
2. Crear una compute instance para notebooks.
3. (Opcional) Crear un clúster para entrenamiento programado.

Sugerencia de configuración:

- Tamaño: DS3_v2 o equivalente
- Auto-stop activado para control de costo
- Convención de nombres por equipo

![Captura de setup Azure 2](../assets/img/azure-portal-photo-2.jpg)

## Paso 4 - Línea Base de Datos y Storage

1. Subir/registrar dataset.
2. Validar esquema y columna objetivo.
3. Versionar dataset desde el día 1.

## Paso 5 - Setup de Fabric (Solo ruta Fabric)

1. Verificar que exista capacidad Fabric y esté asignada.
2. Confirmar que el workspace esté enlazado a la capacidad.
3. Validar que el runtime de notebook pueda instalar paquetes requeridos.

![Captura de setup Azure 3](../assets/img/azure-portal-photo-3.jpg)

## Checklist de Salida

- [ ] Workspace creado y accesible
- [ ] Compute instance en ejecución
- [ ] Dataset registrado con versión
- [ ] Capacidad/workspace Fabric validado (si aplica)
