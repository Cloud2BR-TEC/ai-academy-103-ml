# 01. Setup de Entorno y Acceso

## Configurar el Workspace de Azure ML

- Ir a [Azure Portal](https://portal.azure.com/)
- Crear un **Machine Learning workspace** — definir resource group, nombre y región.
- Abrir **Azure Machine Learning Studio**.

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/c199156f-96cf-4ed0-a8b5-c88db3e7a552" type="video/mp4">
</video>

## Crear una Compute Instance

1. Abrir Azure ML Studio y acceder a **Compute**.
2. Seleccionar la pestaña **Compute instances** y hacer clic en **New**.
3. Ingresar nombre y tamaño de VM (ej. `Standard_DS3_v2`).
4. Hacer clic en **Create** y esperar el aprovisionamiento.

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/bd5f3ce6-7082-4741-8827-8b344cd249a4" type="video/mp4">
</video>

## Configurar Capacidad de Microsoft Fabric

1. Registrar el proveedor de recursos `microsoft.fabric` en la suscripción.
2. Crear un recurso **Microsoft Fabric** en Azure Portal — elegir suscripción, resource group, nombre de capacidad, región, tamaño y administrador.

![Crear capacidad Fabric](https://github.com/user-attachments/assets/a860911c-0ab8-469e-82d9-d0495268bd3b)

3. Habilitar la capacidad Fabric en el workspace de Power BI.

![Habilitar capacidad en Power BI](https://github.com/user-attachments/assets/5dae8ea7-528d-4742-af51-b877d4610fa4)

4. **Pausar el cómputo de Fabric cuando no se use** para ahorrar costos.

![Pausar Fabric](https://github.com/user-attachments/assets/117d6902-cc8b-45b8-a104-9b54180565f2)
