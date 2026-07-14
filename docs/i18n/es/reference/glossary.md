# Glosario

Definiciones concisas y técnicamente precisas de los términos usados a lo largo del curso. Donde es útil,
cada entrada señala *por qué* el concepto importa en la práctica.

## Plataforma Azure ML

- **Workspace (área de trabajo)** : el contenedor de nivel superior de Azure ML y el límite de gobernanza que une
  cómputo, datos, modelos y endpoints bajo una única política de identidad y acceso.
- **Plano de control** : la capa de metadatos/intención (registro de activos, historial de ejecuciones, RBAC, linaje).
- **Plano de datos** : la capa de ejecución (cómputo, movimiento de datos, inferencia) donde se determinan el costo y el
  rendimiento.
- **Instancia de cómputo** : una única VM siempre activa para desarrollo interactivo y depuración.
- **Clúster de cómputo** : cómputo con autoescalado (0?N nodos) para entrenamiento, barridos y pruebas de AutoML;
  se reduce a cero cuando está inactivo.
- **Entorno** : un tiempo de ejecución versionado y fijado (imagen base + dependencias) reutilizado en el entrenamiento
  y la inferencia para eliminar el sesgo entre entrenamiento y servicio.
- **Datastore** : una conexión registrada al almacenamiento subyacente (por ejemplo, blob, data lake).
- **Activo de datos / conjunto de datos** : un puntero versionado a una instantánea de datos específica usada por los trabajos.
- **Registro de modelos** : almacén versionado de artefactos de modelos entrenados con linaje hacia la ejecución,
  los datos y el entorno que los produjeron.
- **Endpoint** : la superficie de servicio desplegada y direccionable para un modelo (en línea o por lotes).
- **Linaje** : la cadena registrada `versión de datos ? ejecución ? versión del modelo ? revisión del endpoint` que
  hace que las predicciones sean reproducibles y auditables.
- **Identidad administrada** : una credencial administrada por Azure adjunta a una carga de trabajo para que los trabajos accedan a
  recursos sin secretos incrustados.
- **RBAC** : control de acceso basado en roles; permisos otorgados a identidades mediante roles, aplicados con
  privilegio mínimo.

## Conceptos básicos de ML

- **Aprendizaje supervisado / no supervisado / por refuerzo** : aprendizaje a partir de datos etiquetados, de
  estructura no etiquetada y de la recompensa del entorno respectivamente.
- **Aprendizaje autosupervisado** : fabricar una señal supervisada a partir de los datos mismos (predecir
  el token enmascarado); la base de los modelos fundacionales.
- **Característica (feature)** : una variable de entrada; el **vector de características** $x \in \mathbb{R}^d$ describe un
  ejemplo.
- **Etiqueta / objetivo** : el valor que predice un modelo supervisado.
- **Parámetro** : un valor aprendido durante el entrenamiento (un peso). **Hiperparámetro** : un valor establecido
  antes del entrenamiento (por ejemplo, tasa de aprendizaje, profundidad del árbol), ajustado con datos de validación.
- **Función de pérdida** : puntúa cuán equivocada está una predicción; el entrenamiento minimiza su promedio
  (**minimización del riesgo empírico**).
- **Descenso de gradiente** : actualización iterativa de parámetros $\theta \leftarrow \theta - \eta\nabla\mathcal{L}$;
  $\eta$ es la **tasa de aprendizaje** (tamaño del paso).
- **Regularización** : una penalización que desalienta la complejidad; **L1** induce dispersión (selección
  de características), **L2** reduce los pesos para mayor estabilidad.
- **Sesgo / varianza** : error por exceso de simplicidad (subajuste) vs error por exceso de sensibilidad
  a la muestra de entrenamiento (sobreajuste).
- **Sobreajuste (overfitting)** : bajo error de entrenamiento pero alto error de prueba; el modelo memorizó el ruido.
- **Validación cruzada** : rotar los pliegues de entrenamiento/validación para estimar la generalización con menor
  varianza.
- **Fuga de datos (data leakage)** : información no disponible en el momento de la predicción que entra en el entrenamiento, inflando
  las métricas sin conexión.

## Modelado y evaluación

- **Featurización** : transformar campos en bruto en características listas para el modelo.
- **Regresión logística / sigmoide** : modelo lineal con una sigmoide que aplasta la salida en una
  probabilidad.
- **Árbol de decisión / bosque aleatorio / gradient boosting** : división de árboles por pureza; agrupar árboles
  (bosque) reduce la varianza; el boosting de árboles reduce secuencialmente el sesgo.
- **Ensamble (bagging / boosting / stacking)** : combinar modelos para reducir la varianza, reducir el sesgo
  o aprender una combinación óptima.
- **Umbral** : el punto de corte de probabilidad que convierte las puntuaciones en decisiones; ajustado por los costos de error.
- **Precisión / exhaustividad / F1** : corrección de los positivos / cobertura de los positivos / su media
  armónica.
- **ROC-AUC / PR-AUC** : calidad de clasificación independiente del umbral en general / enfocada en la clase positiva.
- **MAE / RMSE / $R^2$** : error absoluto promedio / error sensible a valores atípicos / ajuste vs el baseline de la
  media.
- **Calibración** : concordancia entre las probabilidades predichas y las frecuencias observadas.
- **SHAP / LIME / importancia por permutación** : atribución teórica de juegos / explicación local sustituta
  / importancia global basada en mezcla.

## Operaciones y MLOps

- **Drift** : cambio a lo largo del tiempo en las entradas (**drift de covariables**, $P(X)$) o en la relación
  entrada?objetivo (**drift de concepto**, $P(Y\mid X)$).
- **PSI** : Índice de Estabilidad de la Población; un solo número que mide cuánto se movió una distribución
  respecto al baseline.
- **Endpoint (en línea vs por lotes)** : solicitud/respuesta en tiempo real vs scoring masivo programado.
- **Blue/green, canary, shadow** : estrategias de lanzamiento que equilibran la velocidad de reversión y la exposición de
  una nueva versión del modelo.
- **SLI / SLO** : un indicador de fiabilidad medido / su umbral objetivo.
- **Arranque en frío (cold start)** : latencia única de cargar el modelo cuando arranca una réplica de servicio.
- **CrashLoopBackOff** : estado de Kubernetes donde un contenedor arranca y se cierra repetidamente; para ML,
  usualmente una carga fallida del modelo en `init()`.
- **Sonda de preparación / vivacidad** : verificaciones que controlan el tráfico hacia un pod / reinician un pod atascado.
- **Tarjeta de modelo (model card)** : un documento que registra el uso previsto de un modelo, los datos, las métricas, la equidad, los límites
  y las operaciones.
