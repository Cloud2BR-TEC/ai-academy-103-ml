# 02. Crear un Modelo en Azure ML Studio

## Paso 1 — Preparar los Datos

1. Ir a **Data > + Create > From local files** en Azure ML Studio.
2. Subir [`sample_data.csv`](../reference/sample-data-csv.md). Nombre: `employee_data`, Tipo: Tabular.
3. Hacer clic en **Next**, confirmar esquema, luego **Create** para registrar el dataset.

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/f8cbd32c-94fc-43d3-a7a8-00f63cdc543d" type="video/mp4">
</video>

## Paso 2 — Abrir un Notebook

- Abrir un Jupyter notebook en la compute instance.
- Importar las librerías necesarias:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/16650584-11cb-48fb-928d-c032e519c14b" type="video/mp4">
</video>

## Paso 3 — Cargar y Explorar los Datos

```python
import mltable
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
data_asset = ml_client.data.get("employee_data", version="1")
tbl = mltable.load(f'azureml:/{data_asset.id}')
df = tbl.to_pandas_dataframe()
df
```

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/5fa65d95-8502-4ab7-ba0d-dfda66378cc2" type="video/mp4">
</video>

## Paso 4 — Entrenar el Modelo

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoder = LabelEncoder()
df['Department'] = label_encoder.fit_transform(df['Department'])
if 'Name' in df.columns:
    df = df.drop(columns=['Name'])
if df.isnull().sum().any():
    df = df.dropna()

X = df.drop('Salary', axis=1)
y = df['Salary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
```

<video width="100%" controls>
  <source src="https://github.com/user-attachments/assets/2176c795-5fda-4746-93c7-8b137b526a09" type="video/mp4">
</video>

## Paso 5 — Evaluar

```python
import numpy as np
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.2f}")
```

![Resultados de evaluación](https://github.com/user-attachments/assets/6aa19680-cadb-4fe4-a419-a626942e15f9)

![Distribución de errores](https://github.com/user-attachments/assets/d8ec1f2c-eb97-4106-9cee-809849d02796)
