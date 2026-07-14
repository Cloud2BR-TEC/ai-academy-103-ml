# 02. Build One Model in Azure ML Studio

## Step 1 — Prepare Your Data

1. Go to **Data > + Create > From local files** in Azure ML Studio.
2. Upload `sample_data.csv`. Set Name: `employee_data`, Type: Tabular.
3. Click **Next**, confirm the schema, then **Create** to register the dataset.

![Register dataset](https://github.com/user-attachments/assets/f8cbd32c-94fc-43d3-a7a8-00f63cdc543d)

## Step 2 — Open a Notebook

- Open a Jupyter notebook on your compute instance.
- Import required libraries:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

![Open notebook](https://github.com/user-attachments/assets/16650584-11cb-48fb-928d-c032e519c14b)

## Step 3 — Load and Explore the Data

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

![Load data](https://github.com/user-attachments/assets/5fa65d95-8502-4ab7-ba0d-dfda66378cc2)

## Step 4 — Train the Model

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

![Train model](https://github.com/user-attachments/assets/2176c795-5fda-4746-93c7-8b137b526a09)

## Step 5 — Evaluate

```python
import numpy as np
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.2f}")
```

![Evaluation results](https://github.com/user-attachments/assets/6aa19680-cadb-4fe4-a419-a626942e15f9)

![Error distribution](https://github.com/user-attachments/assets/d8ec1f2c-eb97-4106-9cee-809849d02796)
