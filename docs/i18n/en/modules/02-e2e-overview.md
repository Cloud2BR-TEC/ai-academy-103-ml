# Model Creation Pipeline (Azure ML Studio Path)

Use this module if your goal is: **create one model in Azure ML Studio and leave it deployment-ready**.

## Source Assets Used

- `azML-modelcreation/data/sample_data.csv`
- `azML-modelcreation/src/0_ml-model-creation.ipynb`
- `azML-modelcreation/src/score.py`

## Step 1 - Register Data

1. Azure ML Studio -> **Data** -> **Create** -> From local files.
2. Upload `sample_data.csv` (or your own dataset).
3. Confirm schema and target column.
4. Save as versioned data asset.

## Step 2 - Prepare Notebook Runtime

1. Open a notebook on compute instance.
2. Install/import required libraries.
3. Load dataset from data asset reference, not local temporary path.

## Step 3 - Train One Baseline Model

1. Split train/test.
2. Train baseline (e.g., RandomForestRegressor).
3. Track metrics: MAE, RMSE, R2.
4. Save artifact (`model.pkl`).

## Step 4 - Evaluate and Decide

Minimum quality gate example:

- RMSE below agreed threshold
- Stable error distribution
- No obvious feature leakage

If gate fails: iterate features/params, retrain.

## Step 5 - Register and Prepare Inference

1. Register model in Azure ML model registry.
2. Reuse `score.py` pattern:
   - input validation
   - deterministic output schema
   - error handling + logs

## Step 6 - Deploy One Endpoint (Optional but recommended)

1. Create environment/inference config.
2. Deploy to online endpoint.
3. Test with sample payload.
4. Confirm response contract + latency.

## Step 7 - Handover Checklist

- [ ] Dataset version documented
- [ ] Training notebook reproducible
- [ ] Metrics stored with run metadata
- [ ] Model registered
- [ ] Scoring contract tested
- [ ] Endpoint smoke test passed
