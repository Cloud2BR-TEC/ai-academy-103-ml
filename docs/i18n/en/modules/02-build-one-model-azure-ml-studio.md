# 02. Build One Model in Azure ML Studio

## Source Used

- `azML-modelcreation/src/0_ml-model-creation.ipynb`
- `azML-modelcreation/data/sample_data.csv`

## Steps

1. Upload and register dataset in Azure ML Studio Data section.
2. Open notebook on compute instance.
3. Load data asset reference and validate schema.
4. Train baseline model (as in source notebook).
5. Evaluate with MAE, RMSE, and R2.
6. Save model artifact (`model.pkl`).

## Output

- One trained model artifact
- One metric report
- One reproducible notebook run
