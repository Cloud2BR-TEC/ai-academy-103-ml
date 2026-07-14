# 01. Environment and Access Setup

## Goal

Prepare the platform so you can run **one full model path** with no blockers.

## A. Azure ML Studio setup

1. Open Azure Portal and create/select your resource group.
2. Create **Machine Learning workspace** in your target region.
3. Open Azure ML Studio and confirm workspace access.
4. Create one compute instance for notebook execution.
5. Enable cost controls (auto-stop and naming convention).

## B. Dataset setup

1. Go to **Data** in Azure ML Studio.
2. Upload `sample_data.csv`.
3. Register as a versioned data asset.
4. Validate schema and target column before training.

## C. Fabric setup (if using Fabric path)

1. Confirm Fabric provider/capacity in Azure.
2. Assign capacity to your Fabric workspace.
3. Open a notebook and verify dependency install works.

### Example: Fabric capacity configuration screen

![Fabric capacity setup](../assets/img/azure-fabric-capacity-setup.png)

## Exit criteria

- [ ] Workspace and compute are operational
- [ ] Dataset is registered and versioned
- [ ] Fabric workspace-capacity assignment is confirmed
- [ ] Team can start Module 02 or Module 04 immediately
