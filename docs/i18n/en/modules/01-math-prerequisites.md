# Platform and Azure Setup

This module is a direct setup guide before creating your first model.

## Goal

Leave this module with:

- Azure ML workspace configured
- Compute ready for training
- Data registration baseline prepared
- Fabric capacity/workspace ready (if using Fabric path)

## Step 1 - Azure Subscription and Permissions

1. Confirm active subscription.
2. Confirm you can create resources in target resource group.
3. Assign roles needed for ML operations (workspace + storage + compute).

## Step 2 - Create Azure ML Workspace

1. Go to Azure Portal.
2. Create **Machine Learning workspace**.
3. Set: resource group, workspace name, region.
4. Open Azure ML Studio after deployment.

![Azure setup snapshot 1](../assets/img/azure-portal-photo-1.jpg)

## Step 3 - Compute Setup

1. In Azure ML Studio, open **Compute**.
2. Create one compute instance for notebooks.
3. (Optional) Create a compute cluster for scheduled training.

Configuration suggestion:

- Instance size: DS3_v2 or equivalent
- Auto-stop enabled for cost control
- Naming convention for team ownership

![Azure setup snapshot 2](../assets/img/azure-portal-photo-2.jpg)

## Step 4 - Data and Storage Baseline

1. Upload/register dataset.
2. Validate schema and target column.
3. Version your dataset from day one.

## Step 5 - Fabric Setup (Only if using Fabric path)

1. Ensure Fabric capacity exists and is assigned.
2. Confirm workspace is connected to capacity.
3. Verify notebook runtime can install required packages.

![Azure setup snapshot 3](../assets/img/azure-portal-photo-3.jpg)

## Exit Checklist

- [ ] Workspace created and accessible
- [ ] Compute instance running
- [ ] Dataset registered with version
- [ ] Fabric capacity/workspace validated (if needed)
