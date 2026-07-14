# 01. Environment and Access Setup

## Set Up Your Azure ML Workspace

- Go to the [Azure Portal](https://portal.azure.com/)
- Create a **Machine Learning workspace** — set resource group, workspace name, and region.
- Launch **Azure Machine Learning Studio**.

![Workspace setup](https://github.com/user-attachments/assets/c199156f-96cf-4ed0-a8b5-c88db3e7a552)

## Create a Compute Instance

1. Open Azure ML Studio and go to **Compute**.
2. Select the **Compute instances** tab and click **New**.
3. Enter a name, choose a VM size (e.g., `Standard_DS3_v2`).
4. Click **Create** and wait for provisioning.

![Compute instance](https://github.com/user-attachments/assets/bd5f3ce6-7082-4741-8827-8b344cd249a4)

## Set Up Microsoft Fabric Capacity

1. Register the `microsoft.fabric` resource provider in your subscription.
2. Create a new **Microsoft Fabric** resource in Azure Portal — choose subscription, resource group, capacity name, region, size, and administrator.

![Create Fabric capacity](https://github.com/user-attachments/assets/a860911c-0ab8-469e-82d9-d0495268bd3b)

3. Enable Fabric Capacity in your Power BI workspace.

![Enable capacity in Power BI](https://github.com/user-attachments/assets/5dae8ea7-528d-4742-af51-b877d4610fa4)

4. **Pause Fabric compute when not in use** to save costs.

![Pause Fabric compute](https://github.com/user-attachments/assets/117d6902-cc8b-45b8-a104-9b54180565f2)
