# Machine Learning 103 - Advanced Implementation Hub

This hub focuses on production implementation patterns for Azure ML and Fabric integration.

## Source Content Reused

This training reuses and consolidates implementation material from your local source repositories:

- `azML-modelcreation` (model creation flow, notebook + scoring script)
- `infrastructure/azMachineLearning` (Terraform Azure ML setup)
- `infrastructure/msFabric` (Fabric infrastructure setup)
- `msFabric-AI_integration` (Fabric + LLM integration notebook)

![Cloud2BR learning asset](assets/img/org-logo.png)

## Learning Path

<div class="home-grid">
  <a class="home-card" href="modules/01-math-prerequisites/"><span class="card-badge">01</span><h3>Platform and Azure Setup</h3><p>Subscription, RBAC, networking, and baseline workspace readiness.</p></a>
  <a class="home-card" href="modules/02-e2e-overview/"><span class="card-badge">02</span><h3>Model Creation Pipeline</h3><p>From data registration to reproducible model artifacts.</p></a>
  <a class="home-card" href="modules/03-introduction/"><span class="card-badge">03</span><h3>Fabric and Azure ML Integration</h3><p>Data + AI architecture and execution boundaries.</p></a>
  <a class="home-card" href="modules/04-ml-foundations/"><span class="card-badge">04</span><h3>Azure ML Environment Strategy</h3><p>Versioned runtimes, dependency controls, and drift prevention.</p></a>
  <a class="home-card" href="modules/05-neural-networks/"><span class="card-badge">05</span><h3>Advanced Data Preparation</h3><p>Contracts, leakage prevention, and scalable feature logic.</p></a>
  <a class="home-card" href="modules/06-azure-ml-environment/"><span class="card-badge">06</span><h3>Training and Experimentation at Scale</h3><p>Experiment tracking, parallel jobs, and AutoML governance.</p></a>
  <a class="home-card" href="modules/07-environment-setup/"><span class="card-badge">07</span><h3>Terraform for Azure ML</h3><p>Deploy AML platform resources with repeatable IaC.</p></a>
  <a class="home-card" href="modules/08-data-preparation/"><span class="card-badge">08</span><h3>Terraform for Fabric</h3><p>Provision Fabric-supporting infrastructure patterns.</p></a>
  <a class="home-card" href="modules/09-model-types/"><span class="card-badge">09</span><h3>Deployment Reliability and SRE</h3><p>Blue/green, canary, rollback, and endpoint incident response.</p></a>
  <a class="home-card" href="modules/10-training-automl/"><span class="card-badge">10</span><h3>Capstone Architecture and Governance</h3><p>End-to-end reference architecture and operating model.</p></a>
</div>

## Visual Anchors

![Azure ML taxonomy](assets/img/azure-machine-learning-taxonomy.svg)
![Azure ML environment taxonomy](assets/img/azure-ml-environment-taxonomy.svg)
![Deployment flow](assets/img/ml_deployment_flow.svg)

## Azure Portal Snapshots

The repository currently doesn't include portal screenshots as raster images yet.
Add them under `docs/i18n/en/assets/img/` and reference them here, for example:

- `azure-portal-workspace-overview.png`
- `azure-portal-compute-instance.png`
- `azure-portal-endpoint-monitoring.png`
