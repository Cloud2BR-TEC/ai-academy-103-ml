
# Glossary

Concise, technically precise definitions of the terms used across the course. Where useful,
each entry notes *why* the concept matters in practice.

## Azure ML platform

- **Workspace** : the top-level Azure ML container and governance boundary that ties together
  compute, data, models, and endpoints under one identity and access policy.
- **Control plane** : the metadata/intent layer (asset registration, run history, RBAC, lineage).
- **Data plane** : the execution layer (compute, data movement, inference) where cost and
  performance are determined.
- **Compute Instance** : a single always-on VM for interactive development and debugging.
- **Compute Cluster** : auto-scaling compute (0?N nodes) for training, sweeps, and AutoML trials;
  scales back to zero when idle.
- **Environment** : a versioned, pinned runtime (base image + dependencies) reused across training
  and inference to eliminate training/serving skew.
- **Datastore** : a registered connection to underlying storage (e.g. blob, data lake).
- **Data asset / dataset** : a versioned pointer to a specific data snapshot used by jobs.
- **Model registry** : versioned store of trained model artifacts with lineage back to the run,
  data, and environment that produced them.
- **Endpoint** : the deployed, addressable serving surface for a model (online or batch).
- **Lineage** : the recorded chain `data version ? run ? model version ? endpoint revision` that
  makes predictions reproducible and auditable.
- **Managed identity** : an Azure-managed credential attached to a workload so jobs access
  resources without embedded secrets.
- **RBAC** : role-based access control; permissions granted to identities via roles, applied with
  least privilege.

## Core ML concepts

- **Supervised / unsupervised / reinforcement learning** : learning from labeled data, from
  unlabeled structure, and from environment reward respectively.
- **Self-supervised learning** : manufacturing a supervised signal from the data itself (predict
  the masked token); the basis of foundation models.
- **Feature** : an input variable; the **feature vector** $x \in \mathbb{R}^d$ describes one
  example.
- **Label / target** : the value a supervised model predicts.
- **Parameter** : a value learned during training (a weight). **Hyperparameter** : a value set
  before training (e.g. learning rate, tree depth), tuned on validation data.
- **Loss function** : scores how wrong a prediction is; training minimizes its average
  (**empirical risk minimization**).
- **Gradient descent** : iterative parameter update $\theta \leftarrow \theta - \eta\nabla\mathcal{L}$;
  $\eta$ is the **learning rate** (step size).
- **Regularization** : a penalty discouraging complexity; **L1** induces sparsity (feature
  selection), **L2** shrinks weights for stability.
- **Bias / variance** : error from over-simplicity (underfitting) vs error from over-sensitivity
  to the training sample (overfitting).
- **Overfitting** : low training error but high test error; the model memorized noise.
- **Cross-validation** : rotating train/validation folds to estimate generalization with lower
  variance.
- **Data leakage** : information unavailable at prediction time entering training, inflating
  offline metrics.

## Modeling and evaluation

- **Featurization** : transforming raw fields into model-ready features.
- **Logistic regression / sigmoid** : linear model with a sigmoid squashing output into a
  probability.
- **Decision tree / random forest / gradient boosting** : tree splitting by purity; bagging trees
  (forest) reduces variance; boosting trees sequentially reduces bias.
- **Ensemble (bagging / boosting / stacking)** : combining models to reduce variance, reduce bias,
  or learn an optimal combination.
- **Threshold** : the probability cutoff converting scores to decisions; tuned by error costs.
- **Precision / recall / F1** : correctness of positives / coverage of positives / their harmonic
  mean.
- **ROC-AUC / PR-AUC** : threshold-free ranking quality overall / focused on the positive class.
- **MAE / RMSE / $R^2$** : average absolute error / outlier-sensitive error / fit vs the mean
  baseline.
- **Calibration** : agreement between predicted probabilities and observed frequencies.
- **SHAP / LIME / permutation importance** : game-theoretic attribution / local surrogate
  explanation / shuffle-based global importance.

## Operations and MLOps

- **Drift** : change over time in inputs (**covariate drift**, $P(X)$) or in the input?target
  relationship (**concept drift**, $P(Y\mid X)$).
- **PSI** : Population Stability Index; a single number measuring how far a distribution moved
  from baseline.
- **Endpoint (online vs batch)** : real-time request/response vs scheduled bulk scoring.
- **Blue/green, canary, shadow** : release strategies trading off rollback speed and exposure of
  a new model version.
- **SLI / SLO** : a measured reliability indicator / its target threshold.
- **Cold start** : one-time latency of loading the model when a serving replica starts.
- **CrashLoopBackOff** : Kubernetes state where a container repeatedly starts and exits; for ML,
  usually a failed model load in `init()`.
- **Readiness / liveness probe** : checks that gate traffic to a pod / restart a stuck pod.
- **Model card** : a document recording a model's intended use, data, metrics, fairness, limits,
  and operations.

