# Model Creation Pipeline (Advanced)

This module adapts your `azML-modelcreation` workflow into a production-oriented blueprint.

## Implementation Baseline from Source

Source reused:

- `azML-modelcreation/README.md`
- `azML-modelcreation/src/0_ml-model-creation.ipynb`
- `azML-modelcreation/src/score.py`
- `azML-modelcreation/data/sample_data.csv`

## Stage 1 - Platform Readiness

- Confirm workspace, subscription, region, and RBAC model.
- Establish compute strategy (interactive vs training cluster).
- Define naming/version conventions for data, jobs, and models.

## Stage 2 - Data Registration Contract

From the source flow, enforce a stricter contract before training:

- Register dataset with explicit schema expectations.
- Validate nulls, target leakage risks, and high-cardinality identifiers.
- Track dataset version and lineage for every run.

## Stage 3 - Notebook-to-Pipeline Promotion

Use notebook exploration, then promote to repeatable execution:

1. Train/evaluate in notebook.
2. Export core logic into reproducible script steps.
3. Parameterize split ratio, seed, and feature controls.
4. Run as tracked job for comparability.

## Stage 4 - Model Registration Quality Gate

Before registering:

- Define minimum metric thresholds.
- Save evaluation artifacts (plots + metrics JSON/CSV).
- Record environment metadata and dependency versions.
- Register only models that pass the gate.

## Stage 5 - Deployment-Ready Scoring Contract

Leverage `score.py` style, but productionize it:

- Strict input schema validation.
- Deterministic output envelope.
- Structured logging for request tracing.
- Clear error codes for invalid payloads.

## Practical Checklist

- [ ] Data asset version pinned
- [ ] Training job reproducible
- [ ] Metrics threshold documented
- [ ] Model registered with lineage
- [ ] Scoring contract validated with sample payloads

