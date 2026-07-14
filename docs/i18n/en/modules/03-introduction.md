# Fabric and Azure ML Integration (Advanced)

This module is implementation-focused and built from your `msFabric-AI_integration` material.

## Implementation Baseline from Source

Source reused:

- `msFabric-AI_integration/README.md`
- `msFabric-AI_integration/src/fabric-llms-overview_sample.ipynb`

## Integration Goal

Define clear system boundaries:

- Fabric handles data engineering, notebook experimentation, and large-scale transformations.
- Azure ML handles model lifecycle governance, deployment, and endpoint operations.

## Setup Sequence (Production-Oriented)

1. Validate Fabric capacity and workspace assignment.
2. Configure dependencies for SynapseML and LangChain components.
3. Configure Azure OpenAI endpoint/key management via secure secrets.
4. Establish notebook execution standards (inputs, outputs, logging).

## LLM Workflow Pattern in Fabric

From your source notebook pattern:

- Prompt template + chain composition.
- Transformer execution over distributed data.
- Optional PDF/content extraction workflow.
- Output persistence to governed storage.

## Bridge to Azure ML Operations

After Fabric processing:

- Persist processed/feature-ready assets with versioning.
- Trigger Azure ML training or scoring pipelines.
- Register model outputs and enforce deployment policies.
- Monitor online/batch performance and drift in operations.

## Operational Risks and Controls

- Prompt/API key leakage -> use secret stores and no hardcoded keys.
- Unbounded LLM cost -> set budget limits and execution quotas.
- Non-reproducible notebooks -> pin dependencies and runtime images.
- Weak lineage -> tag runs and assets across Fabric and Azure ML.

## Advanced Integration Checklist

- [ ] Fabric capacity and workspace validated
- [ ] SynapseML/LangChain dependencies pinned
- [ ] Azure OpenAI secrets configured securely
- [ ] Artifact lineage defined across platforms
- [ ] Monitoring + cost guardrails in place

![Azure portal snapshot 3](../assets/img/azure-portal-photo-3.jpg)
