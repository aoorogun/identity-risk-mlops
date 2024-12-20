# Prembly Identity Verification Risk & Operations ML

An end-to-end, privacy-safe reference platform for identity-verification decision support. It predicts likely verification failure, detects anomalous traffic, recommends straight-through approval/manual review/rejection, and simulates the operational impact of policy changes.

**Data statement:** the event-level dataset is synthetic and contains no Prembly customer, applicant, biometric, document image or confidential production data. Its schema is inspired only by public identity-verification concepts documented by Prembly.

## Role-aligned capabilities

- Supervised learning: calibrated failure-risk models with time-based validation
- Unsupervised learning: Isolation Forest traffic anomaly scores
- Decision optimisation: expected-cost thresholds for approve/review/reject
- Dynamic reforecasting: policy and traffic shocks recalculate review volume, losses and SLA load
- MLOps: schema validation, data hash, versioned artifacts, tests, API and monitoring
- AWS: SageMaker inference contract, encrypted infrastructure and CI/CD reference
- Deep learning: optional PyTorch multi-task network for failure and latency prediction

## Business problem

The system estimates `verification_failed` from non-sensitive operational signals such as document type, country risk tier, face/liveness scores, device integrity, request velocity, image quality, retry count and vendor latency. A policy layer converts risk into an action:

```text
low risk -> approve automatically
medium risk -> manual review
high risk -> reject/escalate
```

Thresholds are selected on validation data using configurable false-approval, false-rejection and manual-review costs. Predictions support an analyst; they do not replace legal/compliance judgement.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

prembly-risk generate
prembly-risk train
prembly-risk evaluate
prembly-risk scenario --file configs/scenario_high_risk_launch.json
prembly-risk monitor
python -m unittest discover -s tests -v
```

Serve the model:

```bash
uvicorn prembly_risk.api:app --host 0.0.0.0 --port 8080
```

## Repository map

```text
src/prembly_risk/        generation, features, models, policy, scenario, API
configs/                 model and scenario configuration
tests/                   leakage, policy and integration checks
artifacts/               fitted model and metadata
reports/                 verified model, fairness and scenario results
docs/                    model card, architecture, runbook and interview guide
infra/                   AWS CloudFormation baseline
.github/workflows/       CI and controlled deployment
```

See [MODEL_CARD.md](docs/MODEL_CARD.md) before interpreting results.

