# Model card

## Intended use

Prioritise identity-verification events for straight-through processing, manual review or escalation. The model consumes operational scores and metadata, never raw document or biometric images.

## Data and privacy

All event-level records are synthetic. No Prembly production data, customer data, identity numbers, names, images or biometrics are included. Production training would require lawful purpose, consent/contract controls, minimisation, encryption, strict retention and a documented DPIA.

## Evaluation

Candidates are selected on a chronological validation period; a later 2024 period is held out. Report ROC-AUC, average precision, Brier score, maximum F1, action rates, expected policy cost and risk-tier slice metrics. Because failure is uncommon, average precision and calibrated probabilities matter more than raw accuracy.

## Human and fairness controls

Model actions are recommendations. High-impact rejection must support human appeal. Country risk tier can encode structural bias and must be legally reviewed; reports compare performance and review rates by tier. Production approval requires group-level error constraints, calibration checks and documented exceptions. Never infer protected attributes from names, faces or documents.

## Limitations

- Synthetic performance does not establish production accuracy.
- Generated relationships are simpler than adaptive fraud.
- Scores from upstream vendors can drift or fail.
- A single expected-cost matrix cannot represent every client policy.
- Anomaly detection flags novelty, not guilt.

