# Architecture and operations

```text
verification event -> schema/privacy gate -> online features -> risk + anomaly models
                                                            |
                                 policy thresholds -> approve/review/escalate
                                                            |
                                audit log, analyst outcome, monitoring feedback
```

On AWS, API Gateway or an internal service invokes a private SageMaker endpoint. Kinesis/SQS buffers events for feature and outcome processing. Encrypted S3 holds versioned training snapshots; SageMaker Pipelines performs validation, training, calibration and evaluation; Model Registry requires manual approval. CloudWatch tracks latency, traffic, action rates and drift. DynamoDB stores idempotency keys and decision lineage without raw PII.

Operational SLOs: 99.9% service availability, p95 assessment under 250 ms, zero unauthorised raw-biometric persistence, daily data-quality reporting and critical-alert acknowledgement within 30 minutes. Rollback uses the previous immutable endpoint configuration.

