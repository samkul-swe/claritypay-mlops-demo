# Cloud Architecture

## Current Deployment
- **Platform:** Render (Docker container)
- **Database:** In-memory (demo)
- **Storage:** Local file system

## Production-Ready Architecture

### AWS Services (Free Tier Ready)
```
┌─────────────────────────────────────────┐
│           AWS Architecture               │
└─────────────────────────────────────────┘

Data Storage:
├── S3: Model artifacts, training data
├── RDS: Application database (future)
└── DynamoDB: Predictions log (future)

Compute:
├── Lambda: Serverless inference
├── SageMaker: Model training & deployment
└── ECS/Fargate: Container orchestration

Monitoring:
├── CloudWatch: Logs & metrics
└── X-Ray: Distributed tracing
```

### Databricks Integration
```
┌─────────────────────────────────────────┐
│        Databricks Platform               │
└─────────────────────────────────────────┘

Data Processing:
├── Delta Lake: Feature store
└── Spark: Distributed processing

ML Platform:
├── MLflow: Experiment tracking
├── AutoML: Automated model selection
└── Feature Store: Centralized features

Deployment:
├── Model Registry: Version control
└── Model Serving: REST endpoints
```

## Migration Path

### Phase 1: Data Lake (Implemented)
- [x] S3 bucket created
- [x] Model artifacts uploaded
- [x] Training data stored

### Phase 2: Databricks Setup (Implemented)
- [x] Community Edition account
- [x] Training notebook created
- [x] MLflow integration tested

### Phase 3: Production Deployment (Future)
- [ ] Lambda deployment
- [ ] SageMaker pipeline
- [ ] CloudWatch monitoring
- [ ] Auto-scaling setup

## Cost Optimization
- Free Tier: $0/month (first 12 months)
- Community Edition: $0/month (forever)
- Estimated production cost: ~$50-100/month for 10K requests/day
