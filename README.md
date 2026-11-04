
This repository is my hands-on exploration of MLOps, from preparing and training machine learning models to deploying, monitoring, maintaining, and continuously improving them in production.

The goal is to understand what happens after a machine learning model is trained.

Training a model is only one part of a real machine learning system.

A real production workflow looks more like:

```text
Data
  ↓
Validation
  ↓
Preprocessing
  ↓
Feature Engineering
  ↓
Training
  ↓
Evaluation
  ↓
Experiment Tracking
  ↓
Model Registry
  ↓
CI/CD
  ↓
Deployment
  ↓
Model Serving
  ↓
Monitoring
  ↓
Drift Detection
  ↓
Retraining
  ↓
Redeployment
```

This repository explores that complete lifecycle and the engineering problems that appear when machine learning moves from experimentation to production.

---

# Why MLOps Is Needed

A simple machine learning project can look like:

```text
Dataset
   ↓
Train Model
   ↓
Evaluate
   ↓
model.pkl
```

This may work on a laptop.

But production machine learning is much more complicated.

Imagine a model trained today:

```text
Training Data
     ↓
Model v1
     ↓
Production
```

A few weeks later:

```text
Real-World Data Changes
        ↓
Model Performance Drops
        ↓
Production Problem
```

Now we need to know:

```text
Which data trained the model?
Which code version was used?
Which parameters were used?
Which features were generated?
Which model was deployed?
Who deployed it?
When was it deployed?
What was the model's performance?
Has the data distribution changed?
Should the model be retrained?
```

This is where MLOps becomes important.

---

# The Core Problem

Machine learning development has two very different environments.

## Development

```text
Notebook
   ↓
Experiment
   ↓
Train
   ↓
Change Code
   ↓
Train Again
   ↓
Compare Results
```

## Production

```text
Application
   ↓
API
   ↓
Model
   ↓
Predictions
   ↓
Users
   ↓
Real-World Data
```

The challenge is connecting these two worlds reliably.

```text
ML Research
     ↓
ML Engineering
     ↓
Production System
```

MLOps provides the engineering practices needed to make this transition repeatable and maintainable.

---

# What Is MLOps?

MLOps is the set of engineering practices used to build, deploy, operate, monitor, and continuously improve machine learning systems.

It combines ideas from:

```text
Machine Learning
      +
Software Engineering
      +
DevOps
      +
Data Engineering
      +
Cloud Infrastructure
      +
Monitoring
```

A simplified view:

```text
          MLOps

Data
  +
Code
  +
Models
  +
Infrastructure
  +
Automation
  +
Monitoring
```

The goal is not simply to deploy a model.

The goal is to build a reliable machine learning lifecycle.

---

# Complete MLOps Lifecycle

The complete lifecycle explored in this repository is:

```text
                           MLOps LIFECYCLE

Data Collection
      ↓
Data Validation
      ↓
Data Versioning
      ↓
Data Processing
      ↓
Feature Engineering
      ↓
Dataset Creation
      ↓
Model Training
      ↓
Experiment Tracking
      ↓
Model Evaluation
      ↓
Model Registration
      ↓
Model Validation
      ↓
CI/CD
      ↓
Deployment
      ↓
Model Serving
      ↓
Prediction
      ↓
Monitoring
      ↓
Drift Detection
      ↓
Performance Analysis
      ↓
Retraining
      ↓
New Model Version
      ↓
Deployment
      ↓
Monitoring
      ↓
...
```

The lifecycle is continuous.

---

# Development vs Production

A major objective of this repository is understanding the difference between a model experiment and a production ML system.

## Experiment

```text
Notebook
   ↓
Dataset
   ↓
Model
   ↓
Metric
```

## Production

```text
Data
 ↓
Pipeline
 ↓
Training
 ↓
Validation
 ↓
Registry
 ↓
CI/CD
 ↓
Deployment
 ↓
Serving
 ↓
Monitoring
 ↓
Retraining
```

The second system requires much more engineering.

---

# From Notebook to Production

A typical starting point:

```text
train.ipynb
```

Eventually, this needs to become something like:

```text
Data Pipeline
      ↓
Training Pipeline
      ↓
Evaluation Pipeline
      ↓
Model Registry
      ↓
Deployment Pipeline
      ↓
Production Service
      ↓
Monitoring
```

This repository explores that transformation.

---

# 1. Data

Everything starts with data.

```text
Raw Data
   ↓
Validation
   ↓
Cleaning
   ↓
Transformation
   ↓
Training Dataset
```

I will explore:

```text
Data ingestion
Data validation
Data cleaning
Data preprocessing
Feature engineering
Dataset versioning
Data quality
Data lineage
```

The important idea is:

```text
Bad Data
   ↓
Bad Model
   ↓
Bad Production Predictions
```

MLOps therefore begins before model training.

---

# 2. Data Validation

Before training, the dataset should be checked.

For example:

```text
Schema
Missing Values
Duplicates
Data Types
Ranges
Categories
Outliers
Distribution
Label Quality
```

A validation flow:

```text
Dataset
   ↓
Schema Validation
   ↓
Quality Checks
   ↓
Distribution Checks
   ↓
Valid?
  / \
Yes  No
 ↓    ↓
Train  Stop Pipeline
```

The goal is to prevent bad data from silently entering the training pipeline.

---

# 3. Data Versioning

Machine learning experiments depend heavily on data.

Suppose:

```text
Model A
```

was trained on:

```text
Dataset v1
```

and later:

```text
Model B
```

was trained on:

```text
Dataset v2
```

If we do not track the dataset versions, reproducing the experiments becomes difficult.

The relationship should be traceable:

```text
Dataset v1
    ↓
Code v3
    ↓
Config v5
    ↓
Model v10
```

I will explore data versioning and dataset lineage.

---

# 4. Reproducibility

A major MLOps goal is reproducibility.

If I train a model today:

```text
Model v1
```

I should be able to understand how it was produced.

That means tracking:

```text
Code
Data
Dependencies
Configuration
Hyperparameters
Random Seeds
Environment
Model
Metrics
```

Conceptually:

```text
                    MODEL

             +----------------+
             | Model Version  |
             +----------------+
                     |
       +-------------+-------------+
       |             |             |
      Data          Code        Config
       |             |             |
       +-------------+-------------+
                     |
                  Metrics
```

---

# 5. Experiment Tracking

During development, many experiments may be performed.

For example:

```text
Experiment 1
Learning Rate = 0.001
Accuracy = 91%

Experiment 2
Learning Rate = 0.0005
Accuracy = 93%

Experiment 3
Learning Rate = 0.0001
Accuracy = 92%
```

Without tracking, it becomes difficult to know which experiment produced which result.

Experiment tracking records:

```text
Parameters
Metrics
Artifacts
Models
Dataset Versions
Code Version
Environment
```

I will explore how experiment tracking connects experiments to reproducible model versions.

---

# 6. Model Evaluation

A model should not be registered or deployed simply because training completed successfully.

The pipeline should evaluate it.

```text
Training
   ↓
Validation
   ↓
Evaluation
   ↓
Quality Gate
   ↓
Pass?
 / \
Yes No
 ↓   ↓
Register Stop
```

Evaluation may include:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
MAE
MSE
RMSE
Calibration
Business Metrics
Latency
Resource Usage
```

The metrics depend on the ML problem.

---

# 7. Model Registry

After evaluation, models need to be managed.

A model registry provides a central location for model versions and their lifecycle.

Conceptually:

```text
Model v1
Model v2
Model v3
Model v4
```

with lifecycle stages such as:

```text
Development
     ↓
Validation
     ↓
Candidate
     ↓
Production
     ↓
Archived
```

I will explore:

```text
Model Versioning
Model Metadata
Model Artifacts
Model Promotion
Model Approval
Model Rollback
```

---

# 8. Model Lineage

A production model should have traceability.

For example:

```text
Production Model v7
        |
        +── Dataset v12
        |
        +── Code Commit abc123
        |
        +── Configuration v4
        |
        +── Experiment 82
        |
        +── Evaluation Results
        |
        +── Container Image
```

This makes it possible to answer:

```text
Where did this model come from?
```

and:

```text
What exactly changed between model v6 and v7?
```

---

# 9. CI/CD for Machine Learning

Traditional software uses CI/CD:

```text
Code
 ↓
Test
 ↓
Build
 ↓
Deploy
```

Machine learning introduces additional stages:

```text
Code
 ↓
Data Validation
 ↓
Training
 ↓
Evaluation
 ↓
Model Validation
 ↓
Build
 ↓
Deploy
```

This is often described as CI/CD/CT:

```text
CI
Continuous Integration

CD
Continuous Delivery / Deployment

CT
Continuous Training
```

I will explore how these processes fit together.

---

# 10. Continuous Integration

Whenever code changes:

```text
Developer
   ↓
Git Commit
   ↓
CI Pipeline
   ↓
Tests
   ↓
Linting
   ↓
Build
```

For ML systems, testing can include:

```text
Unit Tests
Data Tests
Pipeline Tests
Model Tests
API Tests
Schema Tests
```

---

# 11. Continuous Delivery

After validation:

```text
Build
  ↓
Package
  ↓
Containerize
  ↓
Deploy to Staging
  ↓
Validate
  ↓
Production
```

The objective is to make deployments repeatable.

---

# 12. Continuous Training

Production data changes.

Therefore, the model may eventually need retraining.

```text
Production Data
      ↓
Monitoring
      ↓
Drift Detected
      ↓
Retraining Trigger
      ↓
Training Pipeline
      ↓
New Model
      ↓
Evaluation
      ↓
Registry
      ↓
Deployment
```

This creates a continuous ML lifecycle.

---

# 13. Model Serving

A trained model needs to provide predictions.

A basic serving architecture:

```text
Client
  ↓
API
  ↓
Model Server
  ↓
Model
  ↓
Prediction
  ↓
Response
```

I will explore different serving patterns:

```text
REST API
gRPC
Batch Inference
Online Inference
Streaming Inference
```

---

# 14. Online Inference

For real-time applications:

```text
User Request
     ↓
API
     ↓
Preprocessing
     ↓
Model
     ↓
Postprocessing
     ↓
Response
```

Important metrics include:

```text
Latency
Throughput
Error Rate
CPU Usage
GPU Usage
Memory
```

---

# 15. Batch Inference

Not every prediction needs to happen in real time.

A batch system can look like:

```text
Dataset
   ↓
Batch Job
   ↓
Model
   ↓
Predictions
   ↓
Storage
```

Examples include:

```text
Daily Customer Predictions
Weekly Risk Scores
Recommendation Updates
Offline Scoring
```

I will compare online and batch inference.

---

# 16. Model Deployment

The deployment flow can be:

```text
Model Registry
      ↓
Approved Model
      ↓
Build Artifact
      ↓
Container Image
      ↓
Deployment
      ↓
Serving Infrastructure
```

Deployment strategies explored in this repository include:

```text
Rolling Deployment
Blue-Green Deployment
Canary Deployment
Shadow Deployment
Rollback
```

---

# 17. Canary Deployment

Instead of immediately sending all traffic to a new model:

```text
Old Model → 95% Traffic
New Model → 5% Traffic
```

We can monitor the new version.

If it performs well:

```text
Old Model → 50%
New Model → 50%
```

and eventually:

```text
Old Model → 0%
New Model → 100%
```

If something goes wrong:

```text
New Model
   ↓
Rollback
   ↓
Old Model
```

---

# 18. Shadow Deployment

A new model can receive production inputs without affecting the real response.

```text
                Request
                   |
          +--------+--------+
          |                 |
          v                 v
      Production          Shadow
        Model              Model
          |                 |
          v                 v
      Real Result       Evaluation Only
```

This allows the new model to be evaluated using real traffic before becoming production.

---

# 19. Model Monitoring

Deployment is not the end.

A model can degrade after deployment.

We therefore monitor:

```text
System
  +
Model
  +
Data
  +
Business Metrics
```

---

# 20. System Monitoring

Infrastructure metrics include:

```text
CPU
GPU
Memory
Disk
Network
Latency
Throughput
Error Rate
Request Rate
```

Example:

```text
Requests
   ↓
API
   ↓
Model Server
   ↓
Metrics
   ↓
Monitoring System
   ↓
Alert
```

---

# 21. Data Drift

Real-world data can change.

For example:

```text
Training Data
     ↓
Customer Behavior
```

may look different from:

```text
Production Data
     ↓
New Customer Behavior
```

This is called data drift.

Conceptually:

```text
Training Distribution
        ↓
Production Distribution
        ↓
Compare
        ↓
Drift?
```

I will explore statistical approaches for detecting distribution changes.

---

# 22. Concept Drift

Data distribution can change, but more importantly, the relationship between inputs and targets can change.

For example:

```text
X → Y
```

may behave differently over time.

This can cause model performance to decline even if the input distribution does not change significantly.

I will distinguish:

```text
Data Drift
Concept Drift
Prediction Drift
Performance Drift
```

---

# 23. Prediction Monitoring

Even when labels are not immediately available, predictions themselves can be monitored.

For example:

```text
Prediction Distribution

Class A → 70%
Class B → 25%
Class C → 5%
```

If the distribution suddenly changes:

```text
Class A → 20%
Class B → 10%
Class C → 70%
```

that can be an important production signal.

---

# 24. Model Performance Monitoring

When ground-truth labels eventually become available:

```text
Predictions
     +
Actual Labels
     ↓
Performance Metrics
     ↓
Compare Against Baseline
```

This allows us to determine whether the model is actually degrading.

---

# 25. Alerts

Monitoring becomes useful when it can trigger action.

```text
Metric
  ↓
Threshold
  ↓
Violation
  ↓
Alert
  ↓
Investigation
```

Examples:

```text
Latency Too High
Error Rate Too High
Drift Detected
Accuracy Dropped
GPU Utilization Abnormal
Prediction Distribution Changed
```

---

# 26. Retraining

A production model can eventually require retraining.

The complete loop:

```text
Production
    ↓
Monitoring
    ↓
Drift / Performance Change
    ↓
Retraining Trigger
    ↓
New Dataset
    ↓
Training
    ↓
Evaluation
    ↓
Model Registry
    ↓
Deployment
    ↓
Monitoring
```

This creates the continuous nature of MLOps.

---

# 27. Retraining Strategies

I will explore different strategies.

## Scheduled Retraining

```text
Every Week
     ↓
Train
     ↓
Evaluate
     ↓
Deploy
```

## Trigger-Based Retraining

```text
Drift Detected
     ↓
Retraining
```

## Performance-Based Retraining

```text
Performance < Threshold
     ↓
Retraining
```

The correct strategy depends on the application.

---

# 28. Model Rollback

Production systems need a safe recovery mechanism.

```text
Model v5
   ↓
Production
   ↓
Problem
   ↓
Rollback
   ↓
Model v4
```

The model registry and deployment system should make this possible.

---

# 29. Feature Store

For some production ML systems, features need to be consistently available for both training and inference.

A conceptual feature store architecture:

```text
Raw Data
   ↓
Feature Pipeline
   ↓
Feature Store
   ├──────────────→ Training
   |
   └──────────────→ Inference
```

This can help reduce training-serving skew.

I will explore:

```text
Offline Features
Online Features
Feature Versioning
Feature Consistency
Feature Freshness
```

---

# 30. Training-Serving Skew

A common production problem occurs when training and inference preprocess data differently.

Training:

```text
Raw Data
 ↓
Preprocessing A
 ↓
Model
```

Production:

```text
Raw Data
 ↓
Preprocessing B
 ↓
Model
```

Even if the model is correct, the prediction quality can suffer.

The goal is:

```text
Training Transformation
        ≈
Serving Transformation
```

This repository will explore ways to maintain consistency.

---

# 31. Model Artifacts

A production model is more than a model file.

Artifacts may include:

```text
Model Weights
Tokenizer
Preprocessing Logic
Feature Definitions
Configuration
Metadata
Dependencies
Evaluation Results
```

The deployment package should contain everything required to reproduce inference correctly.

---

# 32. Containerization

A model should behave consistently across environments.

Instead of:

```text
My Laptop
    ↓
"It works here"
```

we can package the application:

```text
Application
+
Dependencies
+
Runtime
+
Model
     ↓
Container
```

I will explore containerizing ML services with Docker.

---

# 33. Infrastructure

Production ML systems may run on:

```text
Cloud
Kubernetes
VMs
Managed ML Platforms
GPU Infrastructure
Serverless Systems
```

I will explore how infrastructure choices affect:

```text
Cost
Latency
Scalability
Reliability
Operations
```

---

# 34. Kubernetes

For larger workloads:

```text
                Kubernetes

              Load Balancer
                    ↓
             Model Service
          +---------+---------+
          |         |         |
        Pod 1     Pod 2     Pod 3
          |         |         |
        Model     Model     Model
```

This allows model serving workloads to be managed and scaled.

---

# 35. Scaling

A production model may receive:

```text
10 requests/sec
```

and later:

```text
10,000 requests/sec
```

The system needs to scale.

Important concepts include:

```text
Horizontal Scaling
Vertical Scaling
Autoscaling
Batching
Caching
GPU Scaling
Load Balancing
```

---

# 36. Model Optimization

Serving a model efficiently may require:

```text
Quantization
Pruning
Compilation
Batching
Caching
ONNX
TensorRT
GPU Optimization
```

The objective is to balance:

```text
Accuracy
Latency
Throughput
Memory
Cost
```

---

# 37. Cost Monitoring

A production ML system has infrastructure costs.

For example:

```text
Training
   +
Storage
   +
GPU
   +
Inference
   +
Networking
   +
Monitoring
```

A model that performs slightly better but costs 10 times more may not be the best production model.

MLOps therefore includes cost awareness.

---

# 38. Security

Production ML systems also need security.

Important areas include:

```text
Authentication
Authorization
Secrets Management
Network Security
Data Security
Model Access
API Security
Container Security
Supply Chain Security
```

Security should be part of the lifecycle rather than added at the end.

---

# 39. Testing

A production ML pipeline needs multiple types of tests.

```text
Unit Tests
     ↓
Data Tests
     ↓
Feature Tests
     ↓
Model Tests
     ↓
API Tests
     ↓
Integration Tests
     ↓
Deployment Tests
```

The goal is to catch failures before production.

---

# 40. ML Pipeline

A training pipeline can look like:

```text
Data Ingestion
      ↓
Data Validation
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Train / Validation Split
      ↓
Training
      ↓
Evaluation
      ↓
Model Validation
      ↓
Model Registry
```

The pipeline should be repeatable rather than dependent on manual notebook execution.

---

# 41. Production Pipeline

The complete production flow becomes:

```text
                  DATA

                   ↓

             Data Pipeline
                   ↓
            Training Pipeline
                   ↓
            Experiment Tracking
                   ↓
             Model Evaluation
                   ↓
              Model Registry
                   ↓
                  CI/CD
                   ↓
              Deployment
                   ↓
             Model Serving
                   ↓
              Predictions
                   ↓
              Monitoring
                   ↓
          Drift / Performance
                   ↓
             Retraining
                   ↓
             New Model
                   ↓
              Deployment
                   ↓
                  ...
```

---

# 42. End-to-End MLOps Architecture

```text
                                  MLOps

                           +----------------+
                           |  Data Sources  |
                           +-------+--------+
                                   |
                                   v
                         +-------------------+
                         | Data Validation   |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Data Processing   |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Feature Pipeline  |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Model Training    |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Experiment Track  |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Model Evaluation  |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Model Registry    |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | CI / CD / CT      |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Model Deployment  |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | Model Serving     |
                         +---------+---------+
                                   |
                                   v
                              Predictions
                                   |
                                   v
                         +-------------------+
                         | Monitoring        |
                         +---------+---------+
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
               System Metrics                ML Metrics
                    |                             |
                    +--------------+--------------+
                                   |
                                   v
                           Drift Detection
                                   |
                                   v
                            Retraining
                                   |
                                   v
                           New Model Version
```

---

# 43. The Three Main MLOps Loops

I want to understand MLOps as three connected loops.

## Development Loop

```text
Experiment
   ↓
Train
   ↓
Evaluate
   ↓
Improve
   ↓
Experiment
```

## Deployment Loop

```text
Model
  ↓
Validate
  ↓
Package
  ↓
Deploy
  ↓
Rollback / Promote
```

## Production Loop

```text
Monitor
   ↓
Detect Change
   ↓
Retrain
   ↓
Evaluate
   ↓
Deploy
   ↓
Monitor
```

Together:

```text
Development
     ↓
Deployment
     ↓
Production
     ↓
Monitoring
     ↓
Retraining
     ↓
Development
```

---

# 44. What I Want to Explore

This repository is organized around practical questions.

```text
How does an ML model move from notebook to production?

How should training become reproducible?

How do we version datasets?

How do we track experiments?

How do we compare model versions?

How should models be registered?

How do we promote a model to production?

How do CI/CD pipelines work for ML?

What is Continuous Training?

How should a model be served?

When should we use online inference?

When should we use batch inference?

How do we deploy models safely?

How do canary deployments work?

How does rollback work?

How do we monitor a production model?

What is data drift?

What is concept drift?

How do we detect model degradation?

When should retraining happen?

How do we prevent training-serving skew?

How do we scale model inference?

How do we reduce inference cost?

How do we make ML pipelines reproducible?

How do we build a complete production ML lifecycle?
```

---

# 45. Repository Structure

```text
mlops-from-training-to-production/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── 01_ml_lifecycle/
│   ├── development/
│   ├── experimentation/
│   ├── training/
│   └── production/
│
├── 02_data/
│   ├── ingestion/
│   ├── validation/
│   ├── preprocessing/
│   ├── versioning/
│   └── lineage/
│
├── 03_features/
│   ├── feature_engineering/
│   ├── feature_store/
│   ├── offline/
│   ├── online/
│   └── training_serving_skew/
│
├── 04_training/
│   ├── pipelines/
│   ├── configuration/
│   ├── reproducibility/
│   ├── distributed_training/
│   └── checkpointing/
│
├── 05_experiment_tracking/
│   ├── experiments/
│   ├── parameters/
│   ├── metrics/
│   ├── artifacts/
│   └── lineage/
│
├── 06_model_evaluation/
│   ├── metrics/
│   ├── validation/
│   ├── quality_gates/
│   └── model_comparison/
│
├── 07_model_registry/
│   ├── versioning/
│   ├── metadata/
│   ├── promotion/
│   ├── approval/
│   └── rollback/
│
├── 08_ci_cd_ct/
│   ├── continuous_integration/
│   ├── continuous_delivery/
│   ├── continuous_deployment/
│   ├── continuous_training/
│   └── pipelines/
│
├── 09_serving/
│   ├── rest/
│   ├── grpc/
│   ├── online/
│   ├── batch/
│   └── streaming/
│
├── 10_deployment/
│   ├── docker/
│   ├── kubernetes/
│   ├── cloud/
│   ├── canary/
│   ├── blue_green/
│   ├── shadow/
│   └── rollback/
│
├── 11_monitoring/
│   ├── system/
│   ├── model/
│   ├── data/
│   ├── prediction/
│   ├── drift/
│   └── alerts/
│
├── 12_retraining/
│   ├── scheduled/
│   ├── trigger_based/
│   ├── performance_based/
│   └── pipelines/
│
├── 13_optimization/
│   ├── batching/
│   ├── quantization/
│   ├── compilation/
│   ├── caching/
│   └── inference/
│
├── 14_security/
│   ├── authentication/
│   ├── authorization/
│   ├── secrets/
│   ├── api_security/
│   └── container_security/
│
├── 15_testing/
│   ├── unit/
│   ├── data/
│   ├── model/
│   ├── api/
│   ├── integration/
│   └── deployment/
│
├── src/
│   └── mlops/
│       ├── data/
│       ├── features/
│       ├── training/
│       ├── evaluation/
│       ├── registry/
│       ├── serving/
│       ├── monitoring/
│       └── pipelines/
│
├── configs/
├── notebooks/
├── scripts/
├── pipelines/
├── tests/
├── experiments/
└── artifacts/
```

---

# 46. Technologies

The repository will use technologies according to the problem being explored.

### Machine Learning

```text
Python
NumPy
Pandas
Scikit-learn
PyTorch
```

### Experiment Tracking

```text
MLflow
Weights & Biases
```

### Data and Pipelines

```text
SQL
Data Validation
Feature Pipelines
Workflow Orchestration
```

### Packaging and Deployment

```text
Docker
FastAPI
Kubernetes
```

### Cloud

```text
AWS
GCP
Azure
```

### Monitoring

```text
Metrics
Logs
Traces
Model Monitoring
Data Drift Detection
```

The focus is not on memorizing tools.

The focus is understanding why each component exists and what problem it solves.

---

# 47. From Training to Production

The main learning path is:

```text
Machine Learning Model
        ↓
Reproducible Training
        ↓
Experiment Tracking
        ↓
Model Evaluation
        ↓
Model Registry
        ↓
Automated Pipelines
        ↓
CI/CD
        ↓
Containerization
        ↓
Model Serving
        ↓
Deployment
        ↓
Monitoring
        ↓
Drift Detection
        ↓
Retraining
        ↓
Continuous Improvement
```

---

# 48. Production Mindset

The main shift I want to make through this repository is:

```text
"I trained a model."
```

to:

```text
"I built a system that can repeatedly train,
validate, deploy, serve, monitor, and improve
a machine learning model."
```

A production ML system should answer:

```text
What data produced this model?

What code produced this model?

What configuration was used?

How was the model evaluated?

Where is the model stored?

Which version is currently serving traffic?

How can the deployment be rolled back?

Is the model still performing correctly?

Has the data changed?

When should the model be retrained?

Can the entire process be reproduced?
```

These are the problems this repository is designed to explore.

---

# 49. Final Goal

The final goal is to understand the complete machine learning lifecycle:

```text
                         MACHINE LEARNING

                              DATA
                               ↓
                         DATA PIPELINE
                               ↓
                         MODEL TRAINING
                               ↓
                      EXPERIMENT TRACKING
                               ↓
                       MODEL EVALUATION
                               ↓
                        MODEL REGISTRY
                               ↓
                         CI / CD / CT
                               ↓
                          DEPLOYMENT
                               ↓
                         MODEL SERVING
                               ↓
                          PREDICTION
                               ↓
                          MONITORING
                               ↓
                    DRIFT / PERFORMANCE
                               ↓
                          RETRAINING
                               ↓
                         NEW MODEL
                               ↓
                          DEPLOYMENT
                               ↓
                              ...
```

The objective is to understand MLOps not as a collection of tools, but as a complete engineering lifecycle for operating machine learning systems reliably in the real world.

This repository takes the journey from a simple trained model to a reproducible, deployable, observable, scalable, and continuously improving production ML system.
