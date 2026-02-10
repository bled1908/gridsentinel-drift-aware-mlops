# Drift-Aware MLOps Pipeline

## 📌 Project Status
Early development – data preprocessing module implemented.

## 📂 Current Structure
- `data/` – raw and processed data directories
- `notebooks/` – exploratory analysis
- `src/` – core pipeline logic

## 🧠 Objective
To build an end-to-end MLOps pipeline with automated preprocessing,
drift detection, retraining, and monitoring.

## 🛠️ Next Steps
- Feature engineering module
- Model training pipeline
- Drift detection
- Experiment tracking

## 🔍 Drift Monitoring (In Progress)
The pipeline includes an initial drift detection and validation layer
to monitor changes in input data distributions and trigger model
retraining when necessary.


## 🔁 Policy-Based Retraining Experiments
The system evaluates multiple retraining policies under different
drift scenarios (holiday, seasonal, long-term) and logs performance
metrics and retraining events for comparative analysis.


⚠️ Project is under active development. Features will be added incrementally.
