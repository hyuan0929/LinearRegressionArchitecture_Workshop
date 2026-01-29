# Linear Regression Architecture Workshop

This repository contains the submission for the **Linear Regression Architecture Workshop**
in **CSCN8010 – Foundations of Machine Learning Frameworks**.

The focus of this workshop is **architecture, modularity, and reproducibility**, not full
model implementation.

## Project Goals

- Demonstrate univariate linear regression architecture
- Separate data loading, preprocessing, modeling, and evaluation concerns
- Use configuration-driven experimentation
- Prepare a reusable structure for future ML experiments

## Project Structure
···
LinearRegressionArchitecture_Workshop/
├── configs/
│ └── experiment_config.yaml # Central experiment configuration
│
├── data/
│ ├── processed/ # Preprocessed datasets
│ └── raw/ # Raw input data (CSV source)
│
├── experiments/
│ └── results.csv # Experiment results / metrics tracking
│
├── notebooks/
│ ├── EDA.ipynb # Exploratory data analysis
│ └── linear_regression.ipynb # Training & visualization notebook
│
├── src/
│ ├── data_loader.py # Data ingestion logic
│ ├── evaluation.py # Model evaluation metrics
│ ├── model.py # Linear regression implementation
│ └── preprocessing.py # Data cleaning & preparation
│
├── README.md # Project documentation
└── requirements.txt # Python dependencies
···

## Configuration-Driven Design

All experiment parameters are defined in:

configs/experiment_config.yaml

This includes:
Dataset definitions (California, Ontario placeholder)
Feature and target selection
Preprocessing options
Training hyperparameters
Evaluation metrics
Experiment tracking paths

## Experiment Tracking

Experiment results are designed to be stored in:

experiments/results.csv

This enables:
Reproducibility
Metric comparison across runs
Extension to future experiment logging systems

At this stage, `results.csv` is a structural placeholder, as full experiment execution
is outside the scope of this workshop.

## Notes on Implementation Scope

Full data ingestion and model training are **not required**
Placeholder logic is intentionally used to demonstrate architecture readiness
Ontario housing data is represented as a future API/database extension

This repository is structured to support a future implementation sprint.

## Author

**Haibo Yuan**  
CSCN8010 – Conestoga College









