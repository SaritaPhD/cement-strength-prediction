# cement-strength-prediction
## Overview
This project predicts the compressive strength of cement based on various features such as cement composition, water content, and age. The goal is to build a regression model that accurately predicts the cement's strength.

cement-strength-prediction/
│
├── README.md
├── poetry.lock
├── pyproject.toml
├── config.py
├── app.py
├── data/
│   └── cement_data.csv
├── notebook/
│   └── cement_strength_eda.ipynb
└── src/
    ├── __init__.py
    ├── eda.py
    ├── data_preprocessing.py
    ├── model_training.py
    ├── model_prediction.py

## Structure
- `src/` contains all the modules for EDA, data preprocessing, model training, and predictions.
- `notebook/` contains the original Jupyter notebook.
- `data/` contains the dataset.
- `config.py` includes configuration settings for the project.
- `app.py` is the Streamlit app for user interaction.

## Setup
This project uses Poetry for dependency management.

1. Install Poetry: `pip install poetry`
2. Install dependencies: `poetry install`

## Usage
Run the Streamlit app with the following command:

