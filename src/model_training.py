import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import streamlit as st

def mean_squared_error(y_true, y_pred, squared=True):
    """Mean squared error regression loss"""
    if squared:
        return np.mean((y_true - y_pred) ** 2)
    else:
        return np.mean(np.abs(y_true - y_pred))


def mean_squared_log_error(y_true, y_pred):
    """Mean squared logarithmic error regression loss"""
    return np.mean((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2)


def train_and_evaluate(xtrain, ytrain, xtest, ytest, preprocessor, save_model_path=None):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1),
        'Lasso Regression': Lasso(alpha=1),
        'Random Forest Regression': RandomForestRegressor(max_depth=5),
        'Gradient Boosting Regression': GradientBoostingRegressor(learning_rate=0.1)
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(xtrain, ytrain)
        y_pred = pipeline.predict(xtest)

        mse = mean_squared_error(ytest, y_pred)
        msle = mean_squared_log_error(ytest, y_pred)
        r2 = r2_score(ytest, y_pred)

        results[name] = {
            'MSE': mse,
            'MSLE': msle,
            'R2': r2
        }

        if save_model_path:
            model_filename = f"{save_model_path}/{name.replace(' ', '_').replace('Regression', '')}_model.joblib"
            joblib.dump(pipeline, model_filename)

    return pd.DataFrame(results).transpose()
