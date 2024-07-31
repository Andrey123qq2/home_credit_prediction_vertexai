from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from  category_encoders import CatBoostEncoder
import pathlib
from pathlib import Path
import pandas as pd
import numpy as np    
import warnings
import joblib
import logging
import gc
import dill

dill.settings['recurse'] = True

# logging.info(f"model_path: {model.metadata['model_path']}")
model_file = "pipeline_hcred_model.pkl"
with open(model_file, 'rb') as file:
    model_pipeline = dill.load(
        file=file,
    )
print(model_pipeline)
