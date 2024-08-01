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
import pickle


model_file = "vertex_ai_model.pkl"
with open(model_file, 'rb') as file:
    model_pipeline = dill.load(
        file=file,
    )
# print(model_pipeline)

with open("vertex_ai_model_2.pkl", 'wb') as file:
    pickle.dump(
        model_pipeline,
        file
    )

# dill.settings['recurse'] = True

# # logging.info(f"model_path: {model.metadata['model_path']}")
# model_file = "vertex_ai_model.pkl"
# with open(model_file, 'rb') as file:
#     model_pipeline = dill.load(
#         file=file,
#     )
# print(model_pipeline)

# test = "gs://cloud-samples-data/vertex-ai/google-cloud-aiplatform-ci-artifacts/models/safe_driver/model"
# print('/'.join(test.split('/')[:-1]))