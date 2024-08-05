import numpy as np
import pandas as pd
import pickle
from numpy import nan
import logging

from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils

import processing
import category_encoders

class CprPredictor(Predictor):
    def __init__(self):
        return
    
    def load(self, artifacts_uri: str):
        """Loads the preprocessor and model artifacts."""
        prediction_utils.download_model_artifacts(artifacts_uri)
        
        with open("model.pkl", "rb") as f:
            self._model = pickle.load(f)

    def predict(self, instances):
        """Performs prediction."""
        inputs = pd.DataFrame(instances["instances"])
        print(f"inputs columns2: {inputs.columns}")
        print(f"inputs shape2: {inputs.shape}")
        all_feats = self._model[0].get_feature_names_out()
        inputs = inputs[all_feats]
        outputs = self._model.predict_proba(inputs)[:, 1].tolist()

        return {"predictions": outputs}