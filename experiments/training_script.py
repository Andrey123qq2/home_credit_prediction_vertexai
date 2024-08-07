import gc
import logging

from google.cloud import aiplatform as aip
from sklearn.metrics import roc_auc_score, accuracy_score
from config import config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, TargetEncoder
from sklearn.linear_model import LogisticRegression
from  category_encoders import CatBoostEncoder
import pandas as pd
import pyarrow

from config import config
from processing import preprocessors as pp

def eval_model(model_pipeline):
    print("Evaluate model started")
    target_col = "target"
    all_feats = list(model_pipeline[0].get_feature_names_out())
    val_set_path = "gs://training_data_hcred-vertexai/pipeline-output/297198310072/gcp-hcred-20240805182500/load-bq-dataset_-8449677315348103168/dataset_val"

    val_df = pd.read_parquet(val_set_path)
    X = val_df.drop(columns=[target_col])
    X = X[all_feats]
    y = val_df[target_col]

    y_pred = model_pipeline.predict(X)

    roc_auc = roc_auc_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    return roc_auc, accuracy

def train_model(params):
    train_set_path = "gs://training_data_hcred-vertexai/pipeline-output/297198310072/gcp-hcred-20240805182500/load-bq-dataset_-8449677315348103168/dataset_train"
    all_feats = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES + config.DATE_FEATURES
    target_col = "target"

    model_pipeline = Pipeline(
        [
            ('TableDtypes Transformer', pp.TableDtypesTransformer()),
            ('Downcast Transformer', pp.DowncastTransformer()),
            ('Numerical Imputer', pp.NumericalImputer(variables=config.NUMERICAL_FEATURES)),
            ('Categorical Imputer', pp.CategoricalImputer(variables=config.CATEGORICAL_FEATURES)),
            ('Categorical Encoder', CatBoostEncoder(cols=config.CATEGORICAL_FEATURES)),
            # ('Categorical Encoder', pp.TargetEncoderCustom(cols=config.CATEGORICAL_FEATURES)),
            ('Dates Imputer', pp.DatesImputer(variables=config.DATE_FEATURES)),
            ('Dates Transformer', pp.DateColsTransformer(date_cols=config.DATE_FEATURES)),
            ('Log Transform', pp.Log1pTransformer(num_cols=config.NUMERICAL_FEATURES)),
            ('Scaler Transform', MinMaxScaler(copy=False)),
            ('Linear Model', LogisticRegression(**params))
        ], verbose=True
    )
    train_df = pd.read_parquet(train_set_path)
    print("train_df load completed")
    X = train_df[all_feats]
    y = train_df[target_col]
    del train_df
    gc.collect()

    model_pipeline.fit(X, y)
    return model_pipeline

params = dict(random_state=1, verbose=1, solver='liblinear', max_iter=100)
print("Train model started")
model = train_model(params)
roc_auc, accuracy = eval_model(model)
print('Model roc_auc:', roc_auc)

aip.log_params(params)
aip.log_metrics({"accuracy": accuracy, "roc_auc": roc_auc})