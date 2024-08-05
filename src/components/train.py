from kfp.dsl import (
    Dataset,
    Input,
    Model,
    Output,
    component,
    Artifact,
)
from config import config

@component(
    base_image=config.BASE_IMAGE
)
def model_train(
    model_name:str,
    target_col: str,
    train_set: Input[Dataset],
    model: Output[Artifact],
    serving_container_image_uri: str,
    model_params: dict = None,
):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, TargetEncoder
    from sklearn.linear_model import LogisticRegression
    from  category_encoders import CatBoostEncoder
    import pandas as pd
    import logging
    import pickle
    import pathlib
    import gc

    from config import config
    from processing import preprocessors as pp

    logging.info("Train job started")

    if model_params is None:
        model_params = {}

    import sklearn
    logging.info(sklearn.__version__)

    all_feats = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES + config.DATE_FEATURES
    train_df = pd.read_parquet(train_set.path)
    logging.info("train_df load completed")
    X = train_df[all_feats]
    y = train_df[target_col]

    del train_df
    gc.collect()

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
            ('Linear Model', LogisticRegression(**model_params))
        ], verbose=True
    )

    model_pipeline.fit(X, y)
    logging.info("model fit completed")
    
    pathlib.Path(model.path).mkdir()
    pickle_output_path = model.path + '/model.pkl'
    with open(pickle_output_path, "wb") as f:
        pickle.dump(model_pipeline, f)

    model.metadata = {
        "containerSpec": {"imageUri": serving_container_image_uri},
        "framework": "scikit-learn",
        "model_name": 'model',
        "model_path": pickle_output_path,
    }