from typing import NamedTuple

from kfp.dsl import (
    Dataset,
    Input,
    Model,
    Output,
    ClassificationMetrics,
    Metrics,
    component,
    Artifact,
)
from google.cloud.aiplatform.gapic import ModelEvaluation
# from google_cloud_pipeline_components.types.artifact_types import ClassificationMetrics #as ClassificationMetricsGCP
from config import config

@component(
    base_image=config.BASE_IMAGE
)
def model_evaluate(
    target_col: str,
    val_set: Input[Dataset],
    model: Input[Artifact],
    metrics_cls: Output[ClassificationMetrics],
    metrics: Output[Metrics],
    metrics_eval: Output[Artifact],
    # metrics_eval_2: Output[Artifact],
    model_name: str,
    project_id: str,
    location: str = 'europe-west1',
    force_upload_model: bool = False,    
) -> NamedTuple("output", [("upload_model", bool)]): # type: ignore
    import google.cloud.aiplatform as aip
    import pandas as pd
    import numpy as np
    from sklearn.metrics import (
        confusion_matrix, 
        roc_curve, 
        roc_auc_score, 
        accuracy_score, 
        auc, log_loss, 
        precision_recall_curve, 
        f1_score, 
        recall_score, 
        precision_score,
    )
    import logging
    import pickle
    # import json

    logging.info("Reading data...")
    val_df = pd.read_parquet(val_set.path)
    X = val_df.drop(columns=[target_col])
    y = val_df[target_col]

    logging.info("Reading model...")
    logging.info(f"model_path: {model.metadata['model_path']}")
    with open(model.metadata["model_path"], "rb") as f:
        model_pipeline = pickle.load(f)
    
    logging.info("Evaluating model...")
    all_feats = list(model_pipeline[0].get_feature_names_out())
    X = X[all_feats]
    X_2 = X.copy()
    
    y_scores = model_pipeline.predict_proba(X_2)[:, 1]
    y_pred = model_pipeline.predict(X)

    ## roc_auc_score
    roc_auc = roc_auc_score(y, y_pred)
    metrics.log_metric("roc_auc", float(roc_auc))
    logging.info(f"roc_auc: {float(roc_auc)}")

    ## accuracy_score
    accuracy = accuracy_score(y, y_pred)
    metrics.log_metric("accuracy", float(accuracy))
    logging.info(f"accuracy: {float(accuracy)}")

    ## auc_precision_recall
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    auc_precision_recall = auc(recall, precision)
    metrics.log_metric("auc_precision_recall", float(auc_precision_recall))
    logging.info(f"auc_precision_recall: {float(auc_precision_recall)}")

    ## auc_precision_recall
    log_loss_score = log_loss(y, y_pred)
    metrics.log_metric("log_loss_score", float(log_loss_score))
    logging.info(f"log_loss_score: {float(log_loss_score)}")
    
    # roc_curve
    # fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_scores, pos_label=True)
    # metrics_cls.log_roc_curve(fpr[0::100], tpr[0::100], thresholds[0::100])
    # logging.info(f"log_roc_curve: {thresholds[0:10].tolist()}")

    ## confusion_matrix
    confusion_matrix_values = confusion_matrix(y, y_pred).tolist()
    metrics_cls.log_confusion_matrix(
        ["False", "True"],
        confusion_matrix_values
    )
    logging.info(f"confusion_matrix: {confusion_matrix_values}")
    tn, fp, fn, tp = np.ravel(confusion_matrix_values)

    metrics_eval.metadata = {
        "auPrc": auc_precision_recall,
        "auRoc": roc_auc,
        "logLoss": log_loss_score,
        "confidenceMetrics": [{
            "confidenceThreshold": 0.5,
            "f1Score": f1_score(y, y_pred),
            # "f1ScoreAt1": 0.66,
            "falsePositiveCount": float(fp),
            "falsePositiveRate": float(fp/(fp + tn)),
            # "falsePositiveRateAt1": 1,
            "precision":  precision_score(y, y_pred),
            # "precisionAt1": 0.5,
            "recall": recall_score(y, y_pred),
            # "recallAt1": 1,
            "truePositiveCount": float(tp)
        }],
        # "confusionMatrix": {
        #     "annotationSpecs": [{"id": "0", "displayName": "0"}, {"id": "1", "displayName": "1"}],
        #     "rows": [{"row": confusion_matrix_values[0]}, {"row": confusion_matrix_values[1]}]
        # }
    }
    logging.info(f"metrics_eval.metadata: {metrics_eval.metadata}")

    aip.init(project=project_id, location=location)
    models = aip.Model.list(filter=f'display_name="{model_name}"', location=location)
    logging.info(f"previous_models count: {len(models)}")

    upload_model = False
    if len(models) > 0:
        previous_model = models[0]
        logging.info(f"previous_model.version_id: {previous_model.version_id}")
        last_model_eval = previous_model.list_model_evaluations()
        if len(last_model_eval) > 0:
            last_model_eval = last_model_eval[0]
            logging.info(f"last_model_eval.display_name: {last_model_eval.display_name}")
            prev_auroc = last_model_eval.gca_resource.metrics['auRoc']
            logging.info(f"prev_auroc: {prev_auroc}")
            if roc_auc > prev_auroc:
                logging.info(f"prev_auroc is less then current value")
                upload_model = True
        else:
            logging.info("No model evals")
    # previous_model.evaluate
    # model_registry = aiplatform.models.ModelRegistry(model=model_id)
    # versions = model_registry.list_versions()

    if force_upload_model:
        logging.info(f"force_upload_model: {force_upload_model}")
        upload_model = True

    return upload_model,