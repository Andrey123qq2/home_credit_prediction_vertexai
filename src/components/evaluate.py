from typing import NamedTuple

from kfp.dsl import (
    Artifact,
    Dataset,
    Input,
    Model,
    Output,
    ClassificationMetrics,
    Metrics,
    component,
    pipeline,
)

@component(
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    packages_to_install=['pyarrow', 'category_encoders', 'dill', 'google.cloud.aiplatform'],
)
def model_evaluate(
    target_col: str,
    # project_id: str,
    # location: str,
    # model_name: str,
    val_set: Input[Dataset],
    model: Input[Model],
    metrics_cls: Output[ClassificationMetrics],
    metrics: Output[Metrics],
    upload_model: bool = False
) -> NamedTuple("output", [("upload_model", str)]): # type: ignore
    
    import google.cloud.aiplatform as aip
    import pandas as pd
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
    import logging
    import dill

    logging.info("Reading data...")
    val_df = pd.read_parquet(val_set.path)
    X = val_df.drop(columns=[target_col])
    y = val_df[target_col]

    logging.info("Reading model...")
    logging.info(f"model_path: {model.path}")
    dill.settings['recurse'] = True
    with open(model.path, 'rb') as file:
        model_pipeline = dill.load(
            file=file,
        )
    
    logging.info("Evaluating model...")
    all_feats = list(model_pipeline[4].get_feature_names_out())
    X = X[all_feats]
    X_2 = X.copy()
    
    y_scores = model_pipeline.predict_proba(X_2)[:, 1]
    y_pred = model_pipeline.predict(X)

    # roc_auc_score
    roc_auc = roc_auc_score(y, y_pred)
    metrics.log_metric("roc_auc", float(roc_auc))
    logging.info(f"roc_auc: {float(roc_auc)}")

    # accuracy_score
    accuracy = accuracy_score(y, y_pred)
    metrics.log_metric("accuracy", float(accuracy))
    logging.info(f"accuracy: {float(accuracy)}")

    # roc_curve
    fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_scores, pos_label=True)
    # metrics_cls.log_roc_curve(fpr.tolist()[0::30], tpr.tolist()[0::30], thresholds.tolist()[0::30])
    logging.info(f"log_roc_curve: {fpr.tolist()[0:10]}")

    # confusion_matrix
    metrics_cls.log_confusion_matrix(
        ["False", "True"],
        confusion_matrix(y, y_pred).tolist(),
    )
    logging.info(f"confusion_matrix")

    # aip.init(project=project_id, location=location)
    # previous_model = aip.Model.list(filter=f'display_name="{model_name}"', location=location)[0]
    # logging.info(f"list_model_evaluations: {previous_model.list_model_evaluations()}")
    # logging.info(f"previous_model: {dir(previous_model)}")
    # logging.info(f"versioning_registry: {previous_model.versioning_registry}")
    # previous_model.evaluate
    # model_registry = aiplatform.models.ModelRegistry(model=model_id)
    # versions = model_registry.list_versions()

    if upload_model:
        upload_model = "true"
    else:
        upload_model = "false"
    return upload_model,