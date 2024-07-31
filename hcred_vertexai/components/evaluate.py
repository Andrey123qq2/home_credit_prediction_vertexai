from kfp.dsl import (
    Artifact,
    Dataset,
    Input,
    Model,
    Output,
    ClassificationMetrics,
    component,
    pipeline,
)

@component(
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    packages_to_install=['pyarrow', 'category_encoders', 'dill'],
)
def model_evaluate(
    target_col: str,
    test_set: Input[Dataset],
    # model: Input[Model],
    model: Input[Artifact],
    metrics: Output[ClassificationMetrics],
):
    from pathlib import Path
    import pandas as pd
    import joblib
    from sklearn.metrics import confusion_matrix, roc_curve
    import logging
    
    val_df = pd.read_parquet(test_set.path)
    X = val_df.drop(columns=[target_col])
    y = val_df[target_col]

    # model_pipeline = joblib.load(model.metadata["model_path"])

    import dill
    dill.settings['recurse'] = True

    logging.info(f"model_path: {model.metadata['model_path']}")
    with open(model.metadata["model_path"], 'rb') as file:
        model_pipeline = dill.load(
            file=file,
        )
    logging.info("model load completed")

    all_feats = list(model_pipeline[3].get_feature_names_out())
    X = X[all_feats]
    X_2 = X.copy()

    y_scores = model_pipeline.predict_proba(X_2)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_scores, pos_label=True)
    metrics.log_roc_curve(fpr.tolist()[0::15], tpr.tolist()[0::15], thresholds.tolist()[0::15])

    y_pred = model_pipeline.predict(X)
    metrics.log_confusion_matrix(
        ["False", "True"],
        confusion_matrix(y, y_pred).tolist(),
    )