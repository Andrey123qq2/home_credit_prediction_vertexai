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
    packages_to_install=['google-cloud-bigquery[all]'],
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
)
def load_bq_dataset(
    project_id: str,
    bq_dataset: str,
    dataset_train: Output[Dataset] = None,
    dataset_test: Output[Dataset] = None,
):
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split
    import logging

    SEED = 42
    TEST_SIZE=0.15
    # PROJECT_ID = "hcred-vertexai"
    # BQ_DATASET = "hcred_train"

    training_query = f"""
        SELECT * FROM `{project_id}.{bq_dataset}.train_base` as tb
        LEFT JOIN `{project_id}.{bq_dataset}.train_static_0` as ts ON tb.case_id = ts.case_id
        LEFT JOIN `{project_id}.{bq_dataset}.train_static_cb_0` as tscb ON ts.case_id = tscb.case_id
    """
        
    client = bigquery.Client(project=project_id)
    full_train_query = client.query(training_query)
    logging.info("bQuery completed")
    full_train_df = full_train_query.to_dataframe()
    logging.info("bQuery to dataframe completed")
    train_df, val_df = train_test_split(full_train_df, random_state=SEED, test_size=TEST_SIZE)

    train_df.to_parquet(dataset_train.path, index=False)
    logging.info("train_df save completed")
    val_df.to_parquet(dataset_test.path, index=False)
    logging.info("val_df save completed")