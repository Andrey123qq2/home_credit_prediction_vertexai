from kfp.dsl import (
    Dataset,
    Output,
    component,
    Input,
)
from google_cloud_pipeline_components.types.artifact_types import BQTable
from typing import NamedTuple

from config import config


@component(
    base_image=config.BASE_IMAGE,
)
def load_bq_dataset(
    project_id: str,
    bq_dataset: str,
    dataset_train: Output[Dataset],
    dataset_val: Output[Dataset],
    test_size: float = 0.15,
    seed: int = 42,
):
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split
    import logging

    training_query = f"""
        SELECT * FROM `{project_id}.{bq_dataset}.train_base` as tb
        LEFT JOIN `{project_id}.{bq_dataset}.train_static_0` as ts ON tb.case_id = ts.case_id
        LEFT JOIN `{project_id}.{bq_dataset}.train_static_cb_0` as tscb ON ts.case_id = tscb.case_id
    """
        
    logging.info(f"Pulling data from {bq_dataset}")
    client = bigquery.Client(project=project_id)
    full_train_query = client.query(training_query)
    logging.info("bQuery completed")
    full_train_df = full_train_query.to_dataframe()
    logging.info("bQuery to dataframe completed")

    train_df, val_df = train_test_split(full_train_df, random_state=seed, test_size=test_size)

    train_df.to_parquet(dataset_train.path, index=False)
    logging.info("train_df save completed")
    val_df.to_parquet(dataset_val.path, index=False)
    logging.info("val_df save completed")


@component(
    base_image=config.BASE_IMAGE,
)
def upload_to_bq(
    project_id: str,
    location: str,
    dest_bq_dataset: str,
    dest_table_id: str, 
    parquet_data: Input[Dataset],
    bq_table: Output[BQTable],
    force_upload: bool = False,
) -> NamedTuple('outputs', [('bq_table_uri', str)]): # type: ignore

    from collections import namedtuple
    import logging
    import pandas as pd
    import pandas_gbq
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound

    client = bigquery.Client(project_id)
    dataset = client.dataset(dest_bq_dataset)
    table_ref = dataset.table(dest_table_id)

    def tbl_exists(client, table_ref):
        try:
            client.get_table(table_ref)
            return True
        except NotFound:
            return False
        
    bq_table.metadata["projectId"] = project_id
    bq_table.metadata["datasetId"] = dest_bq_dataset
    bq_table.metadata["tableId"] = dest_table_id
    logging.info(f"BQ table: {bq_table}\nmetadata: {bq_table.metadata}")

    logging.info(f"Reading data from {parquet_data.path}")
    df = pd.read_parquet(parquet_data.path)

    dest_table = f'{dest_bq_dataset}.{dest_table_id}'
    

    if force_upload or not tbl_exists(client, table_ref):
        client.delete_table(dest_table, not_found_ok=True)
        logging.info(f"Writing to {dest_table}")
        pandas_gbq.to_gbq(
            df,
            destination_table=f"{dest_table}", 
            project_id=project_id, 
            location=location
        )
    else:
        logging.info(f"Table {dest_table} already exists")

    t = namedtuple('outputs', ['bq_table_uri'])
    return t(f'bq://{project_id}.{dest_bq_dataset}.{dest_table_id}')