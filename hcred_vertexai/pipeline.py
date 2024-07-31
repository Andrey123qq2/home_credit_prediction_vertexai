from kfp.dsl import pipeline

from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.custom_job.utils import create_custom_training_job_op_from_component

from config import config
from components import load_data, train, evaluate

@pipeline(name="gcp-hcred-v1", pipeline_root=config.PIPELINE_ROOT_PATH)
def pipeline(
        model_name: str,
        target_col: str,
        force_default: bool = False,
        cols_to_exclude: list = None,
        model_params: dict = None,
        upload_model: bool = False,
        model_description: str = None,
        serving_container: dict = None
):
    data_op = load_data.load_bq_dataset(project_id=config.PROJECT_ID, bq_dataset=config.BQ_DATASET)

    custom_job_distributed_training_op = create_custom_training_job_op_from_component(
        train.model_train, 
        replica_count=1, 
    )

    model_train_op = custom_job_distributed_training_op(
        model_name=model_name,
        target_col=target_col,
        model_params=model_params,
        train_set=data_op.outputs["dataset_train"],
        location=config.REGION,
    )

    model_evaluate_op = evaluate.model_evaluate(
        target_col=target_col,
        test_set=data_op.outputs["dataset_test"],
        model=model_train_op.outputs["model"],
    )

    model_upload_op = ModelUploadOp(
        project=config.PROJECT_ID,
        location=config.REGION,
        display_name="hcred-model",
        unmanaged_container_model=model_train_op.outputs["model"],
    ).after(model_evaluate_op)

    endpoint_create_op = EndpointCreateOp(
        project=config.PROJECT_ID,
        location=config.REGION,
        display_name="hcred-model-endpoint"
    )

    ModelDeployOp(
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=model_upload_op.outputs["model"],
        deployed_model_display_name="hcred-model",
        dedicated_resources_machine_type="n1-standard-4",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    )