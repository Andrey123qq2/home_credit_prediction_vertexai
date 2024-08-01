from kfp import dsl
from kfp.dsl import pipeline, Condition

from config import config
from components import deploy, load_data, train, evaluate, upload
import logging

@pipeline(name="gcp-hcred-v1", pipeline_root=config.PIPELINE_ROOT_PATH)
def pipeline(
        model_name: str,
        target_col: str,
        serving_container_image_uri: str,
        endpoint_display_name: str,
        force_default: bool = False,
        cols_to_exclude: list = None,
        model_params: dict = None,
        upload_model: bool = False,
        model_description: str = "",
):
    data_op = load_data.load_bq_dataset(
        project_id=config.PROJECT_ID, 
        bq_dataset=config.BQ_DATASET
    ) #.set_display_name("Load And Split Data")

    model_train_op = train.model_train(
        model_name=model_name,
        target_col=target_col,
        model_params=model_params,
        train_set=data_op.outputs["dataset_train"],
        serving_container_image_uri=serving_container_image_uri,
    ) #.set_display_name("Train Model")

    model_evaluate_op = evaluate.model_evaluate(
        target_col=target_col,
        val_set=data_op.outputs["dataset_val"],
        model=model_train_op.outputs["model"],
        upload_model=upload_model,
        # project_id=config.PROJECT_ID,
        # location=config.REGION,
        # model_name=model_name,
    ) #.set_display_name("Model Card Graphics")

    with Condition(
        model_evaluate_op.outputs['upload_model'] == "true",
        name="upload-model-condition"
    ):
        upload_op = upload.upload_model(
            project_id = config.PROJECT_ID,
            region = config.REGION,
            model = model_train_op.outputs['model'],
            model_name = model_name,
            serving_image = serving_container_image_uri,
            run = dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
            run_id = dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            model_description = model_description,
            default = force_default,
        ) #.set_display_name("Upload Model")

        # logging.info(f"model.resource_name (pipeline): {upload_op.outputs['uploaded_model'].resource_name}")
        deploy_op = deploy.deploy_to_endpoint(
            # model = upload_op.outputs['uploaded_model'],
            model_resource_name = upload_op.outputs['resource_name'],
            model_name = model_name,
            endpoint_display_name = endpoint_display_name,
            project_id = config.PROJECT_ID,
            location = config.REGION,
        ) #.set_display_name("Deploy Model and Create Vertex AI Endpoint")