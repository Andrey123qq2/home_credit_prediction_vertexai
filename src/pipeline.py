from datetime import datetime
from kfp import dsl
from kfp.dsl import pipeline, Condition

from config import config
from components import deploy, load_data, train, evaluate, upload

# from google_cloud_pipeline_components.preview.model_evaluation.model_evaluation_import_component import model_evaluation_import as ModelImportEvaluationOp
# from google_cloud_pipeline_components._implementation.model import GetVertexModelOp
# from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
# from google_cloud_pipeline_components.v1.model_evaluation import (
# from google_cloud_pipeline_components.v1.model_evaluation import ModelImportEvaluationOp
# from google_cloud_pipeline_components.aiplatform import ModelImportEvaluationOp
# from google_cloud_pipeline_components._implementation.model_evaluation import ModelImportEvaluationOp
#     EvaluationDataSamplerOp, 
#     GetVertexModelOp,
#     ModelEvaluationClassificationOp, 
#     ModelImportEvaluationOp,
#     TargetFieldDataRemoverOp
# )

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")

@pipeline(name=config.PIPELINE_NAME, pipeline_root=config.PIPELINE_ROOT_PATH)
def pipeline(
        model_name: str,
        serving_container_image_uri: str,
        endpoint_display_name: str,
        force_default: bool = False,
        model_params: dict = None,
        force_upload_model: bool = False,
        model_description: str = "",
):
    data_op = load_data.load_bq_dataset(
        project_id=config.PROJECT_ID, 
        bq_dataset=config.BQ_DATASET
    )

    model_train_op = train.model_train(
        model_name=model_name,
        target_col=config.TARGET_COLUMN,
        model_params=model_params,
        train_set=data_op.outputs["dataset_train"],
        serving_container_image_uri=serving_container_image_uri,
    )

    model_evaluate_op = evaluate.model_evaluate(
        target_col=config.TARGET_COLUMN,
        val_set=data_op.outputs["dataset_val"],
        model=model_train_op.outputs["model"],
        force_upload_model=force_upload_model,
        project_id=config.PROJECT_ID,
        location=config.REGION,
        model_name=model_name,
    )

    with Condition(
        model_evaluate_op.outputs['upload_model'] == True,
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
        )

        import_metrics_op = upload.import_metrics(
            model_resource_name=upload_op.outputs['resource_name'],
            metrics_eval = model_evaluate_op.outputs["metrics_eval"],
            region = config.REGION,
        )

        deploy_op = deploy.deploy_to_endpoint(
            model_resource_name = upload_op.outputs['resource_name'],
            model_name = model_name,
            endpoint_display_name = endpoint_display_name,
            project_id = config.PROJECT_ID,
            location = config.REGION,
        )

        ##
        ## Replaced by importing metrics calculated in model_evaluate_op step, and because of very slow processing of steps below 
        ##
        # get_model_op = GetVertexModelOp(
        #     model_name=upload_op.outputs['resource_name'],
        #     model_version='default',
        # )

        # upload_to_bq_op = load_data.upload_to_bq(
        #     project_id=config.PROJECT_ID,
        #     location=config.REGION,
        #     dest_bq_dataset=config.BQ_DATASET,
        #     dest_table_id=config.VAL_BQ_TABLE,
        #     parquet_data=data_op.outputs["dataset_val"],
        # )

        # batch_predict_op = ModelBatchPredictOp(
        #     project=config.PROJECT_ID,
        #     location=config.REGION,
        #     model=get_model_op.outputs['model'],
        #     job_display_name=f"bp-{config.PIPELINE_NAME}-{TIMESTAMP}",
        #     bigquery_source_input_uri=upload_to_bq_op.outputs['bq_table_uri'],
        #     instances_format="bigquery",
        #     predictions_format="bigquery",
        #     bigquery_destination_output_uri=f"bq://{config.PROJECT_ID}.{config.BQ_OUTPUT_DATASET_ID}.{config.PIPELINE_NAME}-bp-{TIMESTAMP}",
        #     excluded_fields=[config.TARGET_COLUMN],
        #     machine_type="n1-standard-8",
        #     starting_replica_count=2,
        #     max_replica_count=8,
        #     instance_type='object',
        #     manual_batch_tuning_parameters_batch_size=config.BATCH_PREDICTION_BATCH_SIZE,
        # )

        # eval_task = ModelEvaluationClassificationOp(
        #     project=config.PROJECT_ID,
        #     location=config.REGION,
        #     class_labels=["0", "1"],
        #     prediction_score_column="prediction",
        #     target_field_name=config.TARGET_COLUMN,
        #     ground_truth_format="bigquery",
        #     ground_truth_bigquery_source=upload_to_bq_op.outputs['bq_table_uri'],
        #     predictions_format="bigquery",
        #     predictions_bigquery_source=batch_predict_op.outputs['bigquery_output_table'],
        #     dataflow_service_account=config.SERVICE_ACCOUNT,
        #     # dataflow_subnetwork=DATAFLOW_NETWORK,
        #     dataflow_use_public_ips=False,
        #     force_runner_mode='Dataflow'
        # )

        # import_evaluation_task = ModelImportEvaluationOp(
        #     classification_metrics=eval_task.outputs["evaluation_metrics"],
        #     model=get_model_op.outputs['model'],
        #     # dataset_type="bigquery",
        # )