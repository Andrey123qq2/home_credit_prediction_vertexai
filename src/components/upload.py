from kfp.dsl import (
    Input,
    Model,
    component,
    Output,
    Artifact,
)
from google.cloud.aiplatform.gapic import ModelEvaluation
from typing import NamedTuple
from config import config

@component(
    base_image=config.BASE_IMAGE
    )
def upload_model(
    model: Input[Artifact],
    uploaded_model: Output[Model],
    model_name: str,
    serving_image: str,
    model_description: str,
    run: str,
    run_id: str,
    project_id: str,
    region: str = 'europe-west1',
    default: bool = False,
) -> NamedTuple("output", [("resource_name", str)]): # type: ignore
    from google.cloud import aiplatform as aip
    from google.cloud.aiplatform import gapic
    import numpy as np
    import logging

    logging.info(f"1 Upload model for run {run} and run ID {run_id}")

    parent_model = None
    models = np.array(aip.Model.list(filter=f'display_name="{model_name}"', location='europe-west1'))
    if len(models) > 0:
        update_time = np.array([model.update_time for model in models])
        parent_model = models[np.argsort(update_time)][-1]

    if parent_model is not None:
        parent_model = parent_model.resource_name
        logging.info(f"Using parent model")
        default = True
    else:
        default = True
        logging.info(f"No parent model")
    
    logging.info(f"model.path: {model.path}")
    uploaded_model = aip.Model.upload(
        project=project_id,
        location=region,
        display_name=model_name,
        artifact_uri=model.path,
        serving_container_image_uri=serving_image,
        parent_model=parent_model,
        is_default_version=default,
        version_aliases=["last-training"],
        version_description=model_description,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )

    logging.info(f"uploaded_model.resource_name: {uploaded_model.resource_name}")

    return uploaded_model.resource_name,


@component(
    base_image=config.BASE_IMAGE
)
def import_metrics(
    model_resource_name: str,
    metrics_eval: Input[Artifact],
    region: str = 'europe-west1',
) -> NamedTuple("output", [("resource_name", str)]): # type: ignore
    from google.cloud import aiplatform as aip
    from google.cloud.aiplatform import gapic
    import logging

    logging.info("Import metrics")
    logging.info(f"metrics_eval.metadata: {metrics_eval.metadata}")

    API_ENDPOINT = f"{region}-aiplatform.googleapis.com"
    model_eval = gapic.ModelEvaluation(
        display_name="eval",
        metrics_schema_uri="gs://google-cloud-aiplatform/schema/modelevaluation/classification_metrics_1.0.0.yaml",
        metrics=metrics_eval.metadata,
    )
    
    client = gapic.ModelServiceClient(
        client_options={"api_endpoint": API_ENDPOINT}
    )
    client.import_model_evaluation(
        parent=model_resource_name, 
        model_evaluation=model_eval
    )
    
    logging.info("Metrics imported")