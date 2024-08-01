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
from typing import NamedTuple

@component(
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    packages_to_install=['google.cloud.aiplatform'],
)
def upload_model(
    model: Input[Model],
    # uploaded_model: Output[Model],
    model_name: str,
    serving_image: str,
    model_description: str,
    run: str,
    run_id: str,
    project_id: str,
    region: str = 'europe-west1',
    default: bool = False,
) -> NamedTuple("output", [("resource_name", str)]): # type: ignore
    import google.cloud.aiplatform as aip
    import numpy as np
    import logging

    logging.info(f"Upload model for run {run} and run ID {run_id}")

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
        artifact_uri='/'.join(model.path.split('/')[:-1]),
        serving_container_image_uri=serving_image,
        parent_model=parent_model,
        is_default_version=default,
        version_aliases=["last-training"],
        version_description=model_description,
    )

    logging.info(f"uploaded_model.resource_name: {uploaded_model.resource_name}")

    return uploaded_model.resource_name,
    # logging.info(f"model.metadata: {uploaded_model.metadata}")

    # uploaded_model.metadata['resource_name'] = uploaded_model.resource_name
    # uploaded_model.resource_name = f'{vertex_model.resource_name}@{vertex_model.version_id}'
    # uploaded_model.uri = f'https://{region}-aiplatform.googleapis.com/v1/{vertex_model.resource_name}@{vertex_model.version_id}'