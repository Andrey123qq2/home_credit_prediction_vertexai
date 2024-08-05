from kfp.dsl import (
    component,
    Input,
    Model,
    Artifact,
)
from google_cloud_pipeline_components.types.artifact_types import VertexModel
from config import config

@component(
    base_image=config.BASE_IMAGE
)
def deploy_to_endpoint(
    model_name: str,
    model_resource_name: str,
    endpoint_display_name: str,
    project_id: str,
    location: str = 'europe-west1',    
):
    import logging
    import google.cloud.aiplatform as aip
    
    logging.info(f"Deploy model 2")
    aip.init(project=project_id, location=location)

    target_endpoint = None
    for endpoint in aip.Endpoint.list(order_by="update_time desc"):
        if endpoint.display_name == endpoint_display_name:
            target_endpoint = endpoint
    
    if target_endpoint is None:
        target_endpoint = aip.Endpoint.create(
            project=project_id,
            display_name=endpoint_display_name,
            location=location,
        )
        logging.info(f"Created new endpoint: {target_endpoint.display_name}")
    else:
        logging.info(f"Using existing endpoint: {endpoint.display_name}")

    model = aip.Model(model_name=model_resource_name)

    deployed_models = target_endpoint.gca_resource.deployed_models
    model_already_deployed = False
    model_ids_to_undeploy = []
    for model_d in deployed_models:
        if model_d.model_version_id == model.version_id:
            model_already_deployed = True
        else:
            model_ids_to_undeploy.append(model_d.id)

    if not model_already_deployed:
        model.deploy(
            endpoint=target_endpoint,
            deployed_model_display_name=model_name,
            traffic_split={"0": 100},
            machine_type='n1-standard-4',
            min_replica_count=1,
            max_replica_count=1,
        )
        model.wait()
        logging.info(f"Model deployed to endpoint: {endpoint.display_name}")
        logging.info(f"deployed model display_name: {model.display_name}")
        logging.info(f"deployed model resource_name: {model.resource_name}")
        logging.info(f"deployed model version_id: {model.version_id}")
    else:
        logging.info(f"Model version_id: {model.version_id} already deployed")

    for mod_id in model_ids_to_undeploy:
        endpoint.undeploy(
            deployed_model_id=mod_id,
            sync=True
        )
        logging.info(f"Undeployed Existing Model Version ID: {mod_id}")