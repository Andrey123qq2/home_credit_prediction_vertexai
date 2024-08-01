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
    base_image="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
    # packages_to_install=['google-cloud-pipeline-components', 'google.cloud.aiplatform'],
    packages_to_install=['google.cloud.aiplatform'],
)
def deploy_to_endpoint(
    # model: Input[Artifact],
    model_name: str,
    model_resource_name: str,
    endpoint_display_name: str,
    project_id: str,
    location: str = 'europe-west1',
):
    # from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp, EndpointCreateOp
    import google.cloud.aiplatform as aip
    import logging

    logging.info(f"Deploy model")
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

    # model.resource_name = "test"
    # logging.info(f"dir: {dir(model)}")
    # logging.info(f"model: {model}")
    # logging.info(f"model.resource_name (deploy): {model.resource_name}")
    model = aip.Model(model_name=model_resource_name)
    target_endpoint.deploy(
        model=model,
        deployed_model_display_name=model_name,
        min_replica_count=1,
        max_replica_count=1,
        machine_type='n1-standard-4', 
        traffic_split={"0": 100},
    )

    # model.deploy(
    #     endpoint=target_endpoint,
    #     deployed_model_display_name=model_name,
    #     # traffic_percentage=traffic_percentage,
    #     traffic_split={"0": 100},
    #     machine_type='n1-standard-4',
    #     min_replica_count=1,
    #     max_replica_count=4,
    #     # explanation_metadata=explanation_metadata,
    #     # explanation_parameters=explanation_parameters,
    #     # metadata=metadata,
    #     # sync=sync,
    # )

    # model.wait()

    # print(model.display_name)
    # print(model.resource_name)

    # deployed_model = model.deploy(
    #     endpoint=endpoint,
    #     deployed_model_display_name=model_name,
    #     machine_type="n1-standard-4",
    #     min_replica_count=1,
    #     max_replica_count=1,
    #     traffic_split={"0": 100}
    # )
    # ModelDeployOp(
    #     endpoint=target_endpoint,
    #     model=model,
    #     deployed_model_display_name=model_name,
    #     dedicated_resources_machine_type="n1-standard-4",
    #     dedicated_resources_min_replica_count=1,
    # )

    logging.info(f"Model deployed to endpoint: {endpoint.display_name}")