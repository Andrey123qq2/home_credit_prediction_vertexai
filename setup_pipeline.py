from google.cloud import aiplatform as aip
from kfp import compiler

from config import config
import pipeline as pl

compiler.Compiler().compile(pipeline_func=pl.pipeline, package_path="gcp-hcred-v1.json")

aip.init(project=config.PROJECT_ID, staging_bucket=config.BUCKET, location=config.REGION)

parameter_values = {
    "model_name": "hcred-model",
    "target_col": "target",
    "serving_container_image_uri": config.PRED_CONTAINER,
    "force_default": False,
    "cols_to_exclude": [],
    "model_params": dict(random_state=1, verbose=1, solver='liblinear', max_iter=100),
    "upload_model": True,
    "model_description": "",
    "endpoint_display_name": "hcred-model-endpoint",
}

job = aip.PipelineJob(
    display_name="gcp-hcred-v1",
    template_path="gcp-hcred-v1.json",
    pipeline_root=config.PIPELINE_ROOT_PATH,
    parameter_values=parameter_values,
    enable_caching=True,
)

job.run(service_account=config.SERVICE_ACCOUNT)