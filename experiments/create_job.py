import time
from google.cloud import aiplatform as aip
from config import config

EXPERIMENT_NAME = 'pipe-lr-1'
BUCKET_URI_STAGING = "gs://hcred_experiments_staging"
TRAINING_JOBS_URI = f"{BUCKET_URI_STAGING}/jobs"
REQUIREMENTS = ['fsspec', 'gcsfs']

aip.init(experiment=EXPERIMENT_NAME, project=config.PROJECT_ID, location=config.REGION)
aip.autolog()

job = aip.CustomJob.from_local_script(
    display_name="pipe-lr-job",
    container_uri=config.BASE_IMAGE,
    script_path="experiments/training_script.py",
    enable_autolog=True,
    staging_bucket=TRAINING_JOBS_URI,
    machine_type="n1-standard-4",
    requirements=REQUIREMENTS,
)
job.run(experiment=EXPERIMENT_NAME, service_account=config.SERVICE_ACCOUNT)
time.sleep(2)
experiment_df = aip.get_experiment_df(experiment=EXPERIMENT_NAME)
print(experiment_df.T)