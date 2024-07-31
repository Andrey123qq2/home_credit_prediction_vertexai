PROJECT_ID = "hcred-vertexai"
BUCKET = "training_data_hcred-vertexai"
REGION = "europe-west1"
SERVICE_ACCOUNT = f"vertex-ai-service-account@{PROJECT_ID}.iam.gserviceaccount.com"
BUCKET = f"training_data_{PROJECT_ID}"
PIPELINE_ROOT_PATH = f"gs://{BUCKET}/pipeline-output/"
BQ_DATASET = "hcred_train"
ENDPOINT_ID = '4189161292059115520'