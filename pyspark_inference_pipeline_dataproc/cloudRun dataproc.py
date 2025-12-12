# main.py
import functions_framework
from google.cloud import dataproc_v1 as dataproc

@functions_framework.http
def trigger_dataproc_job(request):  # <-- added request here
    PROJECT_ID = 'involuted-forge-456406-a1'
    REGION = 'us-central1'
    CLUSTER_NAME = 'cluster-53b3'

    job_client = dataproc.JobControllerClient(
        client_options={"api_endpoint": f"{REGION}-dataproc.googleapis.com:443"}
    )

    job = {
        "placement": {"cluster_name": CLUSTER_NAME},
        "pyspark_job": {
            "main_python_file_uri": "gs://dataproc_pythonfiles/dataproc_pysparkpipeline.py"
        },
    }

    operation = job_client.submit_job_as_operation(
        request={"project_id": PROJECT_ID, "region": REGION, "job": job}
    )

    response = operation.result()
    return f"Job finished with status: {response.status.state.name}", 200