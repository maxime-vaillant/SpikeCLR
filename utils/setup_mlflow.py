# MLflow configuration
import os
import mlflow
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

def setup_mlflow(experiment):
    """Setup MLflow tracking.

    Credentials and URI are read from the .env file at the project root.
    If MLFLOW_TRACKING_URI is not set, it falls back to 'local' (local filesystem tracking).
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "local")

    mlflow.config.enable_async_logging(True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
