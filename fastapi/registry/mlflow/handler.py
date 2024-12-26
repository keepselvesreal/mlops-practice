from mlflow.client import MlflowClient
import mlflow
from mlflow.pyfunc import PyFuncModel
from pprint import pprint
import logging # ts 추가

class MLFlowHandler:
    def __init__(self) -> None:
        # tracking_uri = "http://0.0.0.0:5001"
        tracking_uri = "http://mlflow-server:5000"
        self.tracking_uri = tracking_uri
        # self.artifact_root = "/mlflow/artifacts"
        
        self.client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
    
    def check_mlflow_health(self) -> None:
        try:
            experiments = self.client.search_experiments()   
            for rm in experiments:
                pprint(dict(rm), indent=4)
                return 'Service returning experiments'
        except:
            return 'Error calling MLFlow'
        
    def get_production_model(self, store_id: str) -> PyFuncModel:
        try:
            model_name = f"prophet-retail-forecaster-store-{store_id}"
            model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{model_name}/production",
                # dst_path="/mlflow/artifacts/tmp"  # 임시 저장 경로 지정
            )
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
            

# Handle this properly later ...
def check_mlflow_health():
    # client = MlflowClient(tracking_uri="http://0.0.0.0:5001")
    client = MlflowClient(tracking_uri="http://mlflow-server:5000")
    try:
        experiments = client.search_experiments()   
        for rm in experiments:
            pprint(dict(rm), indent=4)
        return 'Service returning experiments'
    except:
        return 'Error calling MLFlow'