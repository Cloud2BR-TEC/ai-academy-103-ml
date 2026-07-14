import logging
import joblib
import numpy as np
from azureml.core.model import Model

# Configure logging at INFO level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def init():
    """
    Initialize the global model variable.
    This function is called once when the service starts.
    """
    global model
    try:
        model_path = Model.get_model_path("my_model_RegressionModel")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully from: %s", model_path)
    except Exception as e:
        logger.error("Failed to load model due to: %s", e)
        raise

def run(data: dict) -> dict:
    """
    Run scoring on the input data.

    Parameters:
      data (dict): A dictionary containing the key "data" mapped to the input features.
    
    Returns:
      dict: A dictionary with key 'result' containing model predictions,
            or key 'error' with an error message.
    """
    try:
        # Validate input
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary with key 'data'.")
        raw_data = data.get("data")
        if raw_data is None:
            raise ValueError("Missing 'data' key in input.")
        
        # Convert to numpy array
        input_data = np.array(raw_data)
        logger.info("Input data converted to numpy array with shape: %s", input_data.shape)
        
        # Generate predictions
        predictions = model.predict(input_data)
        logger.info("Prediction successful.")
        
        # Return results in a structured format
        return {"result": predictions.tolist()}
    except Exception as e:
        logger.error("Error during run: %s", e)
        return {"error": str(e)}
