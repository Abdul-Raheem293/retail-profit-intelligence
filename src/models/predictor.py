import joblib
import pandas as pd


MODEL_PATH = "models/final_profit_prediction_pipeline.pkl"


def load_model():
    """Load the trained profit prediction pipeline."""
    return joblib.load(MODEL_PATH)


def predict_profit(model, input_data):
    """Predict profit for a transaction."""
    input_df = pd.DataFrame([input_data])
    return model.predict(input_df)[0]

