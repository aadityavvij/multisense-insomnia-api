from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import joblib
import json
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

app = FastAPI(
    title="Insomnia Detection API",
    description="API for predicting insomnia using physiological signals with explainable AI",
    version="1.0.0"
)

# Load the model, features, and statistics at startup
try:
    model = joblib.load('random_forest_insomnia_model.pkl')
    with open('model_features.json', 'r') as f:
        required_features = json.load(f)
    with open('feature_statistics.json', 'r') as f:
        feature_stats = json.load(f)
        feature_defaults = feature_stats['median']  # Use median values for missing features
    print("✓ Model, features, and statistics loaded successfully")
except Exception as e:
    print(f"Error loading files: {e}")
    model = None
    required_features = None
    feature_defaults = {}

# Define the request model with all features as optional
class PredictionRequest(BaseModel):
    wavelet_mean: Optional[float] = Field(None, description="EEG Wavelet Mean")
    eeg_sample_entropy: Optional[float] = Field(None, description="EEG Sample Entropy")
    eeg_shannon_entropy: Optional[float] = Field(None, description="EEG Shannon Entropy")
    eeg_rms: Optional[float] = Field(None, description="EEG RMS")
    ecg_sample_entropy: Optional[float] = Field(None, description="ECG Sample Entropy")
    ecg_shannon_entropy: Optional[float] = Field(None, description="ECG Shannon Entropy")
    ecg_rms: Optional[float] = Field(None, description="ECG RMS")
    mean_rr: Optional[float] = Field(None, description="Mean RR Interval")
    sdnn: Optional[float] = Field(None, description="SDNN")
    rmssd: Optional[float] = Field(None, description="RMSSD")
    mean_hr: Optional[float] = Field(None, description="Mean Heart Rate")
    eog_sample_entropy: Optional[float] = Field(None, description="EOG Sample Entropy")
    eog_shannon_entropy: Optional[float] = Field(None, description="EOG Shannon Entropy")
    eog_rms: Optional[float] = Field(None, description="EOG RMS")
    zero_crossing_rate: Optional[float] = Field(None, description="Zero Crossing Rate")
    emg_sample_entropy: Optional[float] = Field(None, description="EMG Sample Entropy")
    emg_shannon_entropy: Optional[float] = Field(None, description="EMG Shannon Entropy")
    emg_rms: Optional[float] = Field(None, description="EMG RMS")
    emg_activity: Optional[float] = Field(None, description="EMG Activity")
    emg_mobility: Optional[float] = Field(None, description="EMG Mobility")
    emg_complexity: Optional[float] = Field(None, description="EMG Complexity")
    
    class Config:
        json_schema_extra = {
            "example": {
                "wavelet_mean": 0.5,
                "eeg_sample_entropy": 1.2,
                "mean_rr": 800,
                "mean_hr": 75,
                "emg_activity": 0.3
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_normal: float
    probability_insomnia: float
    confidence: float
    explanation: Dict[str, List[Dict[str, float]]]
    missing_features_filled: List[str]

# Feature defaults will be loaded from feature_statistics.json
feature_defaults = {}

def fill_missing_features(input_data: dict, feature_list: list) -> tuple:
    """Fill missing features with median values from training data and track which were filled"""
    filled_features = []
    filled_data = {}
    
    for feature in feature_list:
        if feature in input_data and input_data[feature] is not None:
            filled_data[feature] = input_data[feature]
        else:
            # Use median from training data
            if feature in feature_defaults:
                filled_data[feature] = feature_defaults[feature]
            else:
                # Fallback to 0 if feature not in statistics (shouldn't happen)
                filled_data[feature] = 0.0
                print(f"Warning: Feature '{feature}' not found in statistics, using 0.0")
            filled_features.append(feature)
    
    return filled_data, filled_features

def generate_lime_explanation(model, input_df, feature_names, class_names=['Normal', 'Insomnia']):
    """Generate LIME explanation for the prediction"""
    try:
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            training_data=np.zeros((10, len(feature_names))),  # Dummy training data
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
        
        # Get explanation for the instance
        exp = explainer.explain_instance(
            data_row=input_df.values[0],
            predict_fn=model.predict_proba,
            num_features=len(feature_names)
        )
        
        # Format explanation
        explanation = {
            'top_positive_features': [],
            'top_negative_features': [],
            'all_features': []
        }
        
        # Get feature contributions
        feature_weights = exp.as_list()
        
        for feature_desc, weight in feature_weights:
            feature_info = {
                'feature': feature_desc,
                'weight': float(weight)
            }
            explanation['all_features'].append(feature_info)
            
            if weight > 0:
                explanation['top_positive_features'].append(feature_info)
            else:
                explanation['top_negative_features'].append(feature_info)
        
        # Sort by absolute weight
        explanation['top_positive_features'] = sorted(
            explanation['top_positive_features'], 
            key=lambda x: abs(x['weight']), 
            reverse=True
        )[:5]
        
        explanation['top_negative_features'] = sorted(
            explanation['top_negative_features'], 
            key=lambda x: abs(x['weight']), 
            reverse=True
        )[:5]
        
        return explanation
    
    except Exception as e:
        print(f"Error generating LIME explanation: {e}")
        return {
            'error': str(e),
            'top_positive_features': [],
            'top_negative_features': [],
            'all_features': []
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_insomnia(request: PredictionRequest):
    """
    Predict insomnia from physiological signal features.
    Missing features will be automatically filled with median values.
    """
    if model is None or required_features is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert request to dict
        input_data = request.dict()
        
        # Fill missing features
        filled_data, missing_features = fill_missing_features(input_data, required_features)
        
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([filled_data])[required_features]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Generate LIME explanation
        explanation = generate_lime_explanation(model, input_df, required_features)
        
        # Prepare response
        response = PredictionResponse(
            prediction=int(prediction),
            prediction_label="Insomnia" if prediction == 1 else "Normal",
            probability_normal=float(probabilities[0]),
            probability_insomnia=float(probabilities[1]),
            confidence=float(max(probabilities)),
            explanation=explanation,
            missing_features_filled=missing_features
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
async def get_features():
    """Get list of all features required by the model"""
    if required_features is None:
        raise HTTPException(status_code=500, detail="Features not loaded")
    
    return {
        "total_features": len(required_features),
        "features": required_features
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": required_features is not None
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Insomnia Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Make insomnia prediction",
            "GET /features": "Get required features",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)