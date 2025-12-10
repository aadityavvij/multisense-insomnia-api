from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <- allowed origin(s)
    allow_credentials=True,                   # allow cookies/auth if needed
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],                       # or list specific headers
)

with open('feature_statistics.json', 'r') as f:
    feature_stats = json.load(f)


# Model feature order from training time
model_feature_order = [
    "wavelet_mean",              # 0

    "sample_entropy",            # 1 → eeg_sample_entropy
    "shannon_entropy",           # 2 → eeg_shannon_entropy
    "rms",                       # 3 → eeg_rms

    "sample_entropy",            # 4 → ecg_sample_entropy
    "shannon_entropy",           # 5 → ecg_shannon_entropy
    "rms",                       # 6 → ecg_rms

    "mean_rr",                   # 7
    "sdnn",                      # 8
    "rmssd",                     # 9
    "mean_hr",                   # 10

    "sample_entropy",            # 11 → eog_sample_entropy
    "shannon_entropy",           # 12 → eog_shannon_entropy
    "rms",                       # 13 → eog_rms
    "zero_crossing_rate",        # 14

    "sample_entropy",            # 15 → emg_sample_entropy
    "shannon_entropy",           # 16 → emg_shannon_entropy
    "rms",                       # 17 → emg_rms

    "emg_activity",              # 18
    "emg_mobility",              # 19
    "emg_complexity"             # 20
]

frontend_to_model_index = {
    "wavelet_mean": 0,

    "eeg_sample_entropy": 1,
    "eeg_shannon_entropy": 2,
    "eeg_rms": 3,

    "ecg_sample_entropy": 4,
    "ecg_shannon_entropy": 5,
    "ecg_rms": 6,

    "mean_rr": 7,
    "sdnn": 8,
    "rmssd": 9,
    "mean_hr": 10,

    "eog_sample_entropy": 11,
    "eog_shannon_entropy": 12,
    "eog_rms": 13,
    "zero_crossing_rate": 14,

    "emg_sample_entropy": 15,
    "emg_shannon_entropy": 16,
    "emg_rms": 17,

    "emg_activity": 18,
    "emg_mobility": 19,
    "emg_complexity": 20
}




feature_alias_map = {
    "wavelet_mean": "wavelet_mean",

    "eeg_sample_entropy": "sample_entropy",
    "eeg_shannon_entropy": "shannon_entropy",
    "eeg_rms": "rms",

    "ecg_sample_entropy": "sample_entropy",
    "ecg_shannon_entropy": "shannon_entropy",
    "ecg_rms": "rms",

    "mean_rr": "mean_rr",
    "sdnn": "sdnn",
    "rmssd": "rmssd",
    "mean_hr": "mean_hr",

    "eog_sample_entropy": "sample_entropy",
    "eog_shannon_entropy": "shannon_entropy",
    "eog_rms": "rms",
    "zero_crossing_rate": "zero_crossing_rate",

    "emg_sample_entropy": "sample_entropy",
    "emg_shannon_entropy": "shannon_entropy",
    "emg_rms": "rms",

    "emg_activity": "emg_activity",
    "emg_mobility": "emg_mobility",
    "emg_complexity": "emg_complexity"
}

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

def generate_synthetic_training_data(stats, feature_order, n_samples=500):
    synthetic = []

    for _ in range(n_samples):
        row = []

        for feature in feature_order:

            # Determine correct base feature for duplicates
            base_feature = feature
            if feature in ["sample_entropy", "shannon_entropy", "rms"]:
                base_feature = feature  # direct match
            # For duplicated names (EEG/ECG/EOG/EMG), LIME only needs distribution, so reuse same stats.
            # e.g., eeg_sample_entropy -> sample_entropy
            if "sample_entropy" in feature:
                base_feature = "sample_entropy"
            elif "shannon_entropy" in feature:
                base_feature = "shannon_entropy"
            elif feature.endswith("rms") or feature == "rms":
                base_feature = "rms"

            mean = stats["mean"].get(base_feature, 0)
            std = stats["std"].get(base_feature, 0.001)  # small fallback

            value = np.random.normal(mean, std)
            row.append(value)

        synthetic.append(row)

    return np.array(synthetic)


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
    explanation: Dict[str, List[Dict[str, str | float]]]
    missing_features_filled: List[str]

# Feature defaults will be loaded from feature_statistics.json
# feature_defaults = {}

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
    try:
        explainer = LimeTabularExplainer(
            training_data=synthetic_training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )

        exp = explainer.explain_instance(
            data_row=input_df.iloc[0].values,
            predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names)),
            num_features=len(feature_names)
        )

        explanation = {
            "top_positive_features": [],
            "top_negative_features": [],
            "all_features": []
        }

        for feature_desc, weight in exp.as_list():
            explanation["all_features"].append({
                "feature": feature_desc,
                "weight": float(weight)
            })

            if weight > 0:
                explanation["top_positive_features"].append({"feature": feature_desc, "weight": float(weight)})
            else:
                explanation["top_negative_features"].append({"feature": feature_desc, "weight": float(weight)})

        # sort and trim
        explanation["top_positive_features"] = sorted(
            explanation["top_positive_features"], key=lambda x: abs(x["weight"]), reverse=True
        )[:5]

        explanation["top_negative_features"] = sorted(
            explanation["top_negative_features"], key=lambda x: abs(x["weight"]), reverse=True
        )[:5]

        return explanation

    except Exception as e:
        print("LIME ERROR:", e)
        return {"error": str(e)}


synthetic_training_data = generate_synthetic_training_data(
    feature_stats,
    model_feature_order,
    n_samples=500
)

@app.post("/predict", response_model=PredictionResponse)
async def predict_insomnia(request: PredictionRequest):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        raw_data = request.model_dump()

        # ---- BUILD MODEL INPUT ROW IN CORRECT TRAINING ORDER ----
        row = [None] * len(model_feature_order)

        for key, value in raw_data.items():
            if value is not None and key in frontend_to_model_index:
                idx = frontend_to_model_index[key]
                row[idx] = value

        # ---- FILL MISSING VALUES WITH MEDIANS ----
        missing_features = []
        for i in range(len(row)):
            if row[i] is None:
                feature_name = model_feature_order[i]
                row[i] = feature_defaults.get(feature_name, 0.0)
                missing_features.append(feature_name)

        # ---- BUILD DATAFRAME IN EXACT TRAINING ORDER ----
        input_df = pd.DataFrame([row], columns=model_feature_order)

        # ---- MAKE PREDICTION ----
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # ---- LIME SHOULD USE THE SAME FEATURE ORDER ----
        explanation = generate_lime_explanation(model, input_df, model_feature_order)

        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Insomnia" if prediction == 1 else "Normal",
            probability_normal=float(probabilities[0]),
            probability_insomnia=float(probabilities[1]),
            confidence=float(max(probabilities)),
            explanation=explanation,
            missing_features_filled=missing_features
        )

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