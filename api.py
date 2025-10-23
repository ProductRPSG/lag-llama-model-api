"""
Lag-Llama REST API
A simple Flask API wrapper for Lag-Llama time series forecasting model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
from huggingface_hub import hf_hub_download
import warnings
import traceback
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Fix PyTorch weights loading issue with newer PyTorch versions
torch.serialization.add_safe_globals([
    'gluonts.torch.distributions.studentT.StudentTOutput',
    'gluonts.torch.distributions.implicit_quantile_network.ImplicitQuantileNetwork'
])

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from lag_llama.gluon.estimator import LagLlamaEstimator

app = Flask(__name__)
CORS(app)  # Enable CORS for JavaScript requests

# Global variables
model = None
predictor = None

def log_message(message):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_model():
    """Load the pre-trained Lag-Llama model from HuggingFace"""
    global model, predictor
    
    if predictor is not None:
        log_message("Model already loaded")
        return predictor
    
    try:
        log_message("Starting model download from HuggingFace...")
        
        # Download model from HuggingFace
        repo_id = "time-series-foundation-models/Lag-Llama"
        filename = "lag-llama.ckpt"
        
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        log_message(f"Model downloaded to: {ckpt_path}")
        
        log_message("Initializing Lag-Llama estimator...")
        
        # Fix PyTorch loading by setting weights_only=False for trusted checkpoint
        import torch._weights_only_unpickler
        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = safe_load
        
        # Create estimator with exact configuration from lag_llama.json
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=24,  # Default, will be updated per request
            context_length=32,     # From config
            input_size=1,
            n_layer=8,           # From config: 8 layers
            n_embd_per_head=16,  # From config: 16 dim per head 
            n_head=9,            # From config: 9 heads
            scaling="robust",    # From config
            rope_scaling=None,
            batch_size=1,        # Conservative for API usage
            time_feat=True,      # From config - this affects input size!
            trainer_kwargs={}
        )
        
        log_message("Creating predictor...")
        
        # Create the predictor
        predictor = estimator.create_predictor(
            estimator.create_transformation(),
            estimator.create_lightning_module()
        )
        
        # Restore original torch.load
        torch.load = original_load
        
        log_message("✅ Model loaded successfully!")
        return predictor
        
    except Exception as e:
        log_message(f"❌ Error loading model: {str(e)}")
        log_message(traceback.format_exc())
        return None

def validate_input_data(data):
    """Validate the input data format"""
    errors = []
    
    # Check if data is provided
    if not data:
        errors.append("No data provided")
        return errors
    
    # Check required fields
    if 'data' not in data:
        errors.append("Missing 'data' field with time series values")
    
    # Validate data array
    if 'data' in data:
        time_series = data['data']
        if not isinstance(time_series, list):
            errors.append("'data' must be a list of numbers")
        elif len(time_series) < 10:
            errors.append("'data' must contain at least 10 data points")
        elif not all(isinstance(x, (int, float)) for x in time_series):
            errors.append("All values in 'data' must be numbers")
    
    # Validate prediction_length
    if 'prediction_length' in data:
        pred_len = data['prediction_length']
        if not isinstance(pred_len, int) or pred_len <= 0:
            errors.append("'prediction_length' must be a positive integer")
        elif pred_len > 100:
            errors.append("'prediction_length' cannot exceed 100")
    
    # Validate frequency
    valid_frequencies = ['T', 'min', 'H', 'D', 'W', 'M', 'Q', 'Y']
    if 'frequency' in data:
        freq = data['frequency']
        if freq not in valid_frequencies:
            errors.append(f"'frequency' must be one of: {valid_frequencies}")
    
    # Validate timestamps if provided
    if 'timestamps' in data and data['timestamps']:
        timestamps = data['timestamps']
        if len(timestamps) != len(data.get('data', [])):
            errors.append("Number of timestamps must match number of data points")
    
    return errors

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_status = "loaded" if predictor is not None else "not_loaded"
        return jsonify({
            "status": "healthy", 
            "message": "Lag-Llama API is running",
            "model_status": model_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint for single time series
    
    Expected JSON format:
    {
        "data": [1.2, 1.5, 1.8, 2.1, ...],  // Time series values (required)
        "timestamps": ["2023-01-01", "2023-01-02", ...],  // Optional timestamps
        "prediction_length": 7,  // Number of steps to predict (default: 7)
        "frequency": "D",  // Frequency: D=daily, H=hourly, etc. (default: "D")
        "context_length": 32  // Optional: how much history to use (default: 32)
    }
    """
    try:
        log_message("Received prediction request")
        
        # Parse request data
        data = request.get_json()
        
        # Validate input
        validation_errors = validate_input_data(data)
        if validation_errors:
            return jsonify({
                "error": "Invalid input data",
                "details": validation_errors
            }), 400
        
        # Extract parameters
        time_series = data['data']
        prediction_length = data.get('prediction_length', 7)
        frequency = data.get('frequency', 'D')
        context_length = data.get('context_length', 32)
        timestamps = data.get('timestamps', None)
        item_id = data.get('item_id', 'series_001')
        
        log_message(f"Processing series with {len(time_series)} points, predicting {prediction_length} steps")
        
        # Load model if not already loaded
        current_predictor = load_model()
        if current_predictor is None:
            return jsonify({"error": "Failed to load model"}), 500
        
        # Prepare timestamps
        if timestamps:
            # Use provided timestamps
            if len(timestamps) != len(time_series):
                return jsonify({
                    "error": "Number of timestamps must match number of data points"
                }), 400
            
            start_timestamp = pd.to_datetime(timestamps[0])
            # Validate timestamp consistency with frequency
            try:
                pd.date_range(start=start_timestamp, periods=len(time_series), freq=frequency)
            except Exception:
                return jsonify({
                    "error": f"Timestamps don't match frequency '{frequency}'"
                }), 400
        else:
            # Generate timestamps based on frequency
            start_timestamp = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        
        log_message(f"Using frequency: {frequency}, start: {start_timestamp}")
        
        # Create GluonTS dataset format
        series_data = [{
            "target": time_series,
            "start": start_timestamp,
            "item_id": item_id
        }]
        
        # Create ListDataset
        dataset = ListDataset(series_data, freq=frequency)
        
        log_message("Created dataset, starting prediction...")
        
        # Update predictor's prediction length
        current_predictor.prediction_length = prediction_length
        
        # Make prediction
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=current_predictor,
            num_samples=100  # Number of prediction samples
        )
        
        forecasts = list(forecast_it)
        tss = list(ts_it)
        
        if len(forecasts) == 0:
            return jsonify({"error": "No forecasts generated"}), 500
        
        forecast = forecasts[0]
        
        log_message("Prediction completed, processing results...")
        
        # Extract predictions (mean forecast)
        predictions = forecast.mean.tolist()
        
        # Extract confidence intervals (quantiles)
        quantiles = {}
        for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
            quantiles[f"q{int(q*100)}"] = forecast.quantile(q).tolist()
        
        # Generate future timestamps
        future_timestamps = pd.date_range(
            start=start_timestamp,
            periods=len(time_series) + prediction_length,
            freq=frequency
        )[-prediction_length:]  # Get only the future timestamps
        
        # Prepare response
        response = {
            "success": True,
            "predictions": predictions,
            "future_timestamps": [ts.isoformat() for ts in future_timestamps],
            "quantiles": quantiles,
            "metadata": {
                "input_length": len(time_series),
                "prediction_length": prediction_length,
                "frequency": frequency,
                "context_length": context_length,
                "model_info": "Lag-Llama foundation model"
            }
        }
        
        log_message(f"✅ Prediction successful: {len(predictions)} forecasted values")
        return jsonify(response)
    
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        log_message(f"❌ {error_msg}")
        log_message(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple time series
    
    Expected JSON format:
    {
        "datasets": [
            {
                "name": "series1",
                "data": [1.2, 1.5, 1.8, ...],
                "timestamps": ["2023-01-01", "2023-01-02", ...],  // Optional
                "item_id": "series1"  // Optional
            },
            {
                "name": "series2", 
                "data": [2.1, 2.4, 2.7, ...],
                "timestamps": ["2023-01-01", "2023-01-02", ...],  // Optional
                "item_id": "series2"  // Optional
            }
        ],
        "prediction_length": 7,  // Applied to all series
        "frequency": "D"  // Applied to all series
    }
    """
    try:
        log_message("Received batch prediction request")
        
        # Parse request data
        data = request.get_json()
        
        if not data or 'datasets' not in data:
            return jsonify({"error": "No datasets provided. Include 'datasets' array."}), 400
        
        datasets = data['datasets']
        if not isinstance(datasets, list) or len(datasets) == 0:
            return jsonify({"error": "'datasets' must be a non-empty array"}), 400
        
        if len(datasets) > 50:  # Limit batch size
            return jsonify({"error": "Maximum 50 datasets allowed per batch"}), 400
        
        prediction_length = data.get('prediction_length', 7)
        frequency = data.get('frequency', 'D')
        context_length = data.get('context_length', 32)
        
        log_message(f"Processing batch of {len(datasets)} time series")
        
        # Load model if not already loaded
        current_predictor = load_model()
        if current_predictor is None:
            return jsonify({"error": "Failed to load model"}), 500
        
        results = {}
        errors = {}
        
        for i, dataset in enumerate(datasets):
            series_name = dataset.get('name', f'series_{i+1}')
            
            try:
                # Validate individual dataset
                if 'data' not in dataset:
                    errors[series_name] = "Missing 'data' field"
                    continue
                
                series_data = dataset['data']
                if not isinstance(series_data, list) or len(series_data) < 10:
                    errors[series_name] = "Data must be a list with at least 10 points"
                    continue
                
                if not all(isinstance(x, (int, float)) for x in series_data):
                    errors[series_name] = "All data values must be numbers"
                    continue
                
                # Extract series-specific parameters
                timestamps = dataset.get('timestamps', None)
                item_id = dataset.get('item_id', series_name)
                
                # Process individual series
                log_message(f"Processing {series_name}: {len(series_data)} points")
                
                # Prepare timestamps
                if timestamps:
                    if len(timestamps) != len(series_data):
                        errors[series_name] = "Number of timestamps must match data points"
                        continue
                    start_timestamp = pd.to_datetime(timestamps[0])
                else:
                    start_timestamp = pd.Timestamp("2023-01-01 00:00:00")
                
                # Create dataset
                series_data_dict = [{
                    "target": series_data,
                    "start": start_timestamp,
                    "item_id": item_id
                }]
                
                dataset_obj = ListDataset(series_data_dict, freq=frequency)
                
                # Update predictor
                current_predictor.prediction_length = prediction_length
                
                # Make prediction
                forecast_it, ts_it = make_evaluation_predictions(
                    dataset=dataset_obj,
                    predictor=current_predictor,
                    num_samples=100
                )
                
                forecasts = list(forecast_it)
                if len(forecasts) == 0:
                    errors[series_name] = "No forecasts generated"
                    continue
                
                forecast = forecasts[0]
                
                # Extract results
                predictions = forecast.mean.tolist()
                
                # Extract quantiles
                quantiles = {}
                for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
                    quantiles[f"q{int(q*100)}"] = forecast.quantile(q).tolist()
                
                # Generate future timestamps
                future_timestamps = pd.date_range(
                    start=start_timestamp,
                    periods=len(series_data) + prediction_length,
                    freq=frequency
                )[-prediction_length:]
                
                results[series_name] = {
                    "success": True,
                    "predictions": predictions,
                    "future_timestamps": [ts.isoformat() for ts in future_timestamps],
                    "quantiles": quantiles,
                    "metadata": {
                        "input_length": len(series_data),
                        "prediction_length": prediction_length,
                        "frequency": frequency
                    }
                }
                
                log_message(f"✅ {series_name}: {len(predictions)} predictions generated")
                
            except Exception as e:
                error_msg = f"Error processing {series_name}: {str(e)}"
                log_message(f"❌ {error_msg}")
                errors[series_name] = str(e)
        
        # Prepare final response
        response = {
            "success": True,
            "total_series": len(datasets),
            "successful_predictions": len(results),
            "failed_predictions": len(errors),
            "results": results
        }
        
        if errors:
            response["errors"] = errors
        
        log_message(f"✅ Batch completed: {len(results)} successful, {len(errors)} failed")
        return jsonify(response)
    
    except Exception as e:
        error_msg = f"Batch prediction failed: {str(e)}"
        log_message(f"❌ {error_msg}")
        log_message(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API info"""
    return jsonify({
        "message": "Lag-Llama Time Series Forecasting API",
        "version": "production-v1.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict", 
            "batch_predict": "POST /batch_predict"
        },
        "model_status": "loaded" if predictor is not None else "not_loaded",
        "documentation": "Check API_SETUP.md for usage examples",
        "note": "Real Lag-Llama foundation model for time series forecasting"
    })

if __name__ == '__main__':
    print("🚀 Starting Lag-Llama API Server...")
    print("📥 Loading model on startup (this may take a few minutes)...")
    
    # Pre-load model to avoid cold start delay
    load_model()
    
    print("🌐 API Server ready!")
    print("📋 Available endpoints:")
    print("   GET  /health - Health check")
    print("   POST /predict - Single time series prediction")
    print("   POST /batch_predict - Multiple time series prediction")
    print()
    
    app.run(host='0.0.0.0', port=8000, debug=False)