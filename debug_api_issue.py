#!/usr/bin/env python3
"""
Debug script to reproduce the exact API issue
This will simulate the API flow to identify why we get e-12/e-13 values
"""

import pandas as pd
import numpy as np
import torch
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')

# Fix PyTorch weights loading issue
torch.serialization.add_safe_globals([
    'gluonts.torch.distributions.studentT.StudentTOutput',
    'gluonts.torch.distributions.implicit_quantile_network.ImplicitQuantileNetwork'
])

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from lag_llama.gluon.estimator import LagLlamaEstimator

def simulate_api_prediction():
    """Simulate the exact API prediction flow"""
    print("=" * 60)
    print("SIMULATING API PREDICTION FLOW")
    print("=" * 60)
    
    # Test with actual sales data (similar to what would come from API)
    time_series = [100.0, 120.5, 95.3, 110.7, 88.9, 105.2, 115.8, 92.4, 108.1, 97.6,
                   103.3, 118.7, 89.1, 112.4, 99.8, 106.5, 121.2, 94.7, 107.9, 101.3,
                   114.6, 87.5, 109.8, 96.2, 104.1, 117.3, 91.8, 111.5, 98.4, 105.7,
                   119.9, 93.6, 108.2, 100.1, 113.4]
    
    # API parameters
    prediction_length = 7
    frequency = 'D'
    context_length = 32
    item_id = 'test_series'
    
    print(f"Input data: {len(time_series)} points")
    print(f"Input range: [{min(time_series):.3f}, {max(time_series):.3f}]")
    print(f"Requested prediction length: {prediction_length}")
    
    # === EXACT API MODEL SETUP ===
    print("\n" + "-" * 40)
    print("MODEL SETUP (API Style)")
    print("-" * 40)
    
    # Download model
    repo_id = "time-series-foundation-models/Lag-Llama"
    filename = "lag-llama.ckpt"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"✓ Model downloaded: {ckpt_path}")
    
    # Fix PyTorch loading
    original_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = safe_load
    
    # Create estimator with SAME parameters as API
    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=24,  # ⚠️ API initializes with 24!
        context_length=32,
        input_size=1,
        n_layer=8,
        n_embd_per_head=16,
        n_head=9,
        scaling="robust",
        rope_scaling=None,
        batch_size=1,
        time_feat=True,  # ⚠️ API uses time_feat=True
        trainer_kwargs={}
    )
    
    print("✓ Estimator created (initialized with prediction_length=24)")
    
    # Create predictor
    predictor = estimator.create_predictor(
        estimator.create_transformation(),
        estimator.create_lightning_module()
    )
    
    torch.load = original_load
    print("✓ Predictor created")
    
    # === EXACT API DATA PREPARATION ===
    print("\n" + "-" * 40)
    print("DATA PREPARATION (API Style)")
    print("-" * 40)
    
    # Prepare timestamps (API style)
    start_timestamp = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    print(f"Start timestamp: {start_timestamp}")
    
    # Create GluonTS dataset format (API style)
    series_data = [{
        "target": time_series,
        "start": start_timestamp,
        "item_id": item_id
    }]
    
    dataset = ListDataset(series_data, freq=frequency)
    print(f"✓ Dataset created with freq='{frequency}'")
    
    # === API PREDICTION FLOW ===
    print("\n" + "-" * 40)
    print("PREDICTION (API Style)")  
    print("-" * 40)
    
    # ⚠️ THIS IS THE SUSPICIOUS PART - API changes prediction_length AFTER creation
    print(f"Original predictor.prediction_length: {predictor.prediction_length}")
    predictor.prediction_length = prediction_length  # API does this!
    print(f"Changed predictor.prediction_length to: {predictor.prediction_length}")
    
    # Make prediction
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    if len(forecasts) == 0:
        print("❌ No forecasts generated!")
        return
    
    forecast = forecasts[0]
    print("✓ Forecast generated")
    
    # Debug the forecast
    print(f"\nForecast.mean shape: {forecast.mean.shape}")
    print(f"Forecast.mean full values: {forecast.mean}")
    
    # === API SLICING ===
    print("\n" + "-" * 40)
    print("API SLICING LOGIC")
    print("-" * 40)
    
    print(f"forecast.mean.shape: {forecast.mean.shape}")
    print(f"requested prediction_length: {prediction_length}")
    
    # API does this slicing:
    predictions = forecast.mean.tolist()[:prediction_length]
    print(f"After slicing [:prediction_length]: {predictions}")
    print(f"Predictions range: [{min(predictions):.10f}, {max(predictions):.10f}]")
    
    # Compare with no slicing
    predictions_full = forecast.mean.tolist()
    print(f"Full predictions (no slicing): {predictions_full}")
    
    # Check if the issue is in the slicing
    if len(predictions_full) != prediction_length:
        print(f"⚠️ LENGTH MISMATCH: forecast generated {len(predictions_full)} values but we requested {prediction_length}")
    
    # === ADDITIONAL DEBUG ===
    print("\n" + "-" * 40)
    print("ADDITIONAL DEBUG")
    print("-" * 40)
    
    # Check samples
    print(f"Forecast.samples.shape: {forecast.samples.shape}")
    print(f"Forecast.samples range: [{forecast.samples.min():.6f}, {forecast.samples.max():.6f}]")
    
    # Manual mean calculation
    manual_mean = forecast.samples.mean(axis=0)
    print(f"Manual mean from samples: {manual_mean}")
    print(f"Manual mean sliced: {manual_mean[:prediction_length]}")
    
    # Check for any scaling issues
    print(f"\nInput data mean: {np.mean(time_series):.3f}")
    print(f"Prediction mean: {np.mean(predictions):.3f}") 
    print(f"Ratio: {np.mean(predictions) / np.mean(time_series):.6f}")

def test_different_prediction_lengths():
    """Test different prediction lengths to see if there's a pattern"""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT PREDICTION LENGTHS")
    print("=" * 60)
    
    # Test data
    time_series = [100.0, 120.5, 95.3, 110.7, 88.9, 105.2, 115.8, 92.4, 108.1, 97.6]
    
    for pred_len in [1, 3, 7, 12, 24]:
        print(f"\n--- Testing prediction_length = {pred_len} ---")
        
        # Setup model
        repo_id = "time-series-foundation-models/Lag-Llama"
        filename = "lag-llama.ckpt"
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = safe_load
        
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=24,  # Always init with 24
            context_length=32,
            input_size=1,
            n_layer=8,
            n_embd_per_head=16,
            n_head=9,
            scaling="robust",
            rope_scaling=None,
            batch_size=1,
            time_feat=True,
            trainer_kwargs={}
        )
        
        predictor = estimator.create_predictor(
            estimator.create_transformation(),
            estimator.create_lightning_module()
        )
        torch.load = original_load
        
        # Change prediction length
        predictor.prediction_length = pred_len
        
        # Make dataset
        start_timestamp = pd.Timestamp.now().normalize()
        series_data = [{"target": time_series, "start": start_timestamp, "item_id": "test"}]
        dataset = ListDataset(series_data, freq='D')
        
        # Predict
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=100
        )
        
        forecasts = list(forecast_it)
        if forecasts:
            forecast = forecasts[0]
            predictions = forecast.mean.tolist()[:pred_len]
            print(f"Shape: {forecast.mean.shape}, Predictions: {predictions}")
            print(f"Range: [{min(predictions):.6f}, {max(predictions):.6f}]")
        else:
            print("No forecast generated")

if __name__ == "__main__":
    simulate_api_prediction()
    test_different_prediction_lengths()