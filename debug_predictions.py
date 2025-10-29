#!/usr/bin/env python3
"""
Debug script to investigate the zero prediction issue in Lag-Llama
This script will help us understand:
1. What forecast.mean actually returns
2. Whether the issue is in scaling, model output, or mean calculation
3. Raw forecast.samples analysis
4. Data flow through the pipeline
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

def debug_model_setup():
    """Set up the model with debug information"""
    print("=" * 60)
    print("DEBUGGING LAG-LLAMA PREDICTION ISSUES")
    print("=" * 60)
    
    # Download model from HuggingFace
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
    
    # Create estimator
    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=7,  # We'll test with 7 steps
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
    
    print("✓ Estimator created with robust scaling")
    
    # Create predictor
    predictor = estimator.create_predictor(
        estimator.create_transformation(),
        estimator.create_lightning_module()
    )
    
    # Restore original torch.load
    torch.load = original_load
    
    print("✓ Predictor ready for debugging")
    return predictor, estimator

def debug_data_preparation(data_values):
    """Debug data preparation and scaling"""
    print("\n" + "=" * 40)
    print("DATA PREPARATION DEBUG")
    print("=" * 40)
    
    print(f"Input data shape: {np.array(data_values).shape}")
    print(f"Input data range: [{min(data_values):.3f}, {max(data_values):.3f}]")
    print(f"Input data mean: {np.mean(data_values):.3f}")
    print(f"Input data std: {np.std(data_values):.3f}")
    print(f"Sample values: {data_values[:10]}")
    
    # Create dataset
    start_timestamp = pd.Timestamp("2023-01-01")
    series_data = [{
        "target": data_values,
        "start": start_timestamp,
        "item_id": "debug_series"
    }]
    
    dataset = ListDataset(series_data, freq='D')
    print(f"✓ Dataset created with frequency 'D'")
    
    return dataset

def debug_forecast_object(forecast):
    """Debug the forecast object structure"""
    print("\n" + "=" * 40)
    print("FORECAST OBJECT DEBUG")
    print("=" * 40)
    
    print(f"Forecast type: {type(forecast)}")
    print(f"Forecast attributes: {dir(forecast)}")
    
    # Check if it's a SampleForecast
    if hasattr(forecast, 'samples'):
        print(f"Raw samples shape: {forecast.samples.shape}")
        print(f"Raw samples dtype: {forecast.samples.dtype}")
        print(f"Raw samples range: [{forecast.samples.min():.6f}, {forecast.samples.max():.6f}]")
        print(f"Raw samples mean across all: {forecast.samples.mean():.6f}")
        
        # Debug individual samples
        print("\nFirst 3 samples for first 5 timesteps:")
        for i in range(min(3, forecast.samples.shape[0])):
            sample_slice = forecast.samples[i, :5] if forecast.samples.shape[1] >= 5 else forecast.samples[i, :]
            print(f"  Sample {i}: {sample_slice}")
    
    # Check mean calculation
    if hasattr(forecast, 'mean'):
        print(f"\nForecast.mean shape: {forecast.mean.shape}")
        print(f"Forecast.mean dtype: {forecast.mean.dtype}")
        print(f"Forecast.mean values: {forecast.mean}")
        print(f"Forecast.mean range: [{forecast.mean.min():.10f}, {forecast.mean.max():.10f}]")
    
    # Check if mean is computed from samples
    if hasattr(forecast, 'samples') and hasattr(forecast, 'mean'):
        manual_mean = forecast.samples.mean(axis=0)
        print(f"\nManual mean from samples: {manual_mean}")
        print(f"Difference from forecast.mean: {np.abs(manual_mean - forecast.mean).max():.10f}")
    
    # Check quantiles
    for q in [0.1, 0.5, 0.9]:
        if hasattr(forecast, 'quantile'):
            quantile_val = forecast.quantile(q)
            print(f"Quantile {q}: {quantile_val}")

def debug_transformation_pipeline(estimator, dataset):
    """Debug the transformation pipeline"""
    print("\n" + "=" * 40)
    print("TRANSFORMATION PIPELINE DEBUG")
    print("=" * 40)
    
    transformation = estimator.create_transformation()
    print(f"Transformation: {transformation}")
    
    # Apply transformation to see what happens to data
    transformed_data = list(transformation.apply(dataset))
    print(f"Transformed data length: {len(transformed_data)}")
    
    if transformed_data:
        sample = transformed_data[0]
        print(f"Transformed sample keys: {sample.keys()}")
        
        if 'target' in sample:
            target = sample['target']
            print(f"Transformed target shape: {target.shape if hasattr(target, 'shape') else 'no shape'}")
            print(f"Transformed target: {target}")
        
        if 'past_target' in sample:
            past_target = sample['past_target']
            print(f"Past target shape: {past_target.shape if hasattr(past_target, 'shape') else 'no shape'}")
            print(f"Past target values: {past_target}")

def debug_model_scaling(predictor, dataset):
    """Debug the model's internal scaling"""
    print("\n" + "=" * 40)
    print("MODEL SCALING DEBUG")
    print("=" * 40)
    
    # Access the model through predictor
    model = predictor.prediction_net.model
    print(f"Model scaler type: {type(model.scaler)}")
    print(f"Model scaler config: {model.scaler.__dict__ if hasattr(model.scaler, '__dict__') else 'no dict'}")
    
    # Try to understand scaling parameters
    if hasattr(model.scaler, 'minimum_scale'):
        print(f"Minimum scale: {model.scaler.minimum_scale}")

def main():
    """Main debugging function"""
    # Test data with actual values (not near zero)
    test_data = [100.0, 120.5, 95.3, 110.7, 88.9, 105.2, 115.8, 92.4, 108.1, 97.6,
                 103.3, 118.7, 89.1, 112.4, 99.8, 106.5, 121.2, 94.7, 107.9, 101.3,
                 114.6, 87.5, 109.8, 96.2, 104.1, 117.3, 91.8, 111.5, 98.4, 105.7,
                 119.9, 93.6, 108.2, 100.1, 113.4]
    
    # Set up model
    predictor, estimator = debug_model_setup()
    
    # Debug data preparation
    dataset = debug_data_preparation(test_data)
    
    # Debug transformation pipeline
    debug_transformation_pipeline(estimator, dataset)
    
    # Debug model scaling
    debug_model_scaling(predictor, dataset)
    
    # Make prediction with debugging
    print("\n" + "=" * 40)
    print("MAKING PREDICTION")
    print("=" * 40)
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=100
    )
    
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    if forecasts:
        forecast = forecasts[0]
        print("✓ Forecast generated successfully")
        
        # Debug the forecast object
        debug_forecast_object(forecast)
        
        # Extract predictions
        predictions = forecast.mean.tolist()
        print(f"\nFinal predictions: {predictions}")
        print(f"Predictions range: [{min(predictions):.10f}, {max(predictions):.10f}]")
        
        # Compare with input data scale
        print(f"Input data range: [{min(test_data):.3f}, {max(test_data):.3f}]")
        print(f"Prediction vs input ratio: {np.mean(predictions) / np.mean(test_data):.10f}")
    else:
        print("❌ No forecasts generated!")

if __name__ == "__main__":
    main()