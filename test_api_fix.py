#!/usr/bin/env python3
"""
Test script to verify the API fix
This will test the fixed API directly without starting the Flask server
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the fixed API functions directly
from api import load_estimator, create_predictor_for_length
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions

def test_api_fix():
    """Test the fixed API implementation"""
    print("=" * 60)
    print("TESTING FIXED API IMPLEMENTATION")
    print("=" * 60)
    
    # Test data with actual sales values
    test_data = [100.0, 120.5, 95.3, 110.7, 88.9, 105.2, 115.8, 92.4, 108.1, 97.6,
                 103.3, 118.7, 89.1, 112.4, 99.8, 106.5, 121.2, 94.7, 107.9, 101.3,
                 114.6, 87.5, 109.8, 96.2, 104.1, 117.3, 91.8, 111.5, 98.4, 105.7]
    
    print(f"Input data: {len(test_data)} points")
    print(f"Input range: [{min(test_data):.2f}, {max(test_data):.2f}]")
    print(f"Input mean: {np.mean(test_data):.2f}")
    
    # Load the estimator (once)
    print("\n" + "-" * 40)
    print("LOADING ESTIMATOR")
    print("-" * 40)
    
    estimator = load_estimator()
    if estimator is None:
        print("❌ Failed to load estimator")
        return
    print("✅ Estimator loaded successfully")
    
    # Test different prediction lengths
    for pred_length in [1, 3, 7, 14]:
        print(f"\n" + "-" * 40)
        print(f"TESTING PREDICTION_LENGTH = {pred_length}")
        print("-" * 40)
        
        # Create predictor with specific prediction length
        predictor = create_predictor_for_length(pred_length)
        if predictor is None:
            print(f"❌ Failed to create predictor for length {pred_length}")
            continue
        
        print(f"✅ Predictor created for prediction_length={pred_length}")
        print(f"   predictor.prediction_length = {predictor.prediction_length}")
        
        # Create dataset
        start_timestamp = pd.Timestamp("2023-01-01")
        series_data = [{
            "target": test_data,
            "start": start_timestamp,
            "item_id": "test_series"
        }]
        dataset = ListDataset(series_data, freq='D')
        
        # Make prediction
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=100
        )
        
        forecasts = list(forecast_it)
        if len(forecasts) == 0:
            print(f"❌ No forecasts generated for length {pred_length}")
            continue
        
        forecast = forecasts[0]
        predictions = forecast.mean.tolist()
        
        # Verify results
        print(f"✅ Forecast generated successfully")
        print(f"   Forecast shape: {forecast.mean.shape}")
        print(f"   Expected length: {pred_length}")
        print(f"   Actual length: {len(predictions)}")
        
        if len(predictions) == pred_length:
            print("✅ Length matches expected!")
        else:
            print(f"❌ Length mismatch: expected {pred_length}, got {len(predictions)}")
        
        print(f"   Predictions: {predictions}")
        print(f"   Range: [{min(predictions):.4f}, {max(predictions):.4f}]")
        print(f"   Mean: {np.mean(predictions):.4f}")
        
        # Check if predictions are reasonable (not near zero)
        if all(abs(p) > 1.0 for p in predictions):
            print("✅ Predictions are reasonable (not near zero)")
        else:
            print("❌ Some predictions are near zero")
        
        # Compare with input scale
        pred_vs_input_ratio = np.mean(predictions) / np.mean(test_data)
        print(f"   Prediction/Input ratio: {pred_vs_input_ratio:.4f}")
        
        if 0.5 < pred_vs_input_ratio < 2.0:
            print("✅ Predictions are in reasonable scale vs input")
        else:
            print(f"⚠️  Scale might be off: ratio={pred_vs_input_ratio:.4f}")

def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    # Test with different data scales
    test_cases = [
        ("Small values", [0.1, 0.2, 0.15, 0.18, 0.22, 0.19, 0.17, 0.21, 0.16, 0.20]),
        ("Large values", [10000, 12000, 9500, 11000, 8900, 10500, 11500, 9200, 10800, 9700]),
        ("Zero-heavy data", [0, 0, 1, 0, 2, 0, 0, 3, 1, 0])
    ]
    
    for test_name, test_data in test_cases:
        print(f"\n--- {test_name} ---")
        print(f"Data: {test_data}")
        print(f"Range: [{min(test_data):.3f}, {max(test_data):.3f}]")
        
        try:
            # Create predictor
            predictor = create_predictor_for_length(3)
            if predictor is None:
                print("❌ Failed to create predictor")
                continue
            
            # Create dataset  
            start_timestamp = pd.Timestamp("2023-01-01")
            series_data = [{"target": test_data, "start": start_timestamp, "item_id": "test"}]
            dataset = ListDataset(series_data, freq='D')
            
            # Predict
            forecast_it, ts_it = make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=50)
            forecasts = list(forecast_it)
            
            if forecasts:
                predictions = forecasts[0].mean.tolist()
                print(f"✅ Predictions: {predictions}")
                print(f"   Range: [{min(predictions):.6f}, {max(predictions):.6f}]")
            else:
                print("❌ No forecasts generated")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_api_fix()
    test_edge_cases()
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)