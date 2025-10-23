#!/usr/bin/env python3
"""
Test script for the Lag-Llama API
Run this after starting the API server to test all endpoints
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_single_prediction():
    """Test single time series prediction"""
    print("\n🔍 Testing single prediction...")
    
    # Sample data: simple increasing trend with some noise
    sample_data = [
        100, 102, 98, 105, 107, 104, 110, 108, 112, 115, 
        118, 114, 120, 122, 119, 125, 128, 124, 130, 132
    ]
    
    payload = {
        "data": sample_data,
        "prediction_length": 5,
        "frequency": "D"
    }
    
    try:
        print(f"   Sending data: {len(sample_data)} points")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120  # 2 minute timeout
        )
        
        duration = time.time() - start_time
        print(f"   Status: {response.status_code}")
        print(f"   Duration: {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Predicted {len(result.get('predictions', []))} values")
            print(f"   Predictions: {result.get('predictions', [])[:3]}...")  # Show first 3
            return True
        else:
            print(f"   ❌ Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction with multiple series"""
    print("\n🔍 Testing batch prediction...")
    
    # Sample data for multiple series
    payload = {
        "datasets": [
            {
                "name": "sales_product_A",
                "data": [50, 52, 48, 55, 57, 54, 60, 58, 62, 65, 68, 64, 70, 72],
                "item_id": "product_A"
            },
            {
                "name": "sales_product_B", 
                "data": [200, 205, 195, 210, 215, 208, 220, 218, 225, 230, 235, 228, 240, 245],
                "item_id": "product_B"
            }
        ],
        "prediction_length": 3,
        "frequency": "D"
    }
    
    try:
        print(f"   Sending {len(payload['datasets'])} time series")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE}/batch_predict",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=180  # 3 minute timeout
        )
        
        duration = time.time() - start_time
        print(f"   Status: {response.status_code}")
        print(f"   Duration: {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Processed {result.get('successful_predictions', 0)} series")
            print(f"   Failed: {result.get('failed_predictions', 0)} series")
            
            if result.get('results'):
                for name, series_result in result['results'].items():
                    preds = series_result.get('predictions', [])
                    print(f"   {name}: {preds}")
            
            return True
        else:
            print(f"   ❌ Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid data"""
    print("\n🔍 Testing error handling...")
    
    # Test with insufficient data
    payload = {
        "data": [1, 2, 3],  # Too few data points
        "prediction_length": 5
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 400:  # Should return error
            error_response = response.json()
            print(f"   ✅ Correctly rejected invalid data")
            print(f"   Error details: {error_response.get('details', [])}")
            return True
        else:
            print(f"   ❌ Should have rejected invalid data")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Lag-Llama API")
    print("=" * 50)
    
    results = []
    
    # Test health endpoint
    results.append(test_health())
    
    # Test single prediction
    results.append(test_single_prediction())
    
    # Test batch prediction
    results.append(test_batch_prediction())
    
    # Test error handling
    results.append(test_error_handling())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   ✅ Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("   🎉 All tests passed! API is working correctly.")
    else:
        print("   ⚠️  Some tests failed. Check the API server logs.")
    
    print("\n💡 Next Steps:")
    print("   - Use the API in your JavaScript application")
    print("   - Check API_SETUP.md for integration examples")
    print("   - Monitor server logs for any issues")

if __name__ == "__main__":
    main()