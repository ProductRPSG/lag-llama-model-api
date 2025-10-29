#!/usr/bin/env python3
"""
Test the full API by making actual HTTP requests
This will test the complete API pipeline with realistic sales data
"""

import requests
import json
import time

def test_api_endpoint(base_url="http://localhost:8000"):
    """Test the API endpoints with sales data"""
    print("=" * 60)
    print("TESTING FULL API WITH SALES DATA")
    print("=" * 60)
    
    # Test data representing actual SKU sales (units per day)
    sales_data = [
        45, 52, 38, 61, 47, 55, 49, 43, 58, 51,
        46, 63, 39, 57, 48, 54, 50, 44, 59, 52,
        47, 65, 41, 56, 49, 53, 51, 45, 60, 53,
        48, 62, 40, 58, 47, 55, 52
    ]
    
    print(f"Input sales data: {len(sales_data)} days")
    print(f"Sales range: [{min(sales_data)}, {max(sales_data)}] units/day")
    print(f"Average daily sales: {sum(sales_data)/len(sales_data):.1f} units")
    
    # Test different prediction lengths
    test_cases = [
        {"prediction_length": 3, "description": "3-day forecast"},
        {"prediction_length": 7, "description": "1-week forecast"}, 
        {"prediction_length": 14, "description": "2-week forecast"},
        {"prediction_length": 30, "description": "1-month forecast"}
    ]
    
    for case in test_cases:
        print(f"\n" + "-" * 50)
        print(f"TESTING: {case['description']}")
        print("-" * 50)
        
        # API request payload
        payload = {
            "data": sales_data,
            "prediction_length": case["prediction_length"],
            "frequency": "D",
            "item_id": "SKU_12345"
        }
        
        try:
            # Make API request
            print("Making API request...")
            start_time = time.time()
            response = requests.post(f"{base_url}/predict", json=payload, timeout=60)
            end_time = time.time()
            
            print(f"Response time: {end_time - start_time:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API request successful!")
                
                # Analyze results
                if result.get("success"):
                    predictions = result["predictions"]
                    print(f"✅ Received {len(predictions)} predictions")
                    print(f"✅ Expected length: {case['prediction_length']}, Got: {len(predictions)}")
                    
                    if len(predictions) == case["prediction_length"]:
                        print("✅ Prediction length matches request!")
                    else:
                        print(f"❌ Length mismatch: expected {case['prediction_length']}, got {len(predictions)}")
                    
                    # Check prediction values
                    min_pred = min(predictions)
                    max_pred = max(predictions) 
                    avg_pred = sum(predictions) / len(predictions)
                    
                    print(f"Predictions: {[round(p, 2) for p in predictions]}")
                    print(f"Range: [{min_pred:.2f}, {max_pred:.2f}]")
                    print(f"Average: {avg_pred:.2f}")
                    
                    # Sanity checks
                    if all(p > 0.1 for p in predictions):
                        print("✅ All predictions are positive and reasonable")
                    else:
                        print("❌ Some predictions are near zero or negative")
                    
                    # Compare with historical average
                    historical_avg = sum(sales_data) / len(sales_data)
                    ratio = avg_pred / historical_avg
                    print(f"Prediction vs Historical ratio: {ratio:.3f}")
                    
                    if 0.3 < ratio < 3.0:
                        print("✅ Predictions are in reasonable range vs historical data")
                    else:
                        print(f"⚠️  Predictions may be scaled incorrectly: ratio={ratio:.3f}")
                    
                    # Check quantiles
                    if "quantiles" in result:
                        q50 = result["quantiles"].get("q50", [])
                        if q50:
                            print(f"Median predictions: {[round(q, 2) for q in q50]}")
                    
                    # Check metadata
                    metadata = result.get("metadata", {})
                    print(f"Metadata: {metadata}")
                    
                else:
                    print(f"❌ API returned success=False: {result}")
                
            else:
                print(f"❌ API request failed: {response.status_code}")
                try:
                    error_details = response.json()
                    print(f"Error details: {error_details}")
                except:
                    print(f"Raw response: {response.text}")
                    
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to API server")
            print("   Make sure the API server is running on localhost:8000")
            print("   Start with: python api.py")
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

def test_batch_api(base_url="http://localhost:8000"):
    """Test the batch API endpoint"""
    print(f"\n" + "=" * 60)
    print("TESTING BATCH API")
    print("=" * 60)
    
    # Multiple SKUs with different sales patterns
    batch_data = {
        "datasets": [
            {
                "name": "High_Volume_SKU",
                "data": [100, 110, 95, 120, 105, 115, 98, 125, 108, 118],
                "item_id": "SKU_A"
            },
            {
                "name": "Low_Volume_SKU", 
                "data": [5, 7, 4, 8, 6, 9, 5, 7, 6, 8],
                "item_id": "SKU_B"
            },
            {
                "name": "Seasonal_SKU",
                "data": [20, 25, 30, 35, 40, 35, 30, 25, 20, 15],
                "item_id": "SKU_C"
            }
        ],
        "prediction_length": 5,
        "frequency": "D"
    }
    
    try:
        print("Making batch API request...")
        start_time = time.time()
        response = requests.post(f"{base_url}/batch_predict", json=batch_data, timeout=120)
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch API request successful!")
            
            if result.get("success"):
                print(f"Total series: {result['total_series']}")
                print(f"Successful: {result['successful_predictions']}")
                print(f"Failed: {result['failed_predictions']}")
                
                results = result.get("results", {})
                for series_name, series_result in results.items():
                    print(f"\n--- {series_name} ---")
                    predictions = series_result["predictions"]
                    print(f"Predictions: {[round(p, 2) for p in predictions]}")
                    print(f"Range: [{min(predictions):.2f}, {max(predictions):.2f}]")
                
                # Check for errors
                if "errors" in result:
                    print(f"\nErrors: {result['errors']}")
                
            else:
                print(f"❌ Batch API returned success=False: {result}")
        else:
            print(f"❌ Batch API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch API error: {str(e)}")

def test_health_check(base_url="http://localhost:8000"):
    """Test the health check endpoint"""
    print(f"\n" + "=" * 60)
    print("TESTING HEALTH CHECK")
    print("=" * 60)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check passed!")
            print(f"Status: {health.get('status')}")
            print(f"Model Status: {health.get('model_status')}")
            print(f"Message: {health.get('message')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

if __name__ == "__main__":
    print("🧪 Starting comprehensive API tests...")
    print("📌 Make sure the API server is running: python api.py")
    print()
    
    test_health_check()
    test_api_endpoint()
    test_batch_api()
    
    print(f"\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print("If all tests passed, the zero prediction issue has been resolved!")