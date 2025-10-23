# Lag-Llama API Setup Guide

This guide will help you set up and run the Lag-Llama time series forecasting API.

## Quick Start

### 1. Install Dependencies

```bash
# Install API-specific requirements
pip install -r api-requirements.txt

# OR install everything (if you want to use original Lag-Llama scripts too)
pip install -r requirements.txt flask flask-cors gunicorn
```

### 2. Run the API

```bash
# Development mode (with auto-reload)
python api.py

# Production mode (recommended)
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 300 api:app
```

The API will start on `http://localhost:8000`

**Note:** First startup takes 2-5 minutes to download the model (~1GB) from HuggingFace.

## API Endpoints

### Health Check
```bash
GET /health
```

### Single Time Series Prediction
```bash
POST /predict

# Example request:
{
    "data": [100, 105, 98, 110, 115, 120, 108, 95, 102, 98],
    "prediction_length": 7,
    "frequency": "D"
}
```

### Batch Prediction (Multiple Series)
```bash
POST /batch_predict

# Example request:
{
    "datasets": [
        {
            "name": "sales_series_1",
            "data": [100, 105, 98, 110, 115, 120]
        },
        {
            "name": "sales_series_2", 
            "data": [200, 210, 195, 220, 225, 240]
        }
    ],
    "prediction_length": 5,
    "frequency": "D"
}
```

## JavaScript Usage Examples

### Simple Prediction
```javascript
const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        data: [100, 105, 98, 110, 115, 120, 108, 95, 102],
        prediction_length: 7,
        frequency: "D"
    })
});

const result = await response.json();
console.log('Predictions:', result.predictions);
console.log('Future dates:', result.future_timestamps);
```

### With Custom Timestamps
```javascript
const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        data: [100, 105, 98, 110, 115, 120],
        timestamps: ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
        prediction_length: 3,
        frequency: "D"
    })
});
```

### Batch Processing
```javascript
const response = await fetch('http://localhost:8000/batch_predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        datasets: [
            { name: "product_A", data: [50, 55, 48, 60, 65] },
            { name: "product_B", data: [150, 155, 148, 160, 165] }
        ],
        prediction_length: 3,
        frequency: "D"
    })
});

const result = await response.json();
console.log('Results:', result.results);
```

## Parameters Reference

### Required Parameters
- **data**: Array of numbers (minimum 10 data points)
- **prediction_length**: Number of future steps to predict

### Optional Parameters
- **frequency**: Data frequency - "T"=minute, "H"=hour, "D"=day (default), "W"=week, "M"=month
- **timestamps**: Array of timestamp strings (optional, will generate if not provided)
- **context_length**: History length to use (default: 32)
- **item_id**: Unique identifier for the series

## Response Format

```json
{
    "success": true,
    "predictions": [102.5, 104.2, 106.1, 103.8, 105.5, 107.2, 109.0],
    "future_timestamps": ["2023-01-10T00:00:00", "2023-01-11T00:00:00", ...],
    "quantiles": {
        "q10": [95.2, 96.8, ...],  // 10% confidence interval
        "q90": [108.7, 110.3, ...]  // 90% confidence interval
    },
    "metadata": {
        "input_length": 50,
        "prediction_length": 7,
        "frequency": "D",
        "model_info": "Lag-Llama foundation model"
    }
}
```

## System Requirements

### Minimum (CPU-only)
- 8GB RAM
- 2GB free disk space
- Python 3.8+

### Recommended (with GPU)
- 16GB RAM + 8GB GPU VRAM
- CUDA-compatible GPU

## Production Deployment

### Docker (Recommended)
```bash
# Build image
docker build -t lag-llama-api .

# Run container
docker run -p 8000:8000 lag-llama-api
```

### Cloud Deployment
- **AWS**: Use EC2 with GPU (p3.2xlarge) or CPU instance (t3.2xlarge)
- **Google Cloud**: Use Compute Engine with GPU or Cloud Run
- **DigitalOcean**: Use droplets with 16GB+ RAM

### Production Settings
```bash
# Use Gunicorn with single worker (model is memory-intensive)
gunicorn -w 1 -b 0.0.0.0:8000 --timeout 300 --max-requests 100 api:app
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size or use smaller context length
   - Ensure sufficient RAM available

2. **Model Download Fails**
   - Check internet connection
   - Ensure HuggingFace Hub access

3. **Slow First Request**
   - Model loads on first request (normal)
   - Use health endpoint to pre-warm

4. **CORS Issues**
   - API includes CORS headers
   - Check your frontend domain

### Logs
The API provides detailed logging with timestamps for debugging.

## API Limits

- **Maximum prediction length**: 100 steps
- **Minimum data points**: 10 historical values
- **Batch size limit**: 50 series per request
- **Request timeout**: 300 seconds

## Support

For issues with:
- **Lag-Llama model**: Check the [original repository](https://github.com/time-series-foundation-models/lag-llama)
- **API wrapper**: Create an issue or check the logs for error details