# Vibe Trader Optimizer

A portfolio optimization service that uses the Black-Litterman model and Monte Carlo simulation to optimize portfolio allocation.

## Deployment

The service is deployed as a Cloud Run service in the `vibetrader-458114` project and is accessible at:
```
https://viber-trader-optimizer-189263797377.europe-west3.run.app
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service locally:
```bash
./taskfile.sh run_application
```

## API Endpoints

### Health Check
```bash
curl -X GET http://localhost:8000/health
```
Response:
```json
{"status": "ok"}
```

### Portfolio Optimization
```bash
# Local deployment
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["SPY", "QQQ", "TLT", "GLD", "BIL"],
    "horizon_years": 1,
    "start_portfolio": 100000,
    "target_portfolio": 200000,
    "max_drawdown": 0.25,
    "worst_day_limit": 0.10,
    "sigma_max": 0.20,
    "cash_min": 0.05,
    "upper_bounds": 0.5,
    "scenarios": 100,
    "max_iterations": 1,
    "output_dir": null
  }'

# Cloud Run deployment (requires authentication)
# First, generate a bearer token using the command below
# Then use the token in the Authorization header
curl -X POST https://viber-trader-optimizer-189263797377.europe-west3.run.app/optimize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token --audiences=https://viber-trader-optimizer-189263797377.europe-west3.run.app)" \
  -d '{
    "tickers": ["SPY", "QQQ", "TLT", "GLD", "BIL"],
    "horizon_years": 1,
    "start_portfolio": 100000,
    "target_portfolio": 200000,
    "max_drawdown": 0.25,
    "worst_day_limit": 0.10,
    "sigma_max": 0.20,
    "cash_min": 0.05,
    "upper_bounds": 0.5,
    "scenarios": 100,
    "max_iterations": 1,
    "output_dir": null
  }'
```

Response:
```json
{
  "optimal_weights": {
    "SPY": 0.35,
    "QQQ": 0.25,
    "TLT": 0.15,
    "GLD": 0.10,
    "BIL": 0.15
  },
  "metrics": {
    "probability_of_success": 0.75,
    "average_drawdown": 0.15,
    "worst_day": 0.08,
    "volatility": 0.18,
    "average_final_value": 185000
  }
}
```

### Process Message
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"message": "hello world"}'
```
Response:
```json
{"uppercaseMessage": "HELLO WORLD"}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tickers | string[] | ["SPY", "QQQ", "TLT", "GLD", "BIL"] | List of ticker symbols to optimize |
| horizon_years | int | 10 | Investment horizon in years |
| start_portfolio | float | 100,000 | Initial portfolio value |
| target_portfolio | float | 200,000 | Target portfolio value |
| max_drawdown | float | 0.25 | Maximum allowed drawdown |
| worst_day_limit | float | 0.10 | Maximum allowed single-day drop |
| sigma_max | float | 0.20 | Maximum allowed annual volatility |
| cash_min | float | 0.05 | Minimum cash allocation |
| upper_bounds | float | 0.5 | Maximum weight per asset |
| scenarios | int | 10000 | Number of Monte Carlo simulation scenarios |
| max_iterations | int | 100 | Maximum number of optimization iterations |
| output_dir | string | null | Directory to save output files (null for no file writing) |

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| optimal_weights | object | Dictionary mapping tickers to their optimal weights |
| metrics.probability_of_success | float | Probability of reaching target portfolio value |
| metrics.average_drawdown | float | Average maximum drawdown across scenarios |
| metrics.worst_day | float | Average worst single-day drop |
| metrics.volatility | float | Annual portfolio volatility |
| metrics.average_final_value | float | Average final portfolio value across scenarios |

## Authentication

To access the Cloud Run service, you need to generate a bearer token. First, authenticate with your service account:

```bash
gcloud auth activate-service-account --key-file=service-key.json
```

Then generate the bearer token:

```bash
gcloud auth print-identity-token --audiences=https://viber-trader-optimizer-189263797377.europe-west3.run.app
```

Include the token in the `Authorization` header of your requests:
```bash
-H "Authorization: Bearer <your-token>"
```

Note: The token is valid for 1 hour. You'll need to generate a new token when it expires.
