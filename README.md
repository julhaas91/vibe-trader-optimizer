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

### Comprehensive Example with Black-Litterman Views
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": [
        "ADBE",
        "CRM",
        "IEMG",
        "PLTR",
        "SCHE",
        "VWO",
        "BIL"
    ],
    "horizon_years": 10,
    "start_portfolio": 100000.0,
    "target_portfolio": 150000.0,
    "max_drawdown": 0.15,
    "worst_day_limit": 0.05,
    "sigma_max": 0.15,
    "cash_min": 0.05,
    "cash_ticker": "BIL",
    "upper_bounds": 0.25,
    "scenarios": 5,
    "mc_freq": 12,
    "max_iterations": 10,
    "lambda_sigma": 100,
    "lambda_drawdown": 500,
    "lambda_worst_day": 300,
    "lambda_cash": 1000,
    "output_dir": null,
    "log_prefix": "optimization",
    "bl_view_matrix_P": [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "bl_view_vector_Q": [
        0.15,
        0.08,
        0.12,
        0.05,
        0.04,
        0.04,
        0.08
    ],
    "bl_view_uncertainty_omega": [
        0.005,
        0.007,
        0.001,
        0.01,
        0.01,
        0.01,
        0.005
    ]
}'
```

Response:
```json
{
    "inputs": {
        "tickers": [
            "ADBE",
            "CRM",
            "IEMG",
            "PLTR",
            "SCHE",
            "VWO",
            "BIL"
        ],
        "horizon_years": 10,
        "start_portfolio": 100000.0,
        "target_portfolio": 150000.0,
        "max_drawdown": 0.15,
        "worst_day_limit": 0.05,
        "sigma_max": 0.15,
        "cash_min": 0.05,
        "upper_bounds": 0.25,
        "cash_ticker": "BIL",
        "scenarios": 5,
        "mc_freq": 12,
        "max_iterations": 10,
        "log_prefix": "optimization",
        "output_dir": null,
        "bl_views": {
            "P": [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            "Q": [
                0.15,
                0.08,
                0.12,
                0.05,
                0.04,
                0.04,
                0.08
            ],
            "omega": [
                [0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.007, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005]
            ]
        }
    },
    "results": {
        "weights": [
            0.04233112055570924,
            0.3882593449460815,
            0.03162481150333801,
            0.04976826416788655,
            0.019629098013308404,
            0.18584795993004408,
            0.28253940088363233
        ],
        "success_prob": 0.8,
        "avg_final": 176678.9597255846,
        "avg_drawdown": 0.21354742960832987,
        "avg_worst_day": 0.07487700154469627,
        "volatility": 0.11431742804167379,
        "cash_allocation": 0.28253940088363233,
        "elapsed_time": 2.2431010509999965,
        "iterations": 10
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
| cash_ticker | string | null | Ticker symbol for cash allocation |
| mc_freq | int | 12 | Frequency of Monte Carlo simulation |
| lambda_sigma | float | 100 | Lambda for sigma in Black-Litterman model |
| lambda_drawdown | float | 500 | Lambda for drawdown in Black-Litterman model |
| lambda_worst_day | float | 300 | Lambda for worst day in Black-Litterman model |
| lambda_cash | float | 1000 | Lambda for cash allocation in Black-Litterman model |
| log_prefix | string | optimization | Prefix for log files |
| bl_view_matrix_P | float[][] | null | Matrix P in Black-Litterman model |
| bl_view_vector_Q | float[] | null | Vector Q in Black-Litterman model |
| bl_view_uncertainty_omega | float[] | null | Uncertainty matrix omega in Black-Litterman model |

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

## Questions and Support

> If you have any questions or need technical support, feel free to reach out.  
> You can contact me via [email](mailto:juliushaas91@gmail.com) or connect with me on [LinkedIn](https://www.linkedin.com/in/jh91/).
