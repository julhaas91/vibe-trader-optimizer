import os
import json
from litestar import Litestar, Request, get, post
from src.common import logger
from src.optimization.portfolio_optimizer import PortfolioOptimizer

env = os.environ.get("PYTHON_ENV")


@post("/optimize")
async def optimize_portfolio(request: Request) -> dict:
    """Optimize portfolio allocation based on provided parameters."""
    data = await request.json()

    logger.info(f"Received request: {data}")
    
    
    # Extract parameters from request
    tickers = data.get("tickers", ["SPY", "QQQ", "TLT", "GLD", "BIL"])
    horizon_years = data.get("horizon_years", 10)
    start_portfolio = data.get("start_portfolio", 100_000)
    target_portfolio = data.get("target_portfolio", 200_000)
    max_drawdown = data.get("max_drawdown", 0.25)
    worst_day_limit = data.get("worst_day_limit", 0.10)
    sigma_max = data.get("sigma_max", 0.20)
    cash_min = data.get("cash_min", 0.05)
    upper_bounds = data.get("upper_bounds", 0.5)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(
        tickers=tickers,
        horizon_years=horizon_years,
        start_portfolio=start_portfolio,
        target_portfolio=target_portfolio,
        max_drawdown=max_drawdown,
        worst_day_limit=worst_day_limit,
        sigma_max=sigma_max,
        cash_min=cash_min,
        upper_bounds=upper_bounds
    )
    
    # Run optimization
    results = optimizer.optimize(save_outputs=False)
    
    # Format response
    response = {
        "optimal_weights": {ticker: float(weight) for ticker, weight in zip(tickers, results["results"]["weights"])},
        "metrics": {
            "probability_of_success": float(results["results"]["success_prob"]),
            "average_drawdown": float(results["results"]["avg_drawdown"]),
            "worst_day": float(results["results"]["avg_worst_day"]),
            "volatility": float(results["results"]["volatility"]),
            "average_final_value": float(results["results"]["avg_final"])
        }
    }
    return json.dumps(response)

@post("/process")
async def process_message(request: Request) -> dict:
    data = await request.json()
    message = data.get("message", "")
    return {"uppercaseMessage": message.upper()}

@get("/health")
async def health_check() -> str:
    """Check service health."""
    logger.info("Health check route called.")
    return json.dumps({"status": "ok"})


app = Litestar(
    route_handlers=[process_message, health_check, optimize_portfolio]
)
