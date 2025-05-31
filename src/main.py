import os
import json
import numpy as np
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
    cash_ticker = data.get("cash_ticker", "BIL")
    upper_bounds = data.get("upper_bounds", 0.5)
    scenarios = data.get("scenarios", 100)
    mc_freq = data.get("mc_freq", 12)
    max_iterations = data.get("max_iterations", 5)
    lambda_sigma = data.get("lambda_sigma", 100)
    lambda_drawdown = data.get("lambda_drawdown", 500)
    lambda_worst_day = data.get("lambda_worst_day", 300)
    lambda_cash = data.get("lambda_cash", 1000)
    output_dir = data.get("output_dir")
    log_prefix = data.get("log_prefix", "optimization")
    bl_view_matrix_P = data.get("bl_view_matrix_P", None)
    bl_view_vector_Q = data.get("bl_view_vector_Q", None)
    bl_view_uncertainty_omega = data.get("bl_view_uncertainty_omega", None)

    # Convert numpy arrays to lists if they exist
    if bl_view_matrix_P is not None:
        bl_view_matrix_P = np.array(bl_view_matrix_P)
    if bl_view_vector_Q is not None:
        bl_view_vector_Q = np.array(bl_view_vector_Q)
    if bl_view_uncertainty_omega is not None:
        bl_view_uncertainty_omega = np.array(bl_view_uncertainty_omega)

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
        cash_ticker=cash_ticker,
        upper_bounds=upper_bounds,
        max_iterations=max_iterations,
        scenarios=scenarios,
        mc_freq=mc_freq,
        lambda_sigma=lambda_sigma,
        lambda_drawdown=lambda_drawdown,
        lambda_worst_day=lambda_worst_day,
        lambda_cash=lambda_cash,
        output_dir=output_dir,
        log_prefix=log_prefix,
        bl_view_matrix_P=bl_view_matrix_P,
        bl_view_vector_Q=bl_view_vector_Q,
        bl_view_uncertainty_omega=bl_view_uncertainty_omega
    )
    
    # Run optimization
    results = optimizer.optimize(save_outputs=False)

    return json.dumps(results)

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
