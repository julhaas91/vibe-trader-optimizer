"""Portfolio Optimizer Parameter Validation Module.

This module provides comprehensive validation for PortfolioOptimizer parameters to ensure
they are correct, consistent, and feasible before optimization begins. Early validation
prevents runtime errors and helps identify configuration issues.
"""

from typing import Any, Dict

import numpy as np


def validate_optimizer_params(params: Dict[str, Any]) -> bool:
    """Validate PortfolioOptimizer parameters for correctness and consistency.
    
    Args:
        params: Dictionary of parameters for PortfolioOptimizer
        
    Returns:
        bool: True if all validations pass
        
    Raises:
        ValueError: If any validation fails with descriptive error message
        TypeError: If parameter types are incorrect
    """
    # === BASIC TYPE AND EXISTENCE CHECKS ===
    _validate_required_params(params)
    
    # === PORTFOLIO PARAMETERS ===
    _validate_portfolio_params(params)
    
    # === RISK PARAMETERS ===
    _validate_risk_params(params)
    
    # === ALLOCATION PARAMETERS ===
    _validate_allocation_params(params)
    
    # === SIMULATION PARAMETERS ===
    _validate_simulation_params(params)
    
    # === BLACK-LITTERMAN PARAMETERS ===
    _validate_bl_params(params)
    
    return True


def _validate_required_params(params: Dict[str, Any]) -> None:
    """Validate required parameters exist and have correct types."""
    required_params = {
        "tickers": list,
        "horizon_years": int,
        "start_portfolio": (int, float),
        "target_portfolio": (int, float),
        "max_drawdown": (int, float),
        "worst_day_limit": (int, float),
        "sigma_max": (int, float),
        "cash_min": (int, float),
        "upper_bounds": (int, float),
        "cash_ticker": str,
        "scenarios": int,
        "mc_freq": int,
        "max_iterations": int,
    }
    
    for param, expected_type in required_params.items():
        if param not in params:
            raise ValueError(f"Required parameter '{param}' is missing")


def _validate_portfolio_params(params: Dict[str, Any]) -> None:
    """Validate portfolio-related parameters."""
    tickers = params["tickers"]
    horizon_years = params["horizon_years"]
    start_portfolio = params["start_portfolio"]
    target_portfolio = params["target_portfolio"]
    cash_ticker = params["cash_ticker"]
    
    # Tickers validation
    if len(tickers) < 2:
        raise ValueError("At least 2 tickers are required for portfolio optimization")
    
    if len(set(tickers)) != len(tickers):
        raise ValueError("Duplicate tickers found in tickers list")
    
    if not all(isinstance(ticker, str) and ticker.strip() for ticker in tickers):
        raise ValueError("All tickers must be non-empty strings")
    
    # Cash ticker validation
    if cash_ticker not in tickers:
        raise ValueError(f"Cash ticker '{cash_ticker}' must be included in tickers list")
    
    # Portfolio values validation
    if start_portfolio <= 0:
        raise ValueError("Start portfolio value must be positive")
    
    if target_portfolio <= 0:
        raise ValueError("Target portfolio value must be positive")
    
    if target_portfolio <= start_portfolio:
        raise ValueError("Target portfolio must be greater than start portfolio")
    
    # Time horizon validation
    if horizon_years < 1:
        raise ValueError("Planning horizon must be at least 1 year")
    
    if horizon_years > 50:
        raise ValueError("Planning horizon should not exceed 50 years")


def _validate_risk_params(params: Dict[str, Any]) -> None:
    """Validate risk-related parameters."""
    max_drawdown = params["max_drawdown"]
    worst_day_limit = params["worst_day_limit"]
    sigma_max = params["sigma_max"]
    
    # Range checks
    if not 0 < max_drawdown < 1:
        raise ValueError("Max drawdown must be between 0 and 1 (exclusive)")
    
    if not 0 < worst_day_limit < 1:
        raise ValueError("Worst day limit must be between 0 and 1 (exclusive)")
    
    if not 0 < sigma_max < 1:
        raise ValueError("Maximum volatility (sigma_max) must be between 0 and 1 (exclusive)")
    
    # Logical consistency
    if worst_day_limit >= max_drawdown:
        raise ValueError("Worst day limit should be less than maximum drawdown")
    
    # Reasonable bounds check
    if max_drawdown > 0.6:
        raise ValueError("Maximum drawdown above 60% is unreasonably high")
    
    if worst_day_limit > 0.3:
        raise ValueError("Worst day limit above 30% is unreasonably high")


def _validate_allocation_params(params: Dict[str, Any]) -> None:
    """Validate allocation-related parameters."""
    cash_min = params["cash_min"]
    upper_bounds = params["upper_bounds"]
    num_assets = len(params["tickers"])
    
    if not 0 <= cash_min < 1:
        raise ValueError("Minimum cash allocation must be between 0 and 1")
    
    if not 0 < upper_bounds <= 1:
        raise ValueError("Upper bounds must be between 0 and 1 (inclusive of 1)")
    
    # Check if portfolio is feasible given constraints
    max_possible_allocation = (num_assets - 1) * upper_bounds + cash_min
    if max_possible_allocation < 1.0:
        raise ValueError(
            f"Portfolio constraints are infeasible: "
            f"max possible allocation is {max_possible_allocation:.2%} < 100%"
        )
    
    # Warn about overly restrictive constraints
    if upper_bounds < 1 / num_assets:
        raise ValueError(
            f"Upper bounds ({upper_bounds:.2%}) too restrictive for {num_assets} assets. "
            f"Should be at least {1/num_assets:.2%}"
        )


def _validate_simulation_params(params: Dict[str, Any]) -> None:
    """Validate Monte Carlo simulation parameters."""
    scenarios = params["scenarios"]
    mc_freq = params["mc_freq"]
    max_iterations = params["max_iterations"]
    
    if scenarios < 1000:
        raise ValueError("At least 1,000 scenarios recommended for reliable results")
    
    if scenarios > 100_000:
        raise ValueError("More than 100,000 scenarios may cause performance issues")
    
    if mc_freq < 1:
        raise ValueError("Rebalancing frequency must be at least 1 per year")
    
    if mc_freq > 252:  # Daily rebalancing
        raise ValueError("Rebalancing more than daily (252x/year) is unrealistic")
    
    if max_iterations < 10:
        raise ValueError("At least 10 optimization iterations recommended")
    
    if max_iterations > 1000:
        raise ValueError("More than 1,000 iterations may cause performance issues")


def _validate_bl_params(params: Dict[str, Any]) -> None:
    """Validate Black-Litterman parameters if provided."""
    P = params.get("bl_view_matrix_P")
    Q = params.get("bl_view_vector_Q")
    omega = params.get("bl_view_uncertainty_omega")
    num_assets = len(params["tickers"])
    
    # If any BL parameter is provided, validate the complete set
    bl_params_provided = [P is not None, Q is not None, omega is not None]
    
    if any(bl_params_provided):
        if not all(bl_params_provided):
            raise ValueError(
                "All Black-Litterman parameters (P, Q, omega) must be provided together"
            )
        
        # Validate types
        if not isinstance(P, np.ndarray) or not isinstance(Q, np.ndarray):
            raise TypeError("Black-Litterman P and Q must be numpy arrays")
        
        if not isinstance(omega, np.ndarray):
            raise TypeError("Black-Litterman omega must be numpy array")
        
        # Validate dimensions
        num_views = P.shape[0]
        
        if P.shape[1] != num_assets:
            raise ValueError(
                f"View matrix P must have {num_assets} columns to match number of assets"
            )
        
        if len(Q) != num_views:
            raise ValueError(
                f"View vector Q length ({len(Q)}) must match number of views ({num_views})"
            )
        
        if len(omega) != num_views:
            raise ValueError(
                f"Uncertainty vector omega length ({len(omega)}) must match number of views ({num_views})"
            )
        
        # Validate values
        if np.any(omega <= 0):
            raise ValueError("All uncertainty values in omega must be positive")
        
        if np.any(np.abs(Q) > 1):  # Views should be reasonable annual returns
            raise ValueError("View returns in Q seem unreasonably large (>100% annually)")

