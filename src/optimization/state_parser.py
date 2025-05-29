"""Parse data collected in Agent State to the parameters expected by PortfolioOptimizer."""

import os
import re
import time
from typing import Any, Dict, List

import numpy as np


def parse_state_to_optimizer_params(state: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Convert State object to PortfolioOptimizer parameters.
    
    Args:
        state: State dictionary with user profile and investment data
        **kwargs: Override defaults (scenarios=5000, max_iterations=50, etc.)
        
    Returns:
        Parameters dict ready for PortfolioOptimizer(**params)
    """
    # Determine cash ticker first
    potential_tickers = state.get('tickers', [])
    cash_ticker = _find_cash_ticker(potential_tickers)
    
    # Parse Black-Litterman views and ensure cash ticker is included
    bl_params = _parse_bl_views(state.get("bl_views", {}), cash_ticker)
    
    # Use BL tickers if available, otherwise use state tickers, with fallback
    tickers = bl_params.get('tickers') or state.get('tickers')
    
    # === PORTFOLIO PARAMETERS ===
    start_portfolio = float(state.get("start_portfolio", 100_000))
    target_portfolio = float(state.get("target_amount", 200_000))
    horizon_years = _parse_planning_horizon(state.get("planning_horizon", "10 years"))
    
    # === RISK PARAMETERS ===
    max_drawdown = _safe_percentage_to_decimal(state.get("maximum_drawdown_percentage", 25.0), default=0.25)
    worst_day_limit = _safe_percentage_to_decimal(state.get("worst_day_decline_percentage", 15.0), default=0.15)
    sigma_max = _calculate_vol_limit(max_drawdown, worst_day_limit)
    
    # === CASH PARAMETERS ===
    cash_reserve = state.get("cash_reserve", 0.0)
    cash_min = max(0.02, cash_reserve / start_portfolio) if cash_reserve > 0 else 0.05
    
    # === ALLOCATION PARAMETERS ===
    upper_bounds = min(0.8, _safe_percentage_to_decimal(state.get("max_single_asset_allocation_percentage", 50.0), default=0.5))
    
    # === SIMULATION PARAMETERS ===
    scenarios = 10_000
    mc_freq = 12
    max_iterations = 100
    
    # === PENALTY PARAMETERS ===
    lambda_sigma = 100
    lambda_drawdown = 500
    lambda_worst_day = 300
    lambda_cash = 1000
    
    # === OUTPUT PARAMETERS ===
    output_dir = None # _create_output_directory()
    log_prefix = "optimization"
    
    # === BUILD PARAMETERS DICT ===
    params = {
        # Portfolio basics
        "tickers": tickers,
        "horizon_years": horizon_years,
        "start_portfolio": start_portfolio,
        "target_portfolio": target_portfolio,
        
        # Risk constraints
        "max_drawdown": max_drawdown,
        "worst_day_limit": worst_day_limit,
        "sigma_max": sigma_max,
        
        # Cash management
        "cash_min": cash_min,
        "cash_ticker": cash_ticker,
        
        # Allocation constraints
        "upper_bounds": upper_bounds,
        
        # Monte Carlo simulation
        "scenarios": scenarios,
        "mc_freq": mc_freq,
        "max_iterations": max_iterations,
        
        # Penalty parameters
        "lambda_sigma": lambda_sigma,
        "lambda_drawdown": lambda_drawdown,
        "lambda_worst_day": lambda_worst_day,
        "lambda_cash": lambda_cash,
        
        # Output configuration
        "output_dir": output_dir,
        "log_prefix": log_prefix,
        
        # Black-Litterman parameters
        "bl_view_matrix_P": bl_params.get("bl_view_matrix_P"),
        "bl_view_vector_Q": bl_params.get("bl_view_vector_Q"),
        "bl_view_uncertainty_omega": bl_params.get("bl_view_uncertainty_omega"),
    }
    
    # Apply user overrides
    params.update(kwargs)
    
    return params


def _parse_planning_horizon(horizon_str: str) -> int:
    """Parse '2 years' or '18 months' to integer years."""
    if not horizon_str:
        return 10
    
    numbers = re.findall(r'\d+', str(horizon_str))
    if not numbers:
        return 10
    
    value = int(numbers[0])
    if 'month' in horizon_str.lower():
        return max(1, value // 12)
    return max(1, value)


def _parse_bl_views(bl_views: Dict[str, Any], cash_ticker: str) -> Dict[str, Any]:
    """Parse Black-Litterman views with robust error handling and cash ticker integration."""
    default_return = {
        'bl_view_matrix_P': None,
        'bl_view_vector_Q': None,
        'bl_view_uncertainty_omega': None,
        'tickers': None
    }
    
    # Check if views are complete and have required fields
    if not bl_views.get('views_complete', False):
        return default_return
    
    required_fields = ['p_matrix', 'q_vector', 'sigma_vector', 'tickers']
    if not all(field in bl_views for field in required_fields):
        return default_return
    
    try:
        # Extract data
        p_matrix = bl_views['p_matrix']
        q_vector = bl_views['q_vector']
        sigma_vector = bl_views['sigma_vector']
        tickers = bl_views['tickers']
        
        # Validate we have data
        if not all([p_matrix, q_vector, sigma_vector, tickers]):
            return default_return
        
        # Convert to numpy arrays
        P = np.array(p_matrix, dtype=float)
        Q = np.array(q_vector, dtype=float)
        omega = np.array(sigma_vector, dtype=float)
        
        # Basic dimension validation
        num_views, num_assets = P.shape
        if len(Q) != num_views or len(sigma_vector) != num_views or len(tickers) != num_assets:
            return default_return
        
        # Add cash ticker if not present
        if cash_ticker not in tickers:
            # Add cash ticker to the list
            updated_tickers = tickers + [cash_ticker]
            
            # Add a column of zeros to P matrix for the cash ticker
            # Shape: (num_views, num_assets + 1)
            cash_column = np.zeros((num_views, 1))
            updated_P = np.hstack([P, cash_column])
            
            return {
                'bl_view_matrix_P': updated_P,
                'bl_view_vector_Q': Q,
                'bl_view_uncertainty_omega': omega,
                'tickers': updated_tickers
            }
        else:
            # Cash ticker already present, return as-is
            return {
                'bl_view_matrix_P': P,
                'bl_view_vector_Q': Q,
                'bl_view_uncertainty_omega': omega,
                'tickers': tickers
            }
        
    except Exception:
        # Silent fallback on any error
        return default_return


def _safe_percentage_to_decimal(value: float, default: float) -> float:
    """Safely convert percentage to decimal with fallback."""
    if not value or value <= 0:
        return default
    
    # If > 1, assume percentage (25 = 25%)
    return value / 100.0 if value > 1 else value


def _calculate_vol_limit(max_dd: float, worst_day: float) -> float:
    """Calculate volatility limit based on risk tolerance."""
    if max_dd <= 0.15 and worst_day <= 0.08:
        return 0.15  # Conservative
    elif max_dd <= 0.25 and worst_day <= 0.12:
        return 0.20  # Moderate  
    elif max_dd <= 0.35 and worst_day <= 0.18:
        return 0.25  # Aggressive
    return 0.30  # Very aggressive


def _find_cash_ticker(tickers: List[str]) -> str:
    """Find best cash ticker from list."""
    cash_options = ['BIL', 'SHY', 'VMOT', 'MINT', 'SGOV', 'JPST']
    
    # Try preferred cash tickers first
    for option in cash_options:
        if option in tickers:
            return option
    
    # Look for cash-like patterns
    for ticker in tickers:
        ticker_upper = ticker.upper()
        if any(pattern in ticker_upper for pattern in ['BIL', 'CASH', 'SHORT', 'MONEY', 'TREASURY']):
            return ticker
    
    # Fallback to BIL (standard default)
    return 'BIL'


def _create_output_directory() -> str:
    """Create timestamped output directory."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(os.getcwd(), f"optimization_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

