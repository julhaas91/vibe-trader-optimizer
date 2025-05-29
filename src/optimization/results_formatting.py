"""Portfolio Optimization Results Formatting Module.

This module provides formatters to convert optimization results into human-readable
and LLM-friendly formats. It handles the presentation layer for optimization outputs,
making results easy to interpret and explain.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


def format_results_for_llm(results: Dict[str, Any]) -> str:
    """Format optimization results for LLM processing and natural language explanation.
    
    Takes raw optimization results and converts them into a structured, readable format
    that's optimized for LLM interpretation and user communication.
    
    Args:
        results: Dictionary containing optimization results with structure:
            - "results": Dict with metrics (success_prob, weights, etc.)
            - "inputs": Dict with original parameters (tickers, bl_views, etc.)
    
    Returns:
        str: Formatted text ready for LLM processing or direct display
        
    Raises:
        KeyError: If required result fields are missing
        ValueError: If result data is malformed
        
    Example:
        >>> results = optimizer.optimize()
        >>> formatted = format_results_for_llm(results)
        >>> print(formatted)
        Portfolio Optimization Results:
        
        1. Success Metrics:
           - Success Probability: 85.3%
           - Average Final Value: $247,500.00
           ...
    """
    try:
        # Extract and validate required sections
        metrics = results["results"]
        inputs = results["inputs"]
        
        # Build formatted response
        response_parts = [
            "Portfolio Optimization Results:",
            "",
            _format_success_metrics(metrics),
            _format_asset_allocation(metrics, inputs),
            _format_bl_views(inputs),
            _format_optimization_details(metrics)
        ]
        
        # Join non-empty parts
        return "\n".join(part for part in response_parts if part.strip())
        
    except KeyError as e:
        raise KeyError(f"Missing required result field: {e}")
    except Exception as e:
        raise ValueError(f"Failed to format results: {e}")


def _format_success_metrics(metrics: Dict[str, Any]) -> str:
    """Format the success metrics section."""
    try:
        return f"""1. Success Metrics:
   - Success Probability: {metrics['success_prob']:.1%}
   - Average Final Value: ${metrics['avg_final']:,.2f}
   - Average Drawdown: {metrics['avg_drawdown']:.1%}
   - Average Worst-Day Drop: {metrics['avg_worst_day']:.1%}
   - Portfolio Volatility: {metrics['volatility']:.1%}
   - Cash Allocation: {metrics['cash_allocation']:.1%}"""
    except KeyError as e:
        return f"1. Success Metrics: [Error - Missing {e}]"
    except (ValueError, TypeError):
        return "1. Success Metrics: [Error - Invalid metric values]"


def _format_asset_allocation(metrics: Dict[str, Any], inputs: Dict[str, Any]) -> str:
    """Format the asset allocation section."""
    try:
        weights = metrics["weights"]
        tickers = inputs["tickers"]
        
        if len(weights) != len(tickers):
            return "2. Asset Allocation: [Error - Weights and tickers length mismatch]"
        
        allocation_lines = ["2. Asset Allocation:"]
        
        # Sort by weight (descending) for better readability
        ticker_weights = list(zip(tickers, weights))
        ticker_weights.sort(key=lambda x: x[1], reverse=True)
        
        for ticker, weight in ticker_weights:
            allocation_lines.append(f"   - {ticker}: {weight:.1%}")
        
        return "\n".join(allocation_lines)
        
    except KeyError as e:
        return f"2. Asset Allocation: [Error - Missing {e}]"
    except Exception:
        return "2. Asset Allocation: [Error - Failed to format allocations]"


def _format_bl_views(inputs: Dict[str, Any]) -> str:
    """Format Black-Litterman views section if present."""
    bl_views = inputs.get("bl_views", {})
    
    # Check if BL views are present and valid
    if not _has_valid_bl_views(bl_views):
        return ""
    
    try:
        P = np.asarray(bl_views["P"])
        Q = np.asarray(bl_views["Q"]) 
        omega = bl_views.get("omega")
        tickers = inputs.get("tickers", [])
        
        if len(tickers) != P.shape[1]:
            return "3. Black-Litterman Views: [Error - Dimension mismatch]"
        
        view_lines = ["", "3. Black-Litterman Views:"]
        
        for i in range(len(Q)):
            view_description = _format_single_view(P[i], Q[i], omega, i, tickers)
            view_lines.extend(view_description)
        
        return "\n".join(view_lines)
        
    except Exception:
        return "3. Black-Litterman Views: [Error - Failed to format views]"


def _format_single_view(p_row: NDArray[Any], q_value: Any, 
                       omega: Optional[NDArray[Any]], 
                       view_idx: int, tickers: List[str]) -> List[str]:
    """Format a single Black-Litterman view."""
    try:
        # Find assets involved in this view
        view_assets = []
        for j, ticker in enumerate(tickers):
            if abs(p_row[j]) > 1e-10:  # Handle floating point precision
                sign = "+" if p_row[j] > 0 else ""
                view_assets.append(f"{ticker} ({sign}{p_row[j]:.0f})")
        
        # Format expected return
        q_formatted = _safe_format_percentage(q_value)
        
        view_lines = [f"   View {view_idx + 1}: {' vs '.join(view_assets)} = {q_formatted}"]
        
        # Add uncertainty if available
        if omega is not None:
            try:
                omega_value = _extract_scalar_value(omega[view_idx])
                view_lines.append(f"      Uncertainty: {omega_value:.4f}")
            except (IndexError, TypeError, ValueError):
                pass  # Skip uncertainty if formatting fails
        
        return view_lines
        
    except Exception:
        return [f"   View {view_idx + 1}: [Error formatting view]"]


def _format_optimization_details(metrics: Dict[str, Any]) -> str:
    """Format optimization process details."""
    try:
        return f"""
4. Optimization Details:
   - Computation Time: {metrics['elapsed_time']:.2f} seconds
   - Number of Iterations: {metrics['iterations']}"""
    except KeyError as e:
        return f"\n4. Optimization Details: [Error - Missing {e}]"
    except (ValueError, TypeError):
        return "\n4. Optimization Details: [Error - Invalid detail values]"


def _has_valid_bl_views(bl_views: Dict[str, Any]) -> bool:
    """Check if Black-Litterman views are present and valid."""
    if not bl_views:
        return False
    
    required_keys = ["P", "Q", "omega"]
    return (all(key in bl_views for key in required_keys) and
            bl_views["P"] is not None and 
            bl_views["Q"] is not None and
            bl_views["omega"] is not None)


def _extract_scalar_value(value: Any) -> float:
    """Safely extract a scalar numeric value from various input types."""
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    elif isinstance(value, complex):
        if value.imag == 0:
            return float(value.real)
        else:
            raise ValueError("Cannot convert complex number with non-zero imaginary part")
    elif isinstance(value, np.ndarray | list | tuple):
        if np.size(value) == 1:
            return float(np.asarray(value).item())
        else:
            raise ValueError(f"Expected scalar, got array of size {np.size(value)}")
    else:
        # Try direct conversion as last resort
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert {type(value).__name__} to float: {e}")


def _safe_format_percentage(value: Any) -> str:
    """Safely format a value as a percentage with error handling."""
    try:
        numeric_value = _extract_scalar_value(value)
        return f"{numeric_value:.1%}"
    except (ValueError, TypeError):
        return str(value)  # Fallback to string representation


# Alternative formatters for extensibility
def format_results_for_json(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format results as a clean JSON structure for API responses.
    
    Args:
        results: Raw optimization results
        
    Returns:
        Dict: JSON-serializable results structure
    """
    # Implementation for JSON formatting
    # This can be added later as needed
    raise NotImplementedError("JSON formatter not yet implemented")


def format_results_for_report(results: Dict[str, Any]) -> str:
    """Format results as a detailed report for documentation.
    
    Args:
        results: Raw optimization results
        
    Returns:
        str: Detailed report format
    """
    # Implementation for detailed reporting
    # This can be added later as needed
    raise NotImplementedError("Report formatter not yet implemented")
