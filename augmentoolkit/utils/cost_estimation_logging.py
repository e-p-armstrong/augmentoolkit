import logging
from typing import Dict, List, Union, Optional
from tabulate import tabulate


def calculate_pipeline_cost_efficiency(
    total_input_tokens: int,
    token_counters: List[Dict[str, Union[int, float, str]]],
) -> Dict[str, Union[float, int, Dict]]:
    """
    Calculates the cost efficiency metrics for a pipeline based on token usage.

    Args:
        total_input_tokens: Total number of tokens in the original input
        token_counters: List of dictionaries containing token counts, costs, and names
                        in the format {"name": str, "input_tokens": int, "input_cost": float,
                                      "output_tokens": int, "output_cost": float}
        log_level: Logging level to use for reporting metrics

    Returns:
        Dict containing cost efficiency metrics:
        {
            "total_input_tokens": int,
            "total_model_input_tokens": int,
            "total_model_output_tokens": int,
            "total_cost": float,
            "cost_per_million_original_tokens": float,
            "input_to_model_token_ratio": float,
            "component_metrics": Dict[str, Dict]
        }
    """

    # Initialize counters
    total_model_input_tokens = 0
    total_model_output_tokens = 0
    total_input_cost = 0.0
    total_output_cost = 0.0

    # Dictionary to store metrics for each component
    component_metrics = {}

    # Aggregate token counts and costs from all counters
    for counter in token_counters:
        name = counter.get("name", f"Component {len(component_metrics) + 1}")
        input_tokens = counter.get("input_tokens", 0)
        output_tokens = counter.get("output_tokens", 0)
        input_cost = counter.get("input_cost", 0.0)
        output_cost = counter.get("output_cost", 0.0)

        total_model_input_tokens += input_tokens
        total_model_output_tokens += output_tokens
        total_input_cost += input_cost
        total_output_cost += output_cost

        component_total_cost = input_cost + output_cost

        component_metrics[name] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": component_total_cost,
            "input_tokens_percentage": 0.0,  # Will calculate after getting totals
            "output_tokens_percentage": 0.0,  # Will calculate after getting totals
            "cost_percentage": 0.0,  # Will calculate after getting totals
        }

    total_cost = total_input_cost + total_output_cost

    # Calculate percentages for each component
    for name, metrics in component_metrics.items():
        if total_model_input_tokens > 0:
            metrics["input_tokens_percentage"] = (
                metrics["input_tokens"] / total_model_input_tokens
            ) * 100
        if total_model_output_tokens > 0:
            metrics["output_tokens_percentage"] = (
                metrics["output_tokens"] / total_model_output_tokens
            ) * 100
        if total_cost > 0:
            metrics["cost_percentage"] = (metrics["total_cost"] / total_cost) * 100

    # Calculate cost per million original tokens
    cost_per_million_original = 0.0
    if total_input_tokens > 0:
        cost_per_million_original = (total_cost / total_input_tokens) * 1_000_000

    # Calculate input to model token ratio (how many times each token is processed)
    input_to_model_ratio = 0.0
    if total_input_tokens > 0:
        input_to_model_ratio = total_model_input_tokens / total_input_tokens

    # Prepare results
    results = {
        "total_input_tokens": total_input_tokens,
        "total_model_input_tokens": total_model_input_tokens,
        "total_model_output_tokens": total_model_output_tokens,
        "total_input_cost": total_input_cost,
        "total_output_cost": total_output_cost,
        "total_cost": total_cost,
        "cost_per_million_original_tokens": cost_per_million_original,
        "input_to_model_token_ratio": input_to_model_ratio,
        "component_metrics": component_metrics,
    }

    # Log the results
    print("\n=== Pipeline Cost Efficiency Metrics ===")
    print(f"Original input tokens: {total_input_tokens:,}")
    print(f"Total model input tokens: {total_model_input_tokens:,}")
    print(f"Total model output tokens: {total_model_output_tokens:,}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Cost per million original tokens: ${cost_per_million_original:.2f}")
    print(f"Input-to-model token ratio: {input_to_model_ratio:.2f}x")

    # Create a table for component breakdown
    table_data = []
    headers = [
        "Component",
        "Input Tokens",
        "Output Tokens",
        "Input Cost",
        "Output Cost",
        "Total Cost",
        "% of Total Cost",
    ]

    for name, metrics in component_metrics.items():
        table_data.append(
            [
                name,
                f"{metrics['input_tokens']:,} ({metrics['input_tokens_percentage']:.1f}%)",
                f"{metrics['output_tokens']:,} ({metrics['output_tokens_percentage']:.1f}%)",
                f"${metrics['input_cost']:.2f}",
                f"${metrics['output_cost']:.2f}",
                f"${metrics['total_cost']:.2f}",
                f"{metrics['cost_percentage']:.1f}%",
            ]
        )

    # Add totals row
    table_data.append(
        [
            "TOTAL",
            f"{total_model_input_tokens:,} (100.0%)",
            f"{total_model_output_tokens:,} (100.0%)",
            f"${total_input_cost:.2f}",
            f"${total_output_cost:.2f}",
            f"${total_cost:.2f}",
            "100.0%",
        ]
    )

    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(f"\n=== Component Breakdown ===\n{table}")

    return results
