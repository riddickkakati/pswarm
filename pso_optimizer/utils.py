"""
Utility functions for the PSO optimizer.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import os


def plot_convergence(history: Dict[str, Any], output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the convergence of the optimization process.
    
    Parameters
    ----------
    history : dict
        History dictionary returned by the PSOOptimizer.
    
    output_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    
    Returns
    -------
    matplotlib.pyplot.Figure
        Figure object containing the convergence plot.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot fitness history
    iterations = np.arange(1, len(history['best_fitness']) + 1)
    ax.plot(iterations, history['best_fitness'], 'b-', linewidth=2)
    
    # Add labels and grid
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Function Value', fontsize=12)
    ax.set_title('PSO Convergence', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_results(
    best_parameters: np.ndarray, 
    best_fitness: float, 
    parameter_names: Optional[List[str]] = None,
    output_path: str = 'pso_results.txt'
) -> None:
    """
    Save optimization results to a text file.
    
    Parameters
    ----------
    best_parameters : numpy.ndarray
        Best parameters found by the optimizer.
    
    best_fitness : float
        Best fitness value.
    
    parameter_names : list of str, optional
        Names of the parameters. If None, generic names (p1, p2, ...) are used.
    
    output_path : str, optional
        Path to save the results.
    """
    if parameter_names is None:
        parameter_names = [f'p{i+1}' for i in range(len(best_parameters))]
    
    with open(output_path, 'w') as f:
        f.write(f"Best fitness: {best_fitness}\n\n")
        f.write("Best parameters:\n")
        for name, value in zip(parameter_names, best_parameters):
            f.write(f"{name}: {value}\n")


def load_parameter_bounds_from_config(config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load parameter bounds from a configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    
    Returns
    -------
    tuple
        Parameter minimum and maximum bounds as numpy arrays.
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    parameter_dict = config.get('Optimizer', {}).get('parameters', {})
    
    param_min = []
    param_max = []
    
    for param_name in sorted(parameter_dict.keys()):
        param_min.append(parameter_dict[param_name].get('low', 0.0))
        param_max.append(parameter_dict[param_name].get('high', 1.0))
    
    return np.array(param_min), np.array(param_max)
