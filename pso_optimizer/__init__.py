"""
PSO Optimizer - A simple particle swarm optimization package

This package provides a clean implementation of the Particle Swarm Optimization
algorithm for parameter optimization problems.
"""

__version__ = "0.1.0"

from .PSO import PSOOptimizer  # Changed from .pso to .PSO to match the actual filename
from .utils import plot_convergence, save_results, load_parameter_bounds_from_config

__all__ = [
    'PSOOptimizer',
    'plot_convergence',
    'save_results',
    'load_parameter_bounds_from_config'
]
