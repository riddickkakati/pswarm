"""
Simple example demonstrating the PSO Optimizer.
"""

import numpy as np
from pso_optimizer import PSOOptimizer, plot_convergence, save_results


def sphere_function(x):
    """
    Simple sphere function (quadratic) for testing optimization.
    Minimum is at the origin with value 0.
    """
    return -np.sum(x**2)  # Negative because PSO maximizes by default


def rosenbrock_function(x):
    """
    Rosenbrock function, a classic optimization test problem.
    Minimum is at [1, 1, ..., 1] with value 0.
    """
    result = 0
    for i in range(len(x) - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return -result  # Negative because PSO maximizes by default


def main():
    # Problem setup
    param_min = np.array([-5.0, -5.0, -5.0, -5.0])
    param_max = np.array([5.0, 5.0, 5.0, 5.0])
    
    print("\n--- Optimizing Sphere Function ---")
    optimizer = PSOOptimizer(
        objective_function=sphere_function,
        param_min=param_min,
        param_max=param_max,
        swarm_size=20,
        max_iterations=50,
        maximize=False,  # We're minimizing
        save_history=True
    )
    
    # Run optimization
    best_params, best_fitness, history = optimizer.optimize()
    
    # Print results
    print("\nSpere Function Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best fitness: {best_fitness}")
    
    # Plot convergence
    fig = plot_convergence(history, "sphere_convergence.png")
    
    # Save results
    save_results(
        best_params, 
        best_fitness, 
        [f'x{i+1}' for i in range(len(best_params))], 
        "sphere_results.txt"
    )
    
    print("\n--- Optimizing Rosenbrock Function ---")
    optimizer = PSOOptimizer(
        objective_function=rosenbrock_function,
        param_min=param_min,
        param_max=param_max,
        swarm_size=30,
        max_iterations=100,
        maximize=False,  # We're minimizing
        save_history=True
    )
    
    # Run optimization
    best_params, best_fitness, history = optimizer.optimize()
    
    # Print results
    print("\nRosenbrock Function Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best fitness: {best_fitness}")
    
    # Plot convergence
    fig = plot_convergence(history, "rosenbrock_convergence.png")
    
    # Save results
    save_results(
        best_params, 
        best_fitness, 
        [f'x{i+1}' for i in range(len(best_params))], 
        "rosenbrock_results.txt"
    )


if __name__ == "__main__":
    main()
