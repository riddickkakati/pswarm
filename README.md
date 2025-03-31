pswarm
A clean, simplified implementation of the Particle Swarm Optimization algorithm for parameter optimization problems.
Installation
pip install pso-optimizer
Features
    • Simple, well-documented implementation of PSO 
    • Customizable inertia weight decay 
    • Parameter-specific boundary handling 
    • Built-in early stopping 
    • Optimization history tracking 
    • Utility functions for visualization and result saving 
Quick Start
import numpy as np
from pso_optimizer import PSOOptimizer, plot_convergence, save_results

# Define objective function (example: minimize a simple quadratic function)
def objective_function(x):
    return -np.sum(x**2)  # Negative because PSO maximizes by default

# Define parameter bounds
param_min = np.array([-5.0, -5.0, -5.0])
param_max = np.array([5.0, 5.0, 5.0])

# Create optimizer
optimizer = PSOOptimizer(
    objective_function=objective_function,
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
print(f"Best parameters: {best_params}")
print(f"Best fitness: {best_fitness}")

# Plot convergence
plot_convergence(history, "convergence.png")

# Save results
save_results(best_params, best_fitness, ['x', 'y', 'z'], "results.txt")
Advanced Usage
Special Boundary Handling
You can specify special boundary handling for certain parameters:
# Define parameter with wrapping boundary (like an angle)
special_boundaries = {
    2: {'type': 'wrap'}  # Parameter at index 2 will wrap around its bounds
}

optimizer = PSOOptimizer(
    # ... other parameters ...
    special_boundary_handling=special_boundaries
)
Loading Parameters from Configuration
from pso_optimizer import load_parameter_bounds_from_config

# Load parameter bounds from YAML config
param_min, param_max = load_parameter_bounds_from_config('config.yaml')

optimizer = PSOOptimizer(
    # ... other parameters ...
    param_min=param_min,
    param_max=param_max
)
License
This project is licensed under the MIT License - see the LICENSE file for details.

