"""
PSO (Particle Swarm Optimization) Implementation

This module provides a clean, simplified implementation of the PSO algorithm
for parameter optimization.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union, Dict, Any


class PSOOptimizer:
    """
    Particle Swarm Optimization (PSO) implementation for parameter optimization.
    
    This class implements the PSO algorithm with inertia weight decay and
    boundary handling.
    """
    
    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        param_min: np.ndarray,
        param_max: np.ndarray,
        swarm_size: int = 50,
        max_iterations: int = 100,
        inertia_weight_max: float = 0.9,
        inertia_weight_min: float = 0.4,
        cognitive_weight: float = 2.0,
        social_weight: float = 2.0,
        min_step_size: float = 1e-3,
        early_stopping_threshold: float = 0.9,
        maximize: bool = False,
        save_history: bool = False,
        special_boundary_handling: Dict[int, Dict[str, Any]] = None
    ):
        """
        Initialize PSO optimizer.
        
        Parameters
        ----------
        objective_function : callable
            The function to optimize. Should take a numpy array of parameters
            and return a scalar fitness value.
        
        param_min : numpy.ndarray
            Minimum values for each parameter.
        
        param_max : numpy.ndarray
            Maximum values for each parameter.
        
        swarm_size : int, optional
            Number of particles in the swarm.
        
        max_iterations : int, optional
            Maximum number of iterations to run.
        
        inertia_weight_max : float, optional
            Initial inertia weight.
        
        inertia_weight_min : float, optional
            Final inertia weight.
        
        cognitive_weight : float, optional
            Weight for the cognitive component (particle's own best).
        
        social_weight : float, optional
            Weight for the social component (global best).
        
        min_step_size : float, optional
            Minimum normalized step size for early stopping.
        
        early_stopping_threshold : float, optional
            Proportion of particles that must have converged for early stopping.
        
        maximize : bool, optional
            Whether to maximize (True) or minimize (False) the objective function.
        
        save_history : bool, optional
            Whether to save the optimization history.
        
        special_boundary_handling : dict, optional
            Dictionary specifying special boundary handling for specific parameters.
            Format: {param_index: {'type': 'wrap'}} for parameters that should wrap around.
        """
        # Store parameters
        self.objective_function = objective_function
        self.param_min = np.array(param_min)
        self.param_max = np.array(param_max)
        self.n_parameters = len(param_min)
        self.n_particles = swarm_size
        self.max_iterations = max_iterations
        self.w_max = inertia_weight_max
        self.w_min = inertia_weight_min
        self.c1 = cognitive_weight
        self.c2 = social_weight
        self.min_step_size = min_step_size
        self.early_stopping_threshold = early_stopping_threshold
        self.maximize = maximize
        self.save_history = save_history
        
        # Convert None to empty dict
        if special_boundary_handling is None:
            special_boundary_handling = {}
        self.special_boundary_handling = special_boundary_handling
        
        # Initialize history storage if requested
        self.history = {
            'best_fitness': [],
            'best_parameters': [],
            'iteration_data': []
        } if save_history else None
    
    def _evaluator(self, parameters: np.ndarray) -> float:
        """
        Evaluate objective function, adjusting sign for maximization.
        
        Parameters
        ----------
        parameters : numpy.ndarray
            Parameters to evaluate.
        
        Returns
        -------
        float
            Fitness value (adjusted for maximization/minimization).
        """
        fitness = self.objective_function(parameters)
        return fitness if self.maximize else -fitness
    
    def _handle_boundaries(self, x: np.ndarray, v: np.ndarray, j: int, k: int) -> Tuple[float, float]:
        """
        Handle boundary conditions for parameters.
        
        Parameters
        ----------
        x : numpy.ndarray
            Current particle positions.
        
        v : numpy.ndarray
            Current particle velocities.
        
        j : int
            Parameter index.
        
        k : int
            Particle index.
        
        Returns
        -------
        tuple
            Updated position and velocity.
        """
        # Check if this parameter has special handling
        if j in self.special_boundary_handling:
            if self.special_boundary_handling[j]['type'] == 'wrap':
                # Implement wrapping (modulo) boundary condition
                if x[j, k] > self.param_max[j]:
                    x[j, k] = x[j, k] - np.floor((x[j, k] - self.param_min[j]) / 
                                                 (self.param_max[j] - self.param_min[j])) * \
                                         (self.param_max[j] - self.param_min[j])
                
                if x[j, k] < self.param_min[j]:
                    x[j, k] = self.param_max[j] - (np.ceil((self.param_min[j] - x[j, k]) / 
                                                           (self.param_max[j] - self.param_min[j]) - 1e-10) * \
                                                   (self.param_max[j] - self.param_min[j]))
                return x[j, k], v[j, k]
        
        # Default: absorbing boundary conditions
        if x[j, k] > self.param_max[j]:
            x[j, k] = self.param_max[j]
            v[j, k] = 0.0
        
        if x[j, k] < self.param_min[j]:
            x[j, k] = self.param_min[j]
            v[j, k] = 0.0
        
        return x[j, k], v[j, k]
    
    def optimize(self) -> Tuple[np.ndarray, float, Optional[Dict]]:
        """
        Run the PSO optimization algorithm.
        
        Returns
        -------
        tuple
            Best parameters found, best fitness value, and history (if save_history is True).
        """
        print(f'Starting PSO with {self.n_particles} particles, {self.max_iterations} iterations')
        
        # Initialize particles with random positions and velocities
        x = np.random.uniform(0, 1, (self.n_parameters, self.n_particles))
        v = np.random.uniform(0, 1, (self.n_parameters, self.n_particles))
        pbest = np.zeros_like(x)
        fit = np.zeros(self.n_particles)
        fitbest = np.zeros(self.n_particles)
        
        # Scale parameters to their ranges
        for j in range(self.n_parameters):
            dx_max = self.param_max[j] - self.param_min[j]
            dv_max = 0.5 * dx_max  # Scale velocity to half the parameter range
            x[j, :] = x[j, :] * dx_max + self.param_min[j]
            v[j, :] = v[j, :] * dv_max - dv_max/2  # Center velocities around zero
            pbest[j, :] = x[j, :]
        
        # Initialize fitness
        for k in range(self.n_particles):
            parameters = x[:, k].copy()
            fit[k] = self._evaluator(parameters)
            fitbest[k] = fit[k]
        
        # Find initial global best
        best_idx = np.argmax(fit)
        gbest = x[:, best_idx].copy()
        best_fitness = fit[best_idx]
        
        # Initialize inertia weight
        w = self.w_max
        dw = (self.w_max - self.w_min) / self.max_iterations
        
        # Main optimization loop
        for i in range(self.max_iterations):
            # Update particles
            for k in range(self.n_particles):
                # Generate random coefficients
                r1 = np.random.uniform(0, 1, self.n_parameters)
                r2 = np.random.uniform(0, 1, self.n_parameters)
                
                # Track if particle hits boundary
                status = 0
                
                # Update velocity and position for each parameter
                for j in range(self.n_parameters):
                    # Update velocity using PSO equation
                    v[j, k] = (w * v[j, k] +
                              self.c1 * r1[j] * (pbest[j, k] - x[j, k]) +
                              self.c2 * r2[j] * (gbest[j] - x[j, k]))
                    
                    # Update position
                    x[j, k] = x[j, k] + v[j, k]
                    
                    # Handle boundary conditions
                    x[j, k], v[j, k] = self._handle_boundaries(x, v, j, k)
                    
                    # Check if particle hit boundary
                    if x[j, k] == self.param_min[j] or x[j, k] == self.param_max[j]:
                        status = 1
                
                # Skip evaluation if particle hit boundary (optional optimization)
                if status == 0:
                    parameters = x[:, k].copy()
                    fit[k] = self._evaluator(parameters)
                
                # Update particle's best position
                if fit[k] > fitbest[k]:
                    fitbest[k] = fit[k]
                    pbest[:, k] = x[:, k].copy()
            
            # Update global best
            best_idx = np.argmax(fitbest)
            if fitbest[best_idx] > best_fitness:
                gbest = pbest[:, best_idx].copy()
                best_fitness = fitbest[best_idx]
            
            # Save history if requested
            if self.save_history:
                self.history['best_fitness'].append(best_fitness if self.maximize else -best_fitness)
                self.history['best_parameters'].append(gbest.copy())
                
                # Save detailed iteration data
                iteration_data = {
                    'all_positions': x.copy(),
                    'all_velocities': v.copy(),
                    'all_pbest': pbest.copy(),
                    'all_fitness': fit.copy(),
                    'all_pbest_fitness': fitbest.copy(),
                    'gbest': gbest.copy(),
                    'best_fitness': best_fitness if self.maximize else -best_fitness
                }
                self.history['iteration_data'].append(iteration_data)
            
            # Update inertia weight (linear decay)
            w = w - dw
            
            # Progress reporting (every 10%)
            if (i+1) % max(1, self.max_iterations // 10) == 0:
                print(f"{100.0 * (i+1) / self.max_iterations:.1f}% complete. "
                      f"Best {'fitness' if self.maximize else 'error'}: "
                      f"{best_fitness if self.maximize else -best_fitness:.6f}")
            
            # Check for early stopping
            count = 0
            for k in range(self.n_particles):
                # Calculate normalized distance to global best
                norm = np.sqrt(np.sum(
                    ((pbest[:, k] - gbest) / (self.param_max - self.param_min)) ** 2
                ))
                if norm < self.min_step_size:
                    count += 1
            
            # If enough particles have converged, stop early
            if count >= self.early_stopping_threshold * self.n_particles:
                print(f'Early stopping at iteration {i+1}/{self.max_iterations} '
                      f'({count}/{self.n_particles} particles converged)')
                break
        
        # Return best parameters and fitness (adjust fitness sign for minimization)
        return gbest, (best_fitness if self.maximize else -best_fitness), self.history
