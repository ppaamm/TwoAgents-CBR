import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count

class Particle:
    def __init__(self, dim, bounds):
        """Initialize a particle with random position and velocity"""
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_score = -np.inf  # For maximization problems

def pso(opt_function, dim, bounds, num_particles=30, max_iter=20, w=0.5, c1=1, c2=3):
    """Particle Swarm Optimization (PSO) algorithm"""
    
    # Initialize swarm
    bounds = np.array(bounds)  # Convert bounds to array for easy indexing
    swarm = [Particle(dim, bounds) for _ in range(num_particles)]
    
    global_best_position = None
    global_best_score = -np.inf  # Since we are maximizing

    # PSO iterations
    for iteration in range(max_iter):
        for particle in swarm:
            # Evaluate the fitness of the current position
            score = opt_function(particle.position)
            
            # Update personal best
            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = np.copy(particle.position)
            
            # Update global best
            if score > global_best_score:
                global_best_score = score
                global_best_position = np.copy(particle.position)

        # Update velocities and positions
        for particle in swarm:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            inertia = w * particle.velocity
            cognitive = c1 * r1 * (particle.best_position - particle.position)
            social = c2 * r2 * (global_best_position - particle.position)

            # Update velocity and position
            particle.velocity = inertia + cognitive + social
            particle.position += particle.velocity

            # Apply bounds
            particle.position = np.clip(particle.position, bounds[:, 0], bounds[:, 1])

        # Print progress
        print(f"Iteration {iteration+1}/{max_iter}, Best Score: {global_best_score:.4f}")
        print("Best Position:", global_best_position)
        #print(global_best_score)

    return global_best_position, global_best_score



def grid_optimization(opt_function, dim, bounds, num_samples=30, n_verbose=None):
    """ Minimization of a function using grid search strategy """
    # Create grid points for each dimension
    grid_axes = [np.linspace(low, high, num_samples) for low, high in bounds]
    grid_points = product(*grid_axes)  # Cartesian product of grid axes
    #print(list(grid_points))
    
    best_point = None
    best_value = -float("inf")  # Assuming minimization; use -inf for maximization
    
    max_iter = np.prod([num_samples] * dim)
    print(n_verbose)
    if n_verbose is None: n_verbose = max_iter+1
    
    # Evaluate function on grid
    for i, point in enumerate(grid_points):
        if i % n_verbose == 0:
            print(f"Iteration {i}/{max_iter}, Best Score: {best_value:.4f}")
        value = opt_function(np.array(point))
        #print(point, value)
        if value > best_value:  # Change to `>` for maximization
            best_value = value
            best_point = np.array(point)
    
    return best_point, best_value

    



# Example usage
if __name__ == "__main__":
    # Define a function to maximize (e.g., a simple sphere function with negative sign)
    def objective_function(x):
        return -np.sum(x**2)  # Maximize a negative quadratic function

    dim = 2  # Number of dimensions
    bounds = [(-10, 10), (-10, 10)]  # Search space bounds for each dimension

    best_position, best_score = pso(objective_function, dim, bounds)
    print("\nOptimal Solution Found by PSO:")
    print("Best Position:", best_position)
    print("Best Score:", best_score)
    
    best_position, best_score = grid_optimization(objective_function, dim, bounds, num_samples=5)
    print("\nOptimal Solution Found by grid search:")
    print("Best Position:", best_position)
    print("Best Score:", best_score)
    
