from . numerical_case import NumericalCase, NumericalCaseBase
from CBR.containers import Adaptation, Retrieval
from TACBR.known_adaptation.unknown_target_solution import LearnableParametricRetrieval
from TACBR.known_adaptation.utils import pso, grid_optimization
from typing import List, Any, Dict, Callable
import numpy as np
    
    
    
class WeightedDistanceRetrieval(Retrieval):
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    
    def retrieve(self, target_problem: np.ndarray, CB: NumericalCaseBase, 
                 K: int, ret_parameter: np.ndarray) -> List[NumericalCase]:
        
        CB_list = CB.get_all_cases_as_list()
        
        # squared_distances = [np.dot((case.problem-target_problem) * ret_parameter, case.problem-target_problem)
        #              for case in CB_list]
        # return [CB_list[i] for i in np.argsort(squared_distances)[:K]]
        problems = np.array([case.problem for case in CB_list])  # Shape: (N, d)
    
        diffs = problems - target_problem  # Shape: (N, d)
        weighted_diffs = diffs * ret_parameter  # Apply weights element-wise
        squared_distances = np.sum(weighted_diffs * diffs, axis=1)  # Compute squared distances
        
        # Get indices of K smallest distances
        closest_indices = np.argpartition(squared_distances, K)[:K] 
        
        # Return the K closest cases
        return [CB_list[i] for i in closest_indices]
    
    
    def retrieve_multiple(self, target_problems: List[np.ndarray], CB: NumericalCaseBase, 
                 K: int, ret_parameter: np.ndarray) -> List[NumericalCase]:
        
        CB_list = CB.get_all_cases_as_list()
        
        problems = np.array([case.problem for case in CB_list])  # (N, d)
    
        # Convert target_problems into a NumPy array of shape (M, d)
        target_problems = np.array(target_problems)  # (M, d)
    
        weighted_problems = problems * ret_parameter  # (N, d)
        weighted_targets = target_problems * ret_parameter  # (M, d)
        diffs = weighted_problems[None, :, :] - weighted_targets[:, None, :]  # Shape: (M, N, d)
        squared_distances = np.sum(diffs ** 2, axis=2)  # Sum across dimensions -> (M, N)
    
        # Get indices of K smallest distances for each target problem
        closest_indices = np.argpartition(squared_distances, K, axis=1)[:, :K]  # (M, K)
    
        # Retrieve the K closest cases for each target problem
        return [[CB_list[i] for i in row] for row in closest_indices]
    
    
    
# Should probably not be located here! 
class LearnableWeightedDistanceRetrieval(LearnableParametricRetrieval):
    
    def __init__(self, parameters: Dict[str, Any]):
        assert "retrieval" in parameters, "'retrieval' key must be present in parameters"
        assert isinstance(parameters["retrieval"], WeightedDistanceRetrieval), "'retrieval' must be of type WeightedDistanceRetrieval"
        if not("optimization_method" in parameters):
            parameters["optimization_method"] = "grid"
            print("No optimization method specified => using default (grid)")
        super().__init__(parameters)
    
    def fit(self, CB: NumericalCase, CB_test: NumericalCase, 
            loss: Callable[[Any, Any], float], fit_params: Dict[str, Any]):
        
        assert "bounds" in fit_params, "'bounds' key must be present in fit_params"
        assert "K" in fit_params, "'K' key must be present in fit_params"
        assert isinstance(fit_params["K"], int), "'K' must be an integer"
        assert fit_params["K"] > 0, "'K' must be positive"
        
        dim = CB.problem_shape[0] # TODO: Works only for vectors, to adapt if we have matrices
        bounds = [fit_params['bounds']] * dim 
        
        def objective_function(w):
            CB_test_list = list(CB_test.get_all_cases())
            
            total_loss = 0
            problems = [case.problem for case in CB_test_list]
            retrieved_cases = self.parameters['retrieval'].retrieve_multiple(problems, CB, fit_params['K'], w)
            
            for i in range(len(CB_test_list)):
                case = CB_test_list[i]
                total_loss += loss(self.adaptation.adapt(retrieved_cases[i], case.problem), case.solution)
            return - total_loss[0]
        
        args = fit_params["optimization_params"] if "optimization_params" in fit_params else {}
        if self.parameters["optimization_method"] == "pso":
            best_position, best_score = pso(objective_function, dim, bounds, **args)
        elif self.parameters["optimization_method"] == "grid":
            best_position, best_score = grid_optimization(objective_function, dim, bounds, **args)
        self.optimal_parameter = best_position