from . numerical_case import NumericalCase, NumericalCaseBase
from CBR.containers import Adaptation, Retrieval
from TACBR.known_adaptation.unknown_target_solution import LearnableParametricRetrieval
from TACBR.known_adaptation.utils import pso
from typing import List, Any, Dict, Callable
import numpy as np
    
    
    
class WeightedDistanceRetrieval(Retrieval):
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    
    def retrieve(self, target_problem: np.ndarray, CB: NumericalCaseBase, 
                 K: int, ret_parameter: np.ndarray) -> List[NumericalCase]:
        
        CB_list = CB.get_all_cases_as_list()
        
        distances = [np.sqrt(np.dot((case.problem-target_problem) * ret_parameter, case.problem-target_problem))
                     for case in CB_list]
        return [CB_list[i] for i in np.argsort(distances)[:K]]
    
    
    
    
# Should probably not be located here! 
class LearnableWeightedDistanceRetrieval(LearnableParametricRetrieval):
    
    def __init__(self, parameters: Dict[str, Any]):
        assert "retrieval" in parameters, "'retrieval' key must be present in parameters"
        assert isinstance(parameters["retrieval"], WeightedDistanceRetrieval), "'retrieval' must be of type WeightedDistanceRetrieval"
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
            total_loss = 0
            for case in CB_test.get_all_cases():
                retrieved_cases = self.parameters['retrieval'].retrieve(case.problem, CB, fit_params['K'], w)
                
                total_loss += loss(self.adaptation.adapt(retrieved_cases, case.problem), case.solution)
            return - total_loss
        
        args = fit_params["pso_params"] if "pso_params" in fit_params else {}
        best_position, best_score = pso(objective_function, dim, bounds, **args)
        self.optimal_parameter = best_position