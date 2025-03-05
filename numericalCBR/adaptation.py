from . numerical_case import NumericalCase
from CBR.containers import Adaptation
from typing import List, Any
from collections import Counter
import numpy as np




class WeightedAdaptation(Adaptation):
    def __init__(self, parameters):
        assert "weight" in parameters, "Parameter dictionary must contain the key 'weight'"
        assert np.all(parameters["weight"] >= 0), "All weights must be non-negative"
        super().__init__(parameters)
    
    
    
    def adapt(self, cases: List[NumericalCase], problem: np.ndarray) -> np.ndarray:
        weight = self.parameters['weight']
        barycenter_coefficients = [np.exp(- weight @ np.abs(case.problem - problem)) for case in cases]
        
        weighted_sum = sum(barycenter_coefficients[i] * cases[i].solution for i in range(len(barycenter_coefficients)))
        return weighted_sum / np.sum(barycenter_coefficients)
    
    
    def adapt_multiple(self, cases: List[List[NumericalCase]], problems: List[Any]) -> List[np.ndarray]:
        weight = self.parameters['weight']

        solutions = [[case.solution for case in group] for group in cases]
        problems = np.array(problems)  # (M, d)
    
        adapted_solutions = []
        
        for i, (problem, case_group) in enumerate(zip(problems, cases)):
            case_problems = np.array([case.problem for case in case_group])  # (N, d)
            barycenter_coefficients = np.exp(-np.sum(weight * np.abs(case_problems - problem), axis=1))  # (N,)
            weighted_solutions = np.sum(np.array(solutions[i]) * barycenter_coefficients[:, None], axis=0)
            adapted_solutions.append(weighted_solutions / np.sum(barycenter_coefficients))
    
        return np.array(adapted_solutions)