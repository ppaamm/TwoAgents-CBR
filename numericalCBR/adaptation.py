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
    
    
    
    def adapt(self, cases: List[NumericalCase], problem: np.ndarray) -> str:
        weight = self.parameters['weight']
        delta_problem = [np.abs(case.problem - problem) for case in cases]
        distances = np.array([weight @ delta for delta in delta_problem])
        barycenter_coefficients = np.exp(- distances)
        
        weighted_sum = sum(barycenter_coefficients[i] * cases[i].solution for i in range(len(barycenter_coefficients)))
        return weighted_sum / np.sum(barycenter_coefficients)