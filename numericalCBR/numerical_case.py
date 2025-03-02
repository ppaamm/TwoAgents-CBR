from CBR.CaseBase import Case, CaseBase
from typing import Set, Tuple
import numpy as np

class NumericalCase(Case):
    def __init__(self, problem: np.ndarray, solution: np.ndarray):
        assert isinstance(problem, np.ndarray), "'problem' must be a NumPy array"
        assert isinstance(solution, np.ndarray), "'solution' must be a NumPy array"
        super().__init__(problem, solution)
        
    def problem_shape(self):
        return self.problem.shape
    
    def solution_shape(self):
        return self.solution.shape
        
        
class NumericalCaseBase(CaseBase):
    def __init__(self, problem_shape: Tuple[int], solution_shape: Tuple[int]):
        super().__init__()
        self.problem_shape = problem_shape
        self.solution_shape = solution_shape

    def add_case(self, problem: np.ndarray, solution: np.ndarray) -> NumericalCase:
        # Ensure that the case added is of type NumericalCase
        case = NumericalCase(problem, solution)
        assert self.problem_shape == case.problem_shape(), "The shape of the problem is incorrect"
        assert self.solution_shape == case.solution_shape(), "The shape of the solution is incorrect"
        self.cases.add(case)

    def get_all_cases(self) -> Set[NumericalCase]:
        return super().get_all_cases()