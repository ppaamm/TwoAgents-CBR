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
        if np.isscalar(solution): solution = np.array([solution])
        if np.isscalar(problem): problem = np.array([problem])
        case = NumericalCase(problem, solution)
        
        # Ensure that the case added is of type NumericalCase
        assert self.problem_shape == case.problem_shape(), "The shape of the problem is incorrect"
        assert self.solution_shape == case.solution_shape(), "The shape of the solution is incorrect"
        
        self.cases.add(case)

    def get_all_cases(self) -> Set[NumericalCase]:
        return super().get_all_cases()
    
    
    @classmethod
    def from_numpy(cls, X: np.ndarray, y: np.ndarray) -> "NumericalCaseBase":
        assert isinstance(X, np.ndarray), "'X' must be a NumPy array"
        assert isinstance(y, np.ndarray), "'y' must be a NumPy array"
        assert len(X) == len(y), "X and y must have the same number of samples"
        
        # Extract the shape of individual problems and solutions
        problem_shape = X.shape[1:]  # Ignore batch dimension
        solution_shape = y.shape[1:] if y.ndim > 1 else (1,)  # Ensure it's a tuple

        # Initialize the case base
        CB = cls(problem_shape, solution_shape)

        # Populate with cases
        for i in range(len(X)):
            CB.add_case(X[i], y[i])

        return CB
        