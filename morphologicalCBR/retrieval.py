from . analogy.analogy import solveAnalogy
from . analogy import complexity
from . TextCase import TextCase, TextCaseBase
from CBR.containers import Retrieval
from typing import List, Any, Dict, Optional
import sys


class MorphologicalRetrieval(Retrieval):
    def __init__(self, parameters: Dict[str, Any]):
        assert "distance" in parameters, "Parameter dictionary must contain the key 'distance'"
        super().__init__(parameters)

    def retrieve(self, target_problem: str, CB: TextCaseBase, K: Optional[int] = None) -> List[TextCase]:
        distances = [[case, self.parameters["distance"](case, target_problem)] for case in CB]
        distances.sort(key=lambda x: x[1])
        return [x[0] for x in distances[:K]]





### Important distances based on K complexity    


def analogical_distance(case: TextCase, target_problem: str):
    """
    d1(A:B,C) = min_D K(A:B::C:D)
    """
    _, dist = solveAnalogy(case.problem, case.solution, target_problem)
    return dist
    


def normalized_analogical_distance(case: TextCase, target_problem: str):
    """
    d2(A:B,C) = min_D K(A:B::C:D) - K(A:B)
    """
    _, dist = solveAnalogy(case.problem, case.solution, target_problem)
    min_length = complexity.getK_AB(case.problem, case.solution)
    return dist - min_length


def problem_distance(case: TextCase, target_problem: str):
    """
    d3(A:B,C) = K(A::C)
    """
    return complexity.getK_AC(case.problem, target_problem)


def normalized_problem_distance(case: TextCase, target_problem: str):
    """
    d4(A:B,C) = K(A::C) - K(A)
    """
    return problem_distance(case, target_problem) - complexity.getK_A(case.problem)
