from CBR.CaseBase import Case, CaseBase
from CBR.containers import Retrieval, Adaptation
from typing import List, Any, Dict
import itertools


class DirectingRetrieval(Retrieval):
    def __init__(self, parameters: Dict[str, Any]):
        assert "adaptation" in parameters, "'adaptation' key must be present in parameters"
        assert isinstance(parameters["adaptation"], Adaptation), "'adaptation' must be of type Adaptation"
        assert "distance" in parameters, "'distance' key must be present in parameters"
        assert callable(parameters["distance"]), "'distance' must be a callable function"
        assert len(parameters["distance"].__code__.co_varnames) == 2, "'distance' function must accept exactly two arguments"
        super().__init__(parameters)


    def retrieve(self, target_problem: Any, CB: CaseBase, K) -> List[Case]:
        distance = self.parameters["distance"]
        adaptation = self.parameters["adaptation"]
        dist_ad = lambda C: distance(target_problem.solution, 
                                     adaptation.adapt(C, target_problem.problem))
        cases = min(itertools.combinations(CB, K), key=dist_ad)
        return cases
            