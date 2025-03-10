from CBR.CaseBase import Case, CaseBase
from CBR.containers import Retrieval, Adaptation
from . retrieval import UnknownFiniteAdaptationRetrieval
from typing import List, Any, Dict
import itertools


class UnknownFiniteAdaptDirectingRetrieval(UnknownFiniteAdaptationRetrieval):
    def __init__(self, parameters: Dict[str, Any]):
        assert "loss" in parameters, "'loss' key must be present in parameters"
        assert callable(parameters["loss"]), "'loss' must be a callable function"
        assert len(parameters["loss"].__code__.co_varnames) == 2, "'loss' function must accept exactly two arguments"
        super().__init__(parameters)


    def retrieve(self, target_problem: Any, CB: CaseBase, K) -> List[Case]:
        loss = self.parameters["loss"]
        loss_ad = lambda C, adaptation: loss(target_problem.solution, adaptation.adapt(C, target_problem.problem))
        expected_loss_ad = lambda C: sum([p * loss_ad(C, ad) for ad, p in self.adaptation_probability])
        
        cases = min(itertools.combinations(CB, K), key=expected_loss_ad)
        return cases
    