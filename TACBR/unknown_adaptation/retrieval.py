from CBR.containers import Adaptation, Retrieval
from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod


class UnknownAdaptationRetrieval(Retrieval, ABC):
    
    def __init__(self, parameters: Dict[str, Any]):
        assert "adaptation_probability" in parameters, "'adaptation_probability' key must be present in parameters"
        self.adaptation_probability = parameters['adaptation_probability']
        super().__init__(parameters)
        
    @abstractmethod
    def update_adaptation_probability(self, observations: List[Tuple[Any,Any,List[Any]]]):
        """
        Updates the adaptation probability (inference)

        Parameters
        ----------
        observations : List[Tuple[Any,Any,List[Any]]]
            Observations of the form (problem, solution, retrieved_cases).

        """
        pass
        


class FiniteAdaptationProbability:
    def __init__(self, adaptation_probabilities: List[Tuple[Adaptation, float]]):
        assert sum([proba for _, proba in adaptation_probabilities]) == 1, "Probabilities must sum to 1"
        self.adaptation_probabilities = adaptation_probabilities


class ProbabilitisticSampledAdaptations:
    def __init__(self, sampled_adaptations: List[Adaptation]):
        is_list_ad = isinstance(sampled_adaptations, list) and all(isinstance(item, Adaptation) for item in sampled_adaptations)
        assert not(is_list_ad), "x must be a list of Adaptation"
        self.sampled_adaptations = sampled_adaptations


class UnknownFiniteAdaptationRetrieval(UnknownAdaptationRetrieval, ABC):
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        assert isinstance(parameters["adaptation_probability"], FiniteAdaptationProbability)
        

class UnknownSampledAdaptationRetrieval(UnknownAdaptationRetrieval, ABC):
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        assert isinstance(parameters["adaptation_probability"], ProbabilitisticSampledAdaptations)