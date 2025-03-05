from CBR.CaseBase import Case, CaseBase
from CBR.containers import Retrieval, Adaptation
from typing import List, Any, Dict, Callable
from abc import ABC, abstractmethod


class LearnableParametricRetrieval(Retrieval,ABC):
    def __init__(self, parameters: Dict[str, Any]):
        assert "adaptation" in parameters, "'adaptation' key must be present in parameters"
        assert isinstance(parameters["adaptation"], Adaptation), "'adaptation' must be of type Adaptation"
        assert "retrieval" in parameters, "'retrieval' key must be present in parameters"
        assert isinstance(parameters["retrieval"], Retrieval), "'retrieval' must be of type Retrieval"
        super().__init__(parameters)
        
        self.retrieval = parameters["retrieval"]
        self.adaptation = parameters["adaptation"]
        self.optimal_parameter = None # Because not trained
        
    
    @abstractmethod
    def fit(self, CB: CaseBase, CB_test: CaseBase, loss: Callable[[Any, Any], float]):
        assert isinstance(CB, CaseBase), "'CB' must be of type CaseBase"
        assert isinstance(CB_test, CaseBase), "'CB_test' must be of type CaseBase"
        pass
        
        
        
    def retrieve(self, target_problem: Any, CB: CaseBase, K = 1) -> List[Case]:
        #assert self.optimal_parameter != None, "The retrieval method must be trained before use"
        if self.optimal_parameter is None: 
            raise ValueError("The retrieval method must be trained before use. Please call fit() before retrieve().")
        return self.retrieval.retrieve(target_problem, CB, K=K, ret_parameter=self.optimal_parameter)
            
    

