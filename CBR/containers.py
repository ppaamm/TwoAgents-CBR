from .CaseBase import CaseBase, Case
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional


class Retrieval(ABC):
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    @abstractmethod
    def retrieve(self, target_problem: Any, CB: CaseBase, K: Optional[int] = None) -> List[Case]:
        pass
    
    def retrieve_multiple(self, target_problems: List[Any], CB: CaseBase, K: Optional[int] = None) -> List[List[Case]]:
        return [self.retrieve(target_problem, CB, K) for target_problem in target_problems]
    
    
class Adaptation(ABC):
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    @abstractmethod
    def adapt(self, cases: List[Case], problem: Any) -> Any:
        pass
    
    def adapt_multiple(self, cases: List[List[Case]], problems: List[Any]) -> List[Any]:
        assert len(cases) == len(problems), "There must be the same number of cases and solutions"
        return [self.adapt(cases[i], problems[i]) for i in range(len(problems))]