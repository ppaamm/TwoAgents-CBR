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
    
    def adapt_multiple(self, cases: List[Case], problems: List[Any]) -> List[Any]:
        return [self.adapt(cases, problem) for problem in problems]