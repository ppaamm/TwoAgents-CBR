from .CaseBase import CaseBase, Case
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional


class Retrieval(ABC):
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    @abstractmethod
    def retrieve(self, CB: CaseBase, K: Optional[int] = None) -> List[Case]:
        pass
    
    
class Adaptation(ABC):
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    @abstractmethod
    def adapt(self, cases: List[Case], problem: Any) -> Any:
        pass