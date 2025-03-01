from CBR.CaseBase import Case, CaseBase
from typing import Set

class TextCase(Case):
    def __init__(self, problem: str, solution: str):
        super().__init__(problem, solution)
        
        
class TextCaseBase(CaseBase):
    def __init__(self):
        super().__init__()

    def add_case(self, problem: str, solution: str) -> TextCase:
        # Ensure that the case added is of type TextCase
        case = TextCase(problem, solution)
        self.cases.add(case)

    def get_all_cases(self) -> Set[TextCase]:
        return super().get_all_cases()