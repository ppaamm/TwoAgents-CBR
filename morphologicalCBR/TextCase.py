from CBR.CaseBase import Case, CaseBase
from typing import Set
import pandas as pd

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
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TextCaseBase":
        """Creates a TextCaseBase from a DataFrame where the first column is the problem and the second is the solution."""
        assert df.shape[1] >= 2, "DataFrame must have at least two columns (problem, solution)."
        
        case_base = cls()
        for _, row in df.iterrows():
            case_base.add_case(row.iloc[0], row.iloc[1])  # First column = problem, second column = solution
        
        return case_base
    