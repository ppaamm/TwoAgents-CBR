from typing import Set, Any, List
import uuid

class Case:
    def __init__(self, problem: Any, solution: Any):
        self.id = uuid.uuid4()
        self.problem = problem
        self.solution = solution

    def __repr__(self):
        return f"Case(id={self.id}, problem={self.problem}, solution={self.solution})"

    def __eq__(self, other):
        if isinstance(other, Case):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)


class CaseBase:
    def __init__(self):
        self.cases: Set[Case] = set()
        
    def __len__(self):
        return len(self.cases)

    def add_case(self, problem: Any, solution: Any) -> Case:
        case = Case(problem, solution)
        self.cases.add(case)

    def remove_case(self, case_id: uuid.UUID):
        self.cases = {case for case in self.cases if case.id != case_id}

    def get_all_cases(self) -> Set[Case]:
        return self.cases
    
    def get_all_cases_as_list(self) -> List[Case]:
        return list(self.cases)