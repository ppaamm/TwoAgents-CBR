from CBR.CaseBase import Case

class TextCase(Case):
    def __init__(self, problem: str, solution: str):
        super().__init__(problem, solution)