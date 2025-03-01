from . analogy.analogy import solveAnalogy
from . TextCase import TextCase
from CBR.containers import Adaptation
from typing import List, Any
from collections import Counter


class VowelHarmonyAdaptation(Adaptation):
    def __init__(self, parameters):
        assert "knows_harmony" in parameters, "Parameter dictionary must contain the key 'knows_harmony'"
        super().__init__(parameters)
    
    def one_case_adaptation(source_case: TextCase, target_question:str, harmony:bool):
        target_solution = solveAnalogy(source_case.problem, 
                                       source_case.solution, 
                                       target_question)
        # TODO: Here we select the first element randomly... Better solution?
        target_solution = (target_solution[0][0][0], target_solution[1])
        return (VowelHarmonyAdaptation.apply_harmony(target_question, target_solution[0], harmony), 
                target_solution[1])
    
    def apply_harmony(C, D, harmony):
        if harmony:
            if "a" in C or "o" in C or "u" in C:
                D = D.replace("ä", "a")
                D = D.replace("ö", "o")
                D = D.replace("y", "u")
            else:
                D = D.replace("a", "ä")
                D = D.replace("o", "ö")
                D = D.replace("u", "y")
        return D 
    
    
    def adapt(self, cases: List[TextCase], problem: Any) -> str:
        solutions = [VowelHarmonyAdaptation.one_case_adaptation(source_case, 
                                                                problem, 
                                                                self.parameters['knows_harmony'])
                     for source_case in cases]
        
        most_common_solutions = Counter(solutions).most_common()
        #print("Most common:", most_common_solutions)
        
        top_solutions = [solution for solution, count in most_common_solutions 
                         if count == most_common_solutions[0][1]]
        
        min_length_solutions = [solution for solution, length in top_solutions 
                                if length == min(length for _, length in top_solutions)]
        
        # TODO: Here too we select the first element randomly... Better solution?
        return min_length_solutions[0]



   