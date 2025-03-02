from morphologicalCBR.TextCase import TextCase
from morphologicalCBR.adaptation import VowelHarmonyAdaptation
from morphologicalCBR.retrieval import MorphologicalRetrieval, problem_distance

import numpy as np
from numericalCBR.adaptation import WeightedAdaptation
from numericalCBR.numerical_case import NumericalCase, NumericalCaseBase

from TACBR.known_adaptation.known_target_solution import DirectingRetrieval



# Testing CBR for morphology

CB = [TextCase("kyy", "kyyllä"), 
      TextCase("sota", "sodalla"), 
      TextCase("koira", "koiralla")]

ret = MorphologicalRetrieval({"distance": problem_distance})
cases = ret.retrieve("hyvä", CB, K=2)
ad = VowelHarmonyAdaptation({"knows_harmony": True})
print(ad.adapt(cases, "hyvä"))



# Testing retrieval with known target solution

def naive_text_distance(t1, t2):
    if t1 == t2: return 0
    return 1 + abs(len(t1) - len(t2))

VH_adaptation = VowelHarmonyAdaptation({"knows_harmony": False})


dir_ret_parameters = { "distance": naive_text_distance, 
                       "adaptation": VH_adaptation}

dir_ret = DirectingRetrieval(dir_ret_parameters)
print(dir_ret.retrieve(TextCase("kissa", "kissalla"), CB, 1))





# Testing numerical CBR

CB_num = NumericalCaseBase((2,), (1,))
CB_num.add_case(np.array([1,1]), np.array([1]))
CB_num.add_case(np.array([0,0]), np.array([0]))

cases = list(CB_num.get_all_cases())

weight = np.array([1.0, 10.0])

problem = np.array([0.0, 1.0])

w_adaptation = WeightedAdaptation({"weight": weight})
print(w_adaptation.adapt(cases, problem))