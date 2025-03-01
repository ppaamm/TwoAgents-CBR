from morphologicalCBR.TextCase import TextCase
from morphologicalCBR.adaptation import VowelHarmonyAdaptation
from morphologicalCBR.retrieval import MorphologicalRetrieval, problem_distance

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

VH_adaptation = VowelHarmonyAdaptation({"knows_harmony": True})


dir_ret_parameters = { "distance": naive_text_distance, 
                       "adaptation": VH_adaptation}

dir_ret = DirectingRetrieval(dir_ret_parameters)
print(dir_ret.retrieve(TextCase("hyvä", "hyvällä"), CB, 1))

