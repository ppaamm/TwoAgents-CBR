from CBR.CaseBase import Case, CaseBase
from CBR import containers
from morphologicalCBR.TextCase import TextCase, TextCaseBase

from morphologicalCBR.adaptation import VowelHarmonyAdaptation
from morphologicalCBR.retrieval import MorphologicalRetrieval, problem_distance


CB = [TextCase("kyy", "kyyllä"), 
      TextCase("sota", "sodalla"), 
      TextCase("koira", "koiralla")]


ret = MorphologicalRetrieval({"distance": problem_distance})
cases = ret.retrieve("hyvä", CB, K=2)

ad = VowelHarmonyAdaptation({"knows_harmony": True})

print(ad.adapt(cases, "hyvä"))
