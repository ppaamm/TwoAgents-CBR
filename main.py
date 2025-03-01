from CBR.CaseBase import Case, CaseBase
from CBR import containers
from morphologicalCBR.TextCase import TextCase

from morphologicalCBR.adaptation import VowelHarmonyAdaptation

cases = [TextCase("kyy", "kyyllä"), TextCase("sota", "sodalla")]
ad = VowelHarmonyAdaptation({"knows_harmony": True})

print(ad.adapt(cases, "hyvä"))