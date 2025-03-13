#########################   EXPERIMENT 1.1   ##############################

from morphologicalCBR.TextCase import TextCase, TextCaseBase
from morphologicalCBR.adaptation import VowelHarmonyAdaptation
from morphologicalCBR.retrieval import MorphologicalRetrieval, problem_distance

from TACBR.known_adaptation.known_target_solution import DirectingRetrieval


import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import seaborn as sns
import time
import matplotlib.pyplot as plt
import datetime

np.random.seed(42)

# PARAMETERS

n_run = 10    # Number of runs
n_CB = 20   # Size of the case base
n_test = 50 # Size of the test set
K_max = 5    # Maximal number of cases to retrieve



# Step 1: Data generation

start_time = time.time()

print("Loading data")

df = pd.read_csv ("./data/FI/Inessive/ine.txt")
df = df[df.inessive != '—']
df = df[df.inessive != '–']

#type2 = df[df.type == 2]
#type11 = df[df.type == 11]
#type38 = df[df.type == 38]
#type41 = df[df.type == 41]
type48 = df[df.type == 48]

total_data = type48



def naive_text_distance(t1, t2):
    if t1 == t2: return 0
    #return 1 + abs(len(t1) - len(t2))
    return 1



def run_pipeline(data, K, vowel_harmony):
    df_CB, df_test = train_test_split(data, train_size=n_CB, shuffle=True)
    CB = TextCaseBase.from_dataframe(df_CB)
    df_test = df_test[:n_test]
    
    
    ad = VowelHarmonyAdaptation({"knows_harmony": vowel_harmony})
    dir_ret_parameters = { "loss": naive_text_distance, 
                           "adaptation": ad}
    dir_ret = DirectingRetrieval(dir_ret_parameters)
    
    errors = 0
    for i, test_case in enumerate(df_test.itertuples(index=False, name=None)):
        nominative, inessive, _ = test_case
        retrieved = dir_ret.retrieve(TextCase(nominative, inessive), CB, K)
        
        prediction = ad.adapt(retrieved, nominative)
        if prediction != inessive: errors += 1
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(df_test)} test cases, Errors: {errors}")
    return errors / n_test



# Step 3: Launching all the runs

results = []


for K in range(K_max):
    for harmony in (False, True):
        print(f"K = {K}, Harmony = {harmony}")
        scores = []
        for n in range(n_run):
            score = run_pipeline(total_data, K+1, harmony)
            scores.append(score)
            
        results.append({
            "K": K + 1,
            "harmony": harmony,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores)
        })
    
# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV for later use
df_results.to_csv("results.csv", index=False)


plt.figure(figsize=(8, 5))

for harmony_value in [False, True]:
    subset = df_results[df_results["harmony"] == harmony_value]
    
    # Plot mean scores
    plt.plot(subset["K"], subset["mean_score"], marker="o", label=f"Harmony: {harmony_value}")
    
    # Add shaded variance region
    plt.fill_between(subset["K"], 
                     subset["mean_score"] - subset["std_score"], 
                     subset["mean_score"] + subset["std_score"], 
                     alpha=0.2)

# Labels and legend
plt.xlabel("K")
plt.ylabel("Mean Score")
plt.title("Prediction accuracy")
plt.legend(title="Harmony")
plt.grid()
plt.show()















