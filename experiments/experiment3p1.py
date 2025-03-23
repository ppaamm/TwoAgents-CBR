#########################   EXPERIMENT 1.1   ##############################

from morphologicalCBR.TextCase import TextCase, TextCaseBase
from morphologicalCBR.adaptation import VowelHarmonyAdaptation

from TACBR.known_adaptation.known_target_solution import DirectingRetrieval
from TACBR.unknown_adaptation.retrieval import FiniteAdaptationProbability, UnknownFiniteAdaptationRetrieval
from TACBR.unknown_adaptation.known_target_solution import UnknownFiniteAdaptDirectingRetrieval



import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime

np.random.seed(42)

# PARAMETERS

n_run = 5    # Number of runs
#n_CB = 20   # Size of the case base
n_test = 50 # Size of the test set
n_pretrain = 1
K_max = 1    # Maximal number of cases to retrieve



# Step 1: Data generation

start_time = time.time()

print("Loading data")

df = pd.read_csv ("./data/FI/Inessive/ine.txt")
df = df[df.inessive != '—']
df = df[df.inessive != '–']

type48 = df[df.type == 48]

total_data = type48



def naive_text_distance(t1, t2):
    if t1 == t2: return 0
    #return 1 + abs(len(t1) - len(t2))
    return 1


def noisy_adaptation_likelihood(observations, adaptation, eps=1e-2):
    likelihoods = [
        (1 - eps) if adaptation.adapt(obs[2], obs[0]) == obs[1] else eps
        for obs in observations
    ]
    
    return np.prod(likelihoods)


def run_pipeline(data, K, vowel_harmony, n_CB):
    df_CB, df_test = train_test_split(data, train_size=n_CB, shuffle=True)
    CB = TextCaseBase.from_dataframe(df_CB)
    df_pretrain = df_test[:n_pretrain]
    df_test = df_test[n_pretrain : n_pretrain + n_test]
    
    probabilities = [(VowelHarmonyAdaptation({"knows_harmony": True}), 0.5), 
                     (VowelHarmonyAdaptation({"knows_harmony": False}), 0.5)]
    prior = FiniteAdaptationProbability(probabilities)
    
    retrieval = UnknownFiniteAdaptDirectingRetrieval({"adaptation_probability": prior, 
                                                      "loss": naive_text_distance})
    
    
    # Pre-training
    print("Pre-training")
    
    
    
    ad = VowelHarmonyAdaptation({"knows_harmony": vowel_harmony})
    dir_ret_parameters = { "loss": naive_text_distance, 
                           "adaptation": ad}
    dir_ret = DirectingRetrieval(dir_ret_parameters)
    
    print("... collecting data")
    
    observations = []
    
    for i, test_case in enumerate(df_pretrain.itertuples(index=False, name=None)):
        nominative, inessive, _ = test_case
        retrieved = dir_ret.retrieve(TextCase(nominative, inessive), CB, K)
        prediction = ad.adapt(retrieved, nominative)
        
        observations.append((nominative, prediction, retrieved))
    
    print("... computing posterior")
    retrieval.update_adaptation_probability(observations, noisy_adaptation_likelihood)
    #print(retrieval.adaptation_probability.adaptation_probabilities)


    
    # Comparing methods
    
    # Baseline: correct adaptation
    dir_ret_parameters = { "loss": naive_text_distance, 
                           "adaptation": VowelHarmonyAdaptation({"knows_harmony": vowel_harmony})}
    correct_dir_ret = DirectingRetrieval(dir_ret_parameters)
    
    # Baseline: incorrect adaptation
    dir_ret_parameters = { "loss": naive_text_distance, 
                           "adaptation": VowelHarmonyAdaptation({"knows_harmony": not(vowel_harmony)})}
    incorrect_dir_ret = DirectingRetrieval(dir_ret_parameters)
    


    errors = 0
    errors_correct = 0
    errors_incorrect = 0
    for i, test_case in enumerate(df_test.itertuples(index=False, name=None)):
        nominative, inessive, _ = test_case
        retrieved = retrieval.retrieve(TextCase(nominative, inessive), CB, K)
        retrieved_correct = correct_dir_ret.retrieve(TextCase(nominative, inessive), CB, K)
        retrieved_incorrect = incorrect_dir_ret.retrieve(TextCase(nominative, inessive), CB, K)
        
        prediction = ad.adapt(retrieved, nominative)
        if prediction != inessive: errors += 1
        
        prediction_correct = ad.adapt(retrieved_correct, nominative)
        if prediction_correct != inessive: errors_correct += 1
        
        prediction_incorrect = ad.adapt(retrieved_incorrect, nominative)
        if prediction_incorrect != inessive: errors_incorrect += 1
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(df_test)} test cases, Errors: {errors}")
            
    harmony_proba = retrieval.adaptation_probability.adaptation_probabilities[0][1]
    return errors / n_test, errors_correct / n_test, errors_incorrect / n_test, harmony_proba



# Step 3: Launching all the runs

results = []


for K in range(K_max):
    for harmony in (False, True):
        for n_CB in (20, 30):
            print(f"K = {K+1}, Harmony = {harmony}, CB size = {n_CB}")
            scores = []
            scores_correct = []
            scores_incorrect = []
            harmony_probas = []
            for n in range(n_run):
                score, score_correct, score_incorrect, harmony_proba = run_pipeline(total_data, 
                                                                                    K+1, 
                                                                                    harmony, 
                                                                                    n_CB)
                scores.append(score)
                scores_correct.append(score_correct)
                scores_incorrect.append(score_incorrect)
                harmony_probas.append(harmony_proba)
                
            results.append({
                "K": K + 1,
                "harmony": harmony,
                "n_CB": n_CB,
                "scores": scores,
                "scores_correct": scores_correct,
                "scores_incorrect": scores_incorrect,
                "harmony_probas": harmony_probas
            })
    
# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV for later use
df_results.to_csv("results/results-3-1.csv", index=False)


def compute_stats(series):
    all_values = np.concatenate(series.values)  # Flatten the list of lists
    return np.mean(all_values), np.std(all_values)

# Assuming df is your DataFrame
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

for i, harmony_value in enumerate([False, True]):
    ax = axes[i]
    df_subset = df_results[df_results["harmony"] == harmony_value]
    
    # Compute mean and std for each n_CB
    stats = df_subset.groupby("n_CB").agg({
        "scores": lambda x: compute_stats(x),
        "scores_correct": lambda x: compute_stats(x),
        "scores_incorrect": lambda x: compute_stats(x)
    }).reset_index()

    # Extract means and stds separately
    n_CB_values = stats["n_CB"]
    means = stats[["scores", "scores_correct", "scores_incorrect"]].applymap(lambda x: x[0])
    stds = stats[["scores", "scores_correct", "scores_incorrect"]].applymap(lambda x: x[1])

    # Rename columns for plotting
    means.columns = ["Scores", "Correct Scores", "Incorrect Scores"]
    stds.columns = ["Scores", "Correct Scores", "Incorrect Scores"]

    # Plot bars with error bars
    means.plot(kind="bar", yerr=stds, ax=ax, capsize=4, width=0.8, alpha=0.75)
    
    ax.set_xticks(range(len(n_CB_values)))
    ax.set_xticklabels(n_CB_values, rotation=0)
    ax.set_title(f"Harmony = {harmony_value}")
    ax.set_xlabel("n_CB")
    ax.set_ylabel("Score")
    ax.legend(["Scores", "Correct Scores", "Incorrect Scores"])

plt.savefig("results/exp3-1.png", dpi=300, bbox_inches="tight")
