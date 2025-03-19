import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from numericalCBR.adaptation import WeightedAdaptation
from numericalCBR.numerical_case import NumericalCase, NumericalCaseBase
from numericalCBR.retrieval import WeightedDistanceRetrieval, LearnableWeightedDistanceRetrieval

from TACBR.known_adaptation.known_target_solution import DirectingRetrieval
from TACBR.unknown_adaptation.retrieval import FiniteAdaptationProbability, UnknownFiniteAdaptationRetrieval
from TACBR.unknown_adaptation.known_target_solution import UnknownFiniteAdaptDirectingRetrieval



np.random.seed(42)

def quadratic_distance(x1, x2):
    return (x1 - x2)**2




# PARAMETERS

n_run = 20    # Number of runs
#n_CB = 20   # Size of the case base
n_test = 100 # Size of the test set
K_max = 4    # Maximal number of cases to retrieve





# Step 1: Data generation

# Load California housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target  # Target variable (house price in $100,000s)

# Define features (X) and target (y)
X = df.drop(columns=["Price"])
y = df["Price"]

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.to_numpy()



def run_pipeline(X, y, K, n_CB):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_CB, shuffle=True)
    
    d = X_train.shape[1]
    
    # Defining the adaptation method
    #weight = np.array([1., 0, 0, 0, 0, 0, 0, 0])
    weight = np.random.rand(d)
    w_adaptation = WeightedAdaptation({"weight": weight})
    
    
    # Defining the retrieval strategy
    dir_ret_parameters = { "loss": quadratic_distance, 
                           "adaptation": w_adaptation }
    dir_ret = DirectingRetrieval(dir_ret_parameters)
    
    
    # Creating the case base:
    
    CB = NumericalCaseBase.from_numpy(X_train, y_train)
    CB_list = CB.get_all_cases_as_list()
    
    errors = []
    for i in range(n_test):
        x_tgt = X_test[i,:]
        y_tgt = np.array([y_test[i]])
        retrieved = dir_ret.retrieve(NumericalCase(x_tgt, y_tgt), CB_list, K)
        
        prediction = w_adaptation.adapt(retrieved, x_tgt)
        errors.append(np.abs(prediction - y_tgt))
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_test} test cases, Errors: {np.mean(errors)}")
    return np.mean(errors)




# Step 3: Launching all the runs

results = []


for K in range(K_max):
    for n_CB in (10, 20, 50):
        print(f"K = {K+1}, CB size = {n_CB}")
        for n in range(n_run):
            score = run_pipeline(X, y, K+1, n_CB)
            #scores.append(score)
            
            results.append({
                "K": K + 1,
                "n_CB": n_CB,
                "score": score
            })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV for later use
df_results.to_csv("results-1-2.csv", index=False)



# Assume df is your dataframe
df_grouped = df_results.groupby(["K", "n_CB"])["score"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(8, 6))

# Iterate over unique values of n_CB to plot each as a separate curve
for n_CB in df_grouped["n_CB"].unique():
    subset = df_grouped[df_grouped["n_CB"] == n_CB]
    
    plt.plot(subset["K"], subset["mean"], marker="o", label=f"{n_CB}")
    plt.fill_between(subset["K"], subset["mean"] - subset["std"], subset["mean"] + subset["std"], alpha=0.2)

plt.xlabel("K")
plt.ylabel("Prediction error")
plt.legend(title="Size of the CB:")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.xticks(df_grouped["K"].unique())  # Ensure only integer K values are shown
plt.title("Prediction error")

#plt.show()
plt.savefig("exp1-2-plot.png", dpi=300, bbox_inches="tight")




# Get unique values for plotting
Ks = sorted(df_grouped["K"].unique())
n_CBs = sorted(df_grouped["n_CB"].unique())

bar_width = 0.2  # Adjust bar width
x = np.arange(len(Ks))  # X positions for bars

plt.figure(figsize=(10, 6))

# Plot bars for each n_CB
for i, n_CB in enumerate(n_CBs):
    subset = df_grouped[df_grouped["n_CB"] == n_CB]
    plt.bar(x + i * bar_width, subset["mean"], width=bar_width, label=f"{n_CB}", yerr=subset["std"], capsize=5)

plt.xticks(x + (len(n_CBs) - 1) * bar_width / 2, Ks)  # Center labels under groups
plt.xlabel("K")
plt.ylabel("Prediction error")
plt.legend(title="Size of the CB:")
plt.title("Prediction error")
plt.grid(axis="y", linestyle="--", alpha=0.7)

#plt.show()
plt.savefig("exp1-2-bar.png", dpi=300, bbox_inches="tight")