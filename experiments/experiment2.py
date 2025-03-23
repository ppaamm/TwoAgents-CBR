import numpy as np
import pandas as pd
import seaborn as sns
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

from sklearn.linear_model import LinearRegression

np.random.seed(42)

def quadratic_distance(x1, x2):
    return (x1 - x2)**2




# PARAMETERS

n_run = 500    # Number of runs
#n_CB = 20   # Size of the case base
n_test = 50 # Size of the test set
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
    weight_adaptation = np.random.rand(d)
    incorrect_weight_adaptation = np.random.rand(d)
    w_adaptation = WeightedAdaptation({"weight": incorrect_weight_adaptation})
    
    weight_distance = np.sqrt(np.sum((weight_adaptation - incorrect_weight_adaptation)**2))
    
    
    # Defining the retrieval strategy
    dir_ret_parameters = { "loss": quadratic_distance, 
                           "adaptation": w_adaptation }
    dir_ret = DirectingRetrieval(dir_ret_parameters)
    
    
    # Creating the case base:
    
    CB = NumericalCaseBase.from_numpy(X_train, y_train)
    CB_list = CB.get_all_cases_as_list()
    
    correct_adaptation = WeightedAdaptation({"weight": weight_adaptation})
    
    errors = []
    for i in range(n_test):
        x_tgt = X_test[i,:]
        y_tgt = np.array([y_test[i]])
        retrieved = dir_ret.retrieve(NumericalCase(x_tgt, y_tgt), CB_list, K)
        
        prediction = correct_adaptation.adapt(retrieved, x_tgt)
        errors.append(np.abs(prediction - y_tgt))
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_test} test cases, Errors: {np.mean(errors)}")
    return weight_distance, np.mean(errors)




# Step 3: Launching all the runs

results = []


for K in range(K_max):
    for n_CB in (10, 20, 30):
        print(f"K = {K+1}, CB size = {n_CB}")
        for n in range(n_run):
            weight_distance, score = run_pipeline(X, y, K+1, n_CB)
            #scores.append(score)
            
            results.append({
                "K": K + 1,
                "n_CB": n_CB,
                "weight_distance": weight_distance,
                "score": score
            })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV for later use
df_results.to_csv("results/results-2.csv", index=False)


# Unique values of K and n_CB
K_values = sorted(df_results["K"].unique())
n_CB_values = sorted(df_results["n_CB"].unique())

# Define colors for different n_CB values
palette = sns.color_palette("husl", len(n_CB_values))

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, K in enumerate(K_values):
    ax = axes[idx]
    
    for i, n_CB in enumerate(n_CB_values):
        subset = df_results[(df_results["K"] == K) & (df_results["n_CB"] == n_CB)]
        ax.scatter(subset["weight_distance"], subset["score"], label=f"n_CB={n_CB}", color=palette[i])
        
    ax.set_title(f"K = {K}")
    ax.set_xlabel("Weight Distance")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()




# Store models and coefficients
models = {}

# Unique values of K and n_CB
K_values = sorted(df_results["K"].unique())
n_CB_values = sorted(df_results["n_CB"].unique())

for K in K_values:
    for n_CB in n_CB_values:
        # Filter the data for the specific (K, n_CB)
        subset = df_results[(df_results["K"] == K) & (df_results["n_CB"] == n_CB)]
        
        if len(subset) > 1:  # Ensure there's enough data for regression
            X = subset["weight_distance"].values.reshape(-1, 1)  # Reshape for sklearn
            y = subset["score"].values
            
            # Fit the linear model
            model = LinearRegression()
            model.fit(X, y)
            
            # Store the model
            models[(K, n_CB)] = {
                "model": model,
                "coef": model.coef_[0],  # Slope
                "intercept": model.intercept_,
            }

# Print the coefficients
for (K, n_CB), info in models.items():
    print(f"K={K}, n_CB={n_CB} -> Slope: {info['coef']:.4f}, Intercept: {info['intercept']:.4f}")