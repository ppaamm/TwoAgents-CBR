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


# Load California housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target  # Target variable (house price in $100,000s)

# Define features (X) and target (y)
X = df.drop(columns=["Price"])
y = df["Price"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Use only two dimensions:
X = X[:,:2]

y = y.to_numpy()

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

N = 100

# X_train = X_train.to_numpy()[:100,:]
# y_train = y_train.to_numpy()[:100]

# X_test = X_test.to_numpy()
# y_test = y_test.to_numpy()

X_train = X_train[:N, :]
y_train = y_train[:N]



# Defining the adaptation method
#weight = np.array([1., 0, 0, 0, 0, 0, 0, 0])
weight = np.array([1., 0])
w_adaptation = WeightedAdaptation({"weight": weight})


# Defining the retrieval strategy

def quadratic_distance(x1, x2):
    return (x1 - x2)**2

dir_ret_parameters = { "loss": quadratic_distance, 
                       "adaptation": w_adaptation}

dir_ret = DirectingRetrieval(dir_ret_parameters)


# Creating the case base:

CB = NumericalCaseBase.from_numpy(X_train, y_train)
CB_list = CB.get_all_cases_as_list()


# Retrieval
x_tgt = X_test[0,:]
y_tgt = np.array([y_test[0]])


case_ret = dir_ret.retrieve(NumericalCase(x_tgt, y_tgt), CB_list, 2)



# LearnableWeightedDistanceRetrieval

parameters = {'retrieval': WeightedDistanceRetrieval({}), 'adaptation': w_adaptation, 
              #'optimization_method': 'grid',}
              'optimization_method': 'pso',}
learnable_weight_retrieval = LearnableWeightedDistanceRetrieval(parameters)

CB_test = NumericalCaseBase.from_numpy(X_test, y_test)
fit_params = {'bounds': (0, 1), 'K': 2, 
              'optimization_params': { 'max_iter': 5, 'num_particles': 5, 
                                      'w': 0.8, 'c1': 5, 'c2': 1 } }
              #'optimization_params': { 'num_samples': 2, 'n_verbose': 100 } }

learnable_weight_retrieval.fit(CB, CB_test, quadratic_distance, fit_params)




# UnknownAdaptationRetrieval

probabilities = [(WeightedAdaptation({"weight": np.array([1., 0.])}), 0.4), 
                 (WeightedAdaptation({"weight": np.array([0., 0.])}), 0.4), 
                 (WeightedAdaptation({"weight": np.array([.5, .5])}), 0.2)]
prior = FiniteAdaptationProbability(probabilities)

retrieval = UnknownFiniteAdaptationRetrieval({"adaptation_probability": prior})

def normal_likelihood(observations, adaptation, sigma2=1e-5):
    N = len(observations)
    residuals = np.array([observations[n][1] - adaptation.adapt(observations[n][2], observations[n][0]) for n in range(N)])
    log_likelihood_value = - (N / 2) * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)
    return np.exp(log_likelihood_value)

n_obs = 10
observations = []
for n in range(n_obs):
    x = X_train[n,:]
    cases = learnable_weight_retrieval.retrieve(x, CB_test, 2)
    y = w_adaptation.adapt(cases, x)
    observations.append((x, y, cases))


retrieval.update_adaptation_probability(observations, normal_likelihood)
print(retrieval.adaptation_probability.adaptation_probabilities)