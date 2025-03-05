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
weight = np.array([1., .8, 1.2, .9, .7, .7, .2, .2])
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

parameters = {'retrieval': WeightedDistanceRetrieval({}), 'adaptation': w_adaptation, 'optimization_method': 'grid',}
retrieval = LearnableWeightedDistanceRetrieval(parameters)

CB_test = NumericalCaseBase.from_numpy(X_test, y_test)
fit_params = {'bounds': (0, 1), 'K': 2, 
              #'optimization_params': { 'max_iter': 10, 'num_particles': 10 } }
              'optimization_params': { 'num_samples': 10, 'n_verbose': 100 } }

retrieval.fit(CB, CB_test, quadratic_distance, fit_params)




