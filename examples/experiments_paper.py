
# # Comparsion models 
# In this script we train the different models
import sys
from nsc import datasets
from experiment import *

# Open dataset
dataset = sys.argv[1] # SUPPORT, METABRIC, SYNTHETIC
print("Script running experiments on ", dataset)
x, t, e, _ = datasets.load_dataset(dataset) 


# Hyperparameters and evaluations
horizons = [0.25, 0.5, 0.75]
times = np.quantile(t[e==1], horizons)

max_epochs = 1000
grid_search = 100
layers = [[50], [50, 50], [50, 50, 50], [100], [100, 100], [100, 100, 100]]

# Models

## DSM
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'k' : [2, 3, 4, 5],
    'distribution' : ['LogNormal', 'Weibull'],
    'layers' : layers,
}
DSMExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsm'.format(dataset), times = times).train(x, t, e)

## DCM One risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'k' : [2, 3, 4, 5],
    'layers' : layers,
}
DCMExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dcm'.format(dataset), times = times).train(x, t, e)

## DeepHit
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'nodes' : layers,
}
DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dh'.format(dataset), times = times).train(x, t, e)


## DeepSurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'nodes' : layers,
}
DeepSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_ds'.format(dataset), times = times).train(x, t, e)

## CoxPH
param_grid = {
    'penalizer' : [1e-3, 1.],
}
CoxExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_cox'.format(dataset), times = times).train(x, t, e)

## SuMoNet
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],
    'layers_surv': layers,
    'layers' : layers
}
SuMoExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_sumo'.format(dataset), times = times).train(x, t, e)

## NSC
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'layers_surv': layers,
    'representation': [10, 50, 100],
    'k': [2, 3, 4, 5],
    'layers' : layers,
    'act': ['Tanh'],
}
NSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nsc'.format(dataset), times = times).train(x, t, e)

## TODO: Add your method and the grid search of interest