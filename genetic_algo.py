import numpy as np
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from scipy.stats import boxcox 
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.linalg import pinv2, inv
import time
import numpy as np
import multiprocessing as mp
import pickle
import mlflow
import mlflow.sklearn
import sys
from urllib.parse import urlparse
import logging
import sys


def get_population_fitness(equation_inputs, population):
    #This function gets the fitness score associated with each individual(solution) in the population. 
    fitness_score = np.sum(population*equation_inputs, axis=1)
    return fitness_score
def get_mating_pool(population, fitness_score, mating_parents):
    #This function will return the parents with the best fitness scores for mating to generate new populations
    parents = np.empty((mating_parents, population.shape[1]))
    for mating_parent in range(mating_parents):
        optimum_fitness_idx = np.where(fitness_score == np.max(fitness_score)) #location of the optimum fitness score
        optimum_fitness_idx = optimum_fitness_idx[0][0]
        parents[mating_parent, :] = population[optimum_fitness_idx, :] #getting the parents with optimum fitness scores
        fitness_score[optimum_fitness_idx] = -999
    return parents
def perform_crossover(parents, children_size):
    #this function implements the process of crossing over of genes from mating parents to form new children
    children = np.empty(children_size) #initiating the number of children
    cross_over_loc = np.uint8(children_size[1]/2) #getting the location where the crossover will happen in the children's genes. 
    for k in range(children_size[0]): #for all offsprings .... 
        parent1_idx = k%parents.shape[0] #selecting parent 1 for mating. 
        parent2_idx = (k+1)%parents.shape[0] #selecting the second mating parent
        children[k, 0:cross_over_loc] = parents[parent1_idx, 0:cross_over_loc] #taking the first set of genes from parent 1
        children[k, cross_over_loc:] = parents[parent2_idx, cross_over_loc:]
    return children
def perform_mutation(children_crossover_loc):
    #this function performs slight changes in the genetic structure of the offspring(mutating the genes) to make them different from parents
    for idx in range(children_crossover_loc.shape[0]):
        mutating_agent = np.random.uniform(-1.0, 1.0, 1) #creating a binary mutating agent for the offspring#note for simplicity, the agents are either 0, 1, -1
        children_crossover_loc[idx, 4] = children_crossover_loc[idx, 4] + mutating_agent #introducing the mutation to the genes of the children
        #also note that this mutation is occurring at a specific location in the genes of the children. it may be changed if need be. 
    return children_crossover_loc



windows_10_df = pd.read_csv('windows10_dataset.csv')
features = windows_10_df.drop(columns=['ts', 'type'], axis=1)
numeric_features = features.select_dtypes(exclude=['object'])

## DATA TRANSFOROMATION
## Z-score transformation
def compute_z_score(df):
    z_score_df = df.copy()
    for col in df.columns:
        z_score_df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return z_score_df
## Min-Max transformation
def min_max_scaler(df):
    min_max_scaler_df = df.copy()
    for col in df.columns:
        min_max_scaler_df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
    return min_max_scaler_df
## Yeo-Johnson transformation
def Yeo_Johnson(df):
    yj_df = df.copy()
    yj = PowerTransformer(method='yeo-johnson')
    for col in df.columns:
        yj_df[col] = yj.fit_transform(df[[col]])
    return yj_df 

z_transformed_df = compute_z_score(numeric_features)
scaled_df = min_max_scaler(numeric_features)
yj_df = Yeo_Johnson(numeric_features)

X_original = numeric_features.values
X_z_score = z_transformed_df.values
X_scaled = scaled_df.values
X_yj = yj_df.values
equation_inputs = X_original # features to be optimized. 
num_weights = X_original.shape[1]
solutions_per_population = X_original.shape[0]
num_mating_parents = int(sys.argv[1]) #should be a variable
low_bound = int(sys.argv[2])
high_bound = int(sys.argv[3])
population_size = (solutions_per_population, num_weights)
#creating an initial njew populating from which the mating parents will be selected from. 
#we choose the bounds arbitrarily. This choice is also a variable to be experimented with. 
new_pop = np.random.uniform(low=low_bound, high=high_bound, size=population_size)
#initializing the number of generations over which the algorithm should evolve
generations = 10
with mlflow.start_run():
    for generation in range(generations):
        print("Generation: ", generation)
        fitness_score = get_population_fitness(equation_inputs, new_pop)
        parents = get_mating_pool(new_pop, fitness_score, num_mating_parents)
        children = perform_crossover(parents, children_size =(population_size[0]-parents.shape[0], num_weights))
        children_mutation = perform_mutation(children)
        #creating a new population consisting of both parents and children
        new_pop[0:parents.shape[0], :] = parents
        new_pop[parents.shape[0]:, :] = children_mutation
        #printing the  best result for the specific generation
        print("Optimum fitness score for generation ", str(generation) + ":" + str(np.max(np.sum(new_pop*equation_inputs, axis=1))))
    #Final results 
    fitness_score = get_population_fitness(equation_inputs, new_pop)
    best_fitness_idx = np.where(fitness_score == np.max(fitness_score))
    print("Best weights: ", new_pop[best_fitness_idx, :])
    print("Best fitness score: ", fitness_score[best_fitness_idx]) 
    mlflow.log_param("mating_parents", num_mating_parents)
    mlflow.log_param("low_bound", low_bound)
    mlflow.log_param("high_bound", high_bound)
    mlflow.log_metric("Best fitness", float(fitness_score[best_fitness_idx]))
        
    #mlflow.log_metric("r2", r2)
    #mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    '''

    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(elm1, "model", registered_model_name="ELM_v1")
    else:
        mlflow.sklearn.log_model(elm1, "model")
    '''