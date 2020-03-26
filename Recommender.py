from surprise import Dataset, Reader, SVD, NMF, KNNBasic
from surprise.model_selection import cross_validate
import os
import random
import numpy as np

my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

# load data from a file
file_path = os.path.expanduser('data/restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

# MAE and RMSE are two famous metrics for evaluatingthe performances of a recommender system.

# Split the data for 3-folds cross-validation, and compute the
# MAE and RMSE of the SVD(Singular Value Decomposition)algorithm.

# SVD (Singular Value Decomposition)
print("\n-----------3-folds cross validation SVD----------\n")
algo = SVD()
cross_validate(algo, data, cv=3, verbose=True)

# PMF (Probabilistic Matrix Function)
print("\n-----------3-folds cross validation for PMF----------\n")
algo = SVD(biased=False)
cross_validate(algo, data, cv=3, verbose=True)

# NMF (Non-negative Matrix Factorization)
print("\n-----------3-folds cross validation for NMF----------\n")
algo = NMF()
cross_validate(algo, data, cv=3, verbose=True)

# User based Collaborative Filtering
print("\n-----------3-folds cross validation for User based Collaborative Filtering----------\n")
algo = KNNBasic(sim_options={'user_based': True})
cross_validate(algo, data, cv=3, verbose=True)

# Item based Collaborative Filtering
print("\n-----------3-folds cross validation for Item based Collaborative Filtering----------\n")
algo = KNNBasic(sim_options={'user_based': False})
cross_validate(algo, data, cv=3, verbose=True)

print("\n-----------3-folds cross validation for User based Collaborative Filtering----------")
print("-----------MSD----------\n")
algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': True})
cross_validate(algo, data, cv=3, verbose=True)

print("\n-----------3-folds cross validation for User based Collaborative Filtering----------")
print("-----------Cosine----------\n")
algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
cross_validate(algo, data, cv=3, verbose=True)

print("\n-----------3-folds cross validation for User based Collaborative Filtering----------")
print("-----------pearson----------\n")
algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': True})
cross_validate(algo, data, cv=3, verbose=True)

print("\n-----------3-folds cross validation for Item based Collaborative Filtering----------")
print("-----------MSD----------\n")
algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': True})
cross_validate(algo, data, cv=3, verbose=True)

print("\n-----------3-folds cross validation for Item based Collaborative Filtering----------")
print("-----------Cosine----------\n")
algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': True})
cross_validate(algo, data, cv=3, verbose=True)

print("\n-----------3-folds cross validation for Item based Collaborative Filtering----------")
print("-----------pearson----------\n")
algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': True})
cross_validate(algo, data, cv=3, verbose=True)
