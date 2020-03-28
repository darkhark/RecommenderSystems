from surprise import Dataset, Reader, SVD, NMF, KNNBasic
from surprise.model_selection import cross_validate
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

# load data from a file
file_path = os.path.expanduser('data/restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

# MAE and RMSE are two famous metrics for evaluating the performances of a recommender system.


def compareRmseAndMaeForSvdPmfNmfUserBasedItemBased():
    """
    Returns values for all the algorithms mentioned in the method name

    :return: Nothing
    """
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


def createPlotsDueToSimilarityUsed():
    """
    Plot how Cosine MSD(Mean Squared Difference), and Pearson similarities impact the performances of
    User based Collaborative Filtering andItem based Collaborative Filtering.

    :return: Nothing
    """
    plotArrayRMSE = []
    plotArrayMAE = []
    print("\n-----------3-folds cross validation for User based Collaborative Filtering----------")
    print("-----------MSD----------\n")
    algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': True})
    user_MSD = cross_validate(algo, data, cv=3, verbose=True)
    plotArrayRMSE.append(["User based Collaborative Filtering", 1, user_MSD["test_rmse"].mean()])
    plotArrayMAE.append(["User based Collaborative Filtering", 1, user_MSD["test_mae"].mean()])

    print("\n-----------3-folds cross validation for Item based Collaborative Filtering----------")
    print("-----------MSD----------\n")
    algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': False})
    item_MSD = cross_validate(algo, data, cv=3, verbose=True)
    plotArrayRMSE.append(["Item based Collaborative Filtering", 1, item_MSD["test_rmse"].mean()])
    plotArrayMAE.append(["Item based Collaborative Filtering", 1, item_MSD["test_mae"].mean()])

    print("\n-----------3-folds cross validation for User based Collaborative Filtering----------")
    print("-----------Cosine----------\n")
    algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    user_Cos = cross_validate(algo, data, cv=3, verbose=True)
    plotArrayRMSE.append(["User based Collaborative Filtering", 2, user_Cos["test_rmse"].mean()])
    plotArrayMAE.append(["User based Collaborative Filtering", 2, user_Cos["test_mae"].mean()])

    print("\n-----------3-folds cross validation for Item based Collaborative Filtering----------")
    print("-----------Cosine----------\n")
    algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': False})
    item_Cos = cross_validate(algo, data, cv=3, verbose=True)
    plotArrayRMSE.append(["Item based Collaborative Filtering", 2, item_Cos["test_rmse"].mean()])
    plotArrayMAE.append(["Item based Collaborative Filtering", 2, item_Cos["test_mae"].mean()])

    print("\n-----------3-folds cross validation for User based Collaborative Filtering----------")
    print("-----------pearson----------\n")
    algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': True})
    user_Pear = cross_validate(algo, data, cv=3, verbose=True)
    plotArrayRMSE.append(["User based Collaborative Filtering", 3, user_Pear["test_rmse"].mean()])
    plotArrayMAE.append(["User based Collaborative Filtering", 3, user_Pear["test_mae"].mean()])

    print("\n-----------3-folds cross validation for Item based Collaborative Filtering----------")
    print("-----------pearson----------\n")
    algo = KNNBasic(sim_options={'name': 'MSD', 'user_based': False})
    item_Pear = cross_validate(algo, data, cv=3, verbose=True)
    plotArrayRMSE.append(["Item based Collaborative Filtering", 3, item_Pear["test_rmse"].mean()])
    plotArrayMAE.append(["Item based Collaborative Filtering", 3, item_Pear["test_mae"].mean()])

    plotRmseDF = pd.DataFrame(data=plotArrayRMSE, columns=["Filtering Method Used", "Algorithm", "RMSE"])
    plotRmseDF.pivot("Algorithm", "Filtering Method Used", "RMSE").plot(kind="bar")
    plt.title("User vs Item Based Collaboration (RMSE)")
    plt.ylabel("RMSE")
    plt.ylim(.9, 1.1)
    plt.show()

    plotMaeDF = pd.DataFrame(data=plotArrayMAE, columns=["Filtering Method Used", "Algorithm", "MAE"])
    plotMaeDF.pivot("Algorithm", "Filtering Method Used", "MAE").plot(kind="bar")
    plt.title("User vs Item Based Collaboration (MAE)")
    plt.ylabel("MAE")
    plt.ylim(.7, .9)
    plt.show()


def compareNearestNeighborsUserItemBased():
    plotNearest = []
    i = 1
    while i < 21:
        algo = KNNBasic(k=i, sim_options={'name': 'MSD', 'user_based': True})
        user_MSD = cross_validate(algo, data, cv=3, verbose=True)
        plotNearest.append(["User based Collaborative Filtering", i, user_MSD["test_rmse"].mean()])
        algo = KNNBasic(k=i, sim_options={'name': 'MSD', 'user_based': False})
        item_MSD = cross_validate(algo, data, cv=3, verbose=True)
        plotNearest.append(["Item based Collaborative Filtering", i, item_MSD["test_rmse"].mean()])
        print("\n--------- Iteration:", i, "--------------\n")
        i += 1
    plotDF = pd.DataFrame(data=plotNearest, columns=["Classifier", "K Value", "Score"])
    plotDF.pivot("K Value", "Classifier", "Score").plot(kind="bar")
    plt.ylim(.9, 1.5)
    plt.title("User vs Item Based Collaboration (K-value)")
    plt.ylabel("RMSE")
    plt.show()


compareRmseAndMaeForSvdPmfNmfUserBasedItemBased()
createPlotsDueToSimilarityUsed()
compareNearestNeighborsUserItemBased()

