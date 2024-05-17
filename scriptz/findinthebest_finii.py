# -*- coding: utf-8 -*-

# #!/usr/bin/env python3

###################################################################################
# This script Initializes DeepRFreg Module to extract the weighted matrix and     #
# other criterion of the best deep RF modelto belonging to                        #
#                            the loglikelihoods of Belle II Detector              #
# 								                  #
#  Writtien by Alì Bavarchee                                                      #
# ##################################################################################


#import basf2 as b2
import ROOT
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import h5py
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, dirname
from tqdm.auto import tqdm
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as la
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, mean_squared_error, auc, accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import glob
import time
import pickle
from xgboost import XGBClassifier
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_diabetes, make_regression
from deeprfreg import DeepRFreg  # Import DeepRFreg class here


""
# _make_const_lists gives some cte values ; Particles and their corrispond PDG_code, Detectors
def _make_const_lists():
    """Moving this code into a function to avoid a top-level ROOT import."""
    import ROOT.Belle2

    PARTICLES, PDG_CODES = [], []
    for i in range(len(ROOT.Belle2.Const.chargedStableSet)):
        particle = ROOT.Belle2.Const.chargedStableSet.at(i)
        name = (particle.__repr__()[7:-1]
                .replace("-", "")
                .replace("+", "")
                .replace("euteron", ""))
        PARTICLES.append(name)
        PDG_CODES.append(particle.getPDGCode())
    # PARTICLES = ["e", "mu", "pi", "K", "p", "d"]
    # PDG_CODES = [11, 13, 211, 321, 2212, 1000010020]

    DETECTORS = []
    for det in ROOT.Belle2.Const.PIDDetectors.set():
        DETECTORS.append(ROOT.Belle2.Const.parseDetectors(det))
    # DETECTORS = ["SVD", "CDC", "TOP", "ARICH", "ECL", "KLM"]

    return PARTICLES, PDG_CODES, DETECTORS

#This is a common pytorch data loader which loads data and splits them to train and test(val)
def load_training_data(directory, p_lims=None, theta_lims=None, device=None):
    """Loads training and validation data within the given momentum and theta
    limits (if given).

    Args:
        directory (str): Directory containing the train and validation sets.
        p_lims (tuple(float), optional): Minimum and maximum momentum. Defaults
            to None.
        theta_lims (tuple(float), optional): Minimum and maximum theta in
            degrees. Defaults to None.
        device (torch.device, optional): Device to move the data onto. Defaults
            to None.

    Returns:
        torch.Tensor: Training log-likelihood data.
        torch.Tensor: Training labels.
        torch.Tensor: Validation log-likelihood data.
        torch.Tensor: Validation labels.
    """
    p_lo, p_hi = p_lims if p_lims is not None else (-np.inf, +np.inf)
    t_lo, t_hi = theta_lims if theta_lims is not None else (-np.inf, +np.inf)
    t_lo, t_hi = np.radians(t_lo), np.radians(t_hi)

    def _load(filename):
        data = np.load(filename)
        X, y, p, t = data["X"], data["y"], data["p"], data["theta"]
        mask = np.logical_and.reduce([p >= p_lo, p <= p_hi, t >= t_lo, t <= t_hi])
        X = torch.tensor(X[mask]).to(device=device, dtype=torch.float)
        y = torch.tensor(y[mask]).to(device=device, dtype=torch.long)
        return X, y

    X_tr, y_tr = _load(join(directory, "train.npz"))
    X_va, y_va = _load(join(directory, "val.npz"))
    return X_tr, y_tr, X_va, y_va


#... Define Input data ... 


data_folder = './data/slim_dstar'
df = load_training_data(data_folder)
X_tr, y_tr, X_va, y_va = load_training_data(data_folder)



        
        



""


""


""
# feed the GS:
X_train = X_tr  # training data
y_train = y_tr  # training labels
X_val = X_va    # validation data
y_val = y_va    # validation labels

""
# Initialize GS

hyperparameters = {
    'lr': [1e-3, 5e-4, 1e-4],
    'hidden_units': [1, 106],
    'dropout': [0.0, 0.2, 0.4],
    'rf_params': {
        'n_estimators': [10, 100],
        'max_depth': [None, 5, 10],
    }
}

search = HyperparameterSearch(hyperparameters)
search.perform_search(X_train, y_train, X_val, y_val)
search.plot_roc_curves(save_path='./final_best_model_XX_')
search.plot_hyperparam_heatmap(save_path='./final_best_model_XX_')

""

