#!/usr/bin/env python3

######################################################################################
# This script tune DeepRFreg Module is a class of mix of NN and RF regressor model   #
# for Charged Particles and extract Weight Matrix                                    #
# to Optimize the loglikelihoods                                                     #
#								                                                     #
#  Writtien by Alì Bavarchee                                                         #
######################################################################################
import ROOT
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#from .binary_tree import TorchDecisionTreeClassifier
#from .binary_tree import TorchDecisionTreeRegressor
#from .utils import sample_vectors, sample_dimensions
from os import makedirs
from os.path import join, dirname
from tqdm.auto import tqdm
import torch.nn.functional as F
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as la
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob
from os import makedirs
from os.path import join, dirname
from tqdm.auto import tqdm
import time
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.datasets import make_regression
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

data_folder = './data/slim_dstar'
df = load_training_data(data_folder)

X_tr, y_tr, X_va, y_va = load_training_data(data_folder)

#mask pdg code of pion and kaon (target sets) to 1 and -1

#Yy_tr = torch.where(y_tr==2,torch.tensor(1),y_tr)
#Yy_tr = torch.where(y_tr==3,torch.tensor(-1),y_tr)

#Yy_va = torch.where(y_va==2,torch.tensor(1),y_va)
#Yy_va = torch.where(y_va==3,torch.tensor(-1),y_va)



# INPUTS:
X_train= X_tr
y_train=y_tr  #.view(-1)
X_val= X_va
y_val= y_va   #.view(-1)


#DeepRFreg


class DeepRFreg(nn.Module):
    def __init__(self, n_class, n_detector, const_init=None, rf_params=None, pretrained_model=None):
        super(DeepRFreg, self).__init__()  # Call the base class's constructor
        
        # Initialize attributes
        self.n_class = n_class
        self.n_detector = n_detector
        self.const_init = const_init
        self.rf_params = rf_params
        
        self.rf_net = None  # Initialize rf_net

        if pretrained_model is not None:
            self.neural_net = pretrained_model
            self._initialize_pretrained_layers(n_class, n_detector)
        else:
            self.neural_net = models.resnet18(pretrained=False)
            self._initialize_new_layers(n_class, n_detector)
        self.fcs = nn.ModuleList([nn.Linear(n_detector, 1) for _ in range(n_class)])  # Initialize fcs

        if self.const_init is not None:
            self.const_layer = nn.Linear(self.n_detector, self.n_class)
            self.const_layer.bias.data = torch.tensor([self.const_init] * self.n_class, dtype=torch.float32)
        
        if self.rf_params is not None:
            self.random_forest = RandomForestRegressor(
                n_estimators=self.rf_params['n_estimators'],
                max_depth=self.rf_params['max_depth']
            )
            

    def _initialize_new_layers(self, n_class, n_detector):
        num_features = self.neural_net.fc.in_features
        self.neural_net.fc = nn.Sequential(
            nn.Linear(num_features, n_class),  # Adjust output size for regression
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_class, n_detector * n_class)  # Output for each class-detector pair
        )

    def forward(self, x):
        n = self.n_detector
        outs = [self.fcs[i](x[:, i * n: (i + 1) * n]) for i in range(self.n_class)]
        out = torch.cat(outs, dim=1)
        return out

    def const_init(self, const):
        with torch.no_grad():
            for fc in self.fcs:
                fc.weight.fill_(const)

    def random_init(self, mean=1.0, std=0.5):
        with torch.no_grad():
            for fc in self.fcs:
                fc.weight.fill_(0)
                fc.weight.add_(torch.normal(mean=mean, std=std, size=fc.weight.size()))

    def kill_unused(self, only):
        if only is not None:
            for i, pdg in enumerate(PDG_CODES):
                if pdg in only:
                    continue
                self.fcs[i].weight.requires_grad = False
                self.fcs[i].weight.fill_(1)

    def train_neural(self, X_train, y_train, device='cpu', epochs=10, use_tqdm=True):
        self.to(device=device)
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=5e-4)
        iterator = range(epochs)
        neural_losses = []
        rf_losses = []
        if use_tqdm:
            iterator = tqdm(iterator)

        for epoch in iterator:
            self.train()
            opt.zero_grad()
            out = self(X_train)
            y_train_reshaped = y_train.float().reshape(-1, 1)
            neural_loss_value = self.neural_loss(out, y_train_reshaped)
            neural_losses.append(neural_loss_value.item())
            # Perform the backward pass
            neural_loss_value.backward()
            #loss.backward()
            opt.step()
            # Calculate and record RF loss
            # Record neural loss value
            #rf_loss_value = self.rf_net.loss(X_train, y_train).item()
            #rf_losses.append(rf_loss_value)
            rf_losses.append(0)  # Placeholder since random forest doesn't use the same loss
            

    def neural_loss(self, output, target):
        loss = nn.MSELoss()
        return loss(output, target)

    def train_rf(self, X_train, y_train):
        if self.rf_net is None:
            self.rf_net = RandomForestRegressor(**self.rf_params)  # Initialize rf_net if not already initialized
        self.rf_net.fit(X_train, y_train)  # Train rf_net

    def predict(self, X_test):
        neural_predictions = self.forward(X_test)
        rf_predictions = self.rf_net.predict(X_test)
        
        # Detach tensors from the computation graph and then convert to NumPy arrays
        neural_predictions_np = neural_predictions.detach().numpy()
        rf_predictions_np = rf_predictions.reshape(-1, 1)  # Reshape to match neural_predictions
        
        combined_predictions = (neural_predictions_np + rf_predictions_np) / 2.0
        return combined_predictions

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        #mse = mean_squared_error(y_test, predictions)
        # Repeat each target value for each output unit
        y_true_reshaped = np.repeat(y_test[:, np.newaxis], 6, axis=1)
        mse_per_output = mean_squared_error(y_true_reshaped, predictions, multioutput='raw_values')
        avg_mse = mse_per_output.mean()
        return avg_mse

    def get_weights(self, to_numpy=False, device=None):
        neural_weights = None
        if hasattr(self, 'neural_net'):
            neural_weights = [param.detach().cpu().numpy() for param in self.neural_net.parameters()]
   
        rf_weights = self.rf_net.feature_importances_

        if neural_weights is None:
            combined_weights_matrix = rf_weights
        else:
            num_neural_weights = len(neural_weights)
            num_rf_weights = len(rf_weights)
            combined_weights_matrix = np.zeros((num_neural_weights, num_rf_weights))

            for i in range(num_neural_weights):
                for j in range(num_rf_weights):
                    if neural_weights[i].shape == rf_weights[j].shape:
                        combined_weights_matrix[i, j] = neural_weights[i] + rf_weights[j]
                    else:
                        # Handle the case where shapes are not compatible
                        # You might need to reshape, slice, or modify the weights here
                        combined_weights_matrix[i, j] = rf_weights[j]  # Placeholder value, adjust as needed

        if to_numpy:
            return combined_weights_matrix
        else:
            return torch.tensor(combined_weights_matrix, device=device)
    
    def save_model(self, file_path):
        torch.save(self.neural_net.state_dict(), file_path)
        
    def load_model(self, file_path):
        self.neural_net.load_state_dict(torch.load(file_path))
