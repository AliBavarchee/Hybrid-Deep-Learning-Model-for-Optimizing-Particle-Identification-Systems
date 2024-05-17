#!/usr/bin/env python3

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from sklearn.datasets import load_diabetes, make_regression
from deeprfreg import DeepRFreg  # Import DeepRFreg class here



PARTICLES = ["e", "mu", "pi", "K", "p", "d"]
PDG_CODES = [11, 13, 211, 321, 2212, 1000010020]
#DETECTORS = ["SVD", "CDC", "TOP", "ARICH", "ECL", "KLM"]
DETECTORS = ["SD1", "SD2", "SD3", "SD4", "SD5", "SD6"]

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


#DeepRFreg HyperparameterSearch and methods ...

class HyperparameterSearch:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.mse_grid = np.zeros((len(self.hyperparameters['lr']), len(self.hyperparameters['hidden_units']),
                                 len(self.hyperparameters['dropout']), len(self.hyperparameters['rf_params']['n_estimators']),
                                 len(self.hyperparameters['rf_params']['max_depth'])))
        self.roc_curves = []
        self.best_mse = float('inf')
        self.best_model = None
        self.best_hyperparameters = {}
        self.best_auc = 0.0

    def perform_search(self, X_train, y_train, X_val, y_val):
        for i, lr in enumerate(self.hyperparameters['lr']):
            for j, hidden_units in enumerate(self.hyperparameters['hidden_units']):
                for k, dropout in enumerate(self.hyperparameters['dropout']):
                    for m, n_estimators in enumerate(self.hyperparameters['rf_params']['n_estimators']):
                        for n, max_depth in enumerate(self.hyperparameters['rf_params']['max_depth']):
                            model = DeepRFreg(
                                n_class=6,
                                n_detector=6,
                                const_init=1,
                                rf_params={
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth
                                },
                                pretrained_model=None
                            )
                            model.train_neural(X_train, y_train, device='cpu', epochs=100)
                            model.train_rf(X_train, y_train)
                            combined_predictions = model.predict(X_val)
                            mse = model.score(X_val, y_val)
                            self.mse_grid[i, j, k, m, n] = mse
                            if mse < self.best_mse:
                                self.best_mse = mse
                                self.best_model = model
                                self.best_hyperparameters = {
                                    'lr': lr,
                                    'hidden_units': hidden_units,
                                    'dropout': dropout,
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth
                                }
                                #self.best_model.save_model('best_DeepMMM.pth')
                                for class_idx in range(6):
                                    fpr, tpr, _ = roc_curve(y_val, combined_predictions[:, class_idx], pos_label=class_idx)
                                    roc_auc = auc(fpr, tpr)
                                    self.roc_curves.append((fpr, tpr, roc_auc))
                                    plt.plot(fpr, tpr, label=f'Class {class_idx} (AUC = {roc_auc:.2f})')

                                plt.xlabel('False Positive Rate')
                                plt.ylabel('True Positive Rate')
                                plt.title('ROC Curves for Each Class')
                                #plt.legend()
                                plt.grid()
                                plt.savefig('JJJROC_Curve_final_best_model_T3.pdf')

                                ###
            
        #plt.show()
            
        # Save the best model
        self.best_model.save_model('NNNbest_Deep_RF_XTX.pth')  
        
    def plot_loss_vs_hyperparameters(self, save_path=None):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        lr_values = self.hyperparameters['lr']
        hidden_units_values = self.hyperparameters['hidden_units']
        n_estimators_values = self.hyperparameters['rf_params']['n_estimators']

        for i, lr in enumerate(lr_values):
            for j, hidden_units in enumerate(hidden_units_values):
                for m, n_estimators in enumerate(n_estimators_values):
                    mse = self.mse_grid[i, j, m, :, :].mean()
                    ax.scatter(lr, hidden_units, n_estimators, c=mse, cmap='viridis', s=100)

        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Hidden Units')
        ax.set_zlabel('n_estimators')
        ax.set_title('Loss Function vs Hyperparameters')

        if save_path:
            plt.savefig(save_path + 'NNN_loss_vs_hyperparameters.pdf')
        plt.show()
        
                                
    
    def plot_roc_curves(self, save_path=None):
        num_subplots = len(self.hyperparameters['lr']) * len(self.hyperparameters['hidden_units'])
        fig = make_subplots(rows=len(self.hyperparameters['lr']), cols=len(self.hyperparameters['hidden_units']),
                            subplot_titles=[f"Learning Rate: {lr}, Hidden Units: {hu}" for lr in self.hyperparameters['lr'] for hu in self.hyperparameters['hidden_units']],
                            specs=[[{'rowspan': 1, 'colspan': 1}] * len(self.hyperparameters['hidden_units'])] * len(self.hyperparameters['lr']))

        for i in range(min(num_subplots, len(self.roc_curves))):
            fpr, tpr, roc_auc = self.roc_curves[i]
            row = i // len(self.hyperparameters['hidden_units']) + 1
            col = i % len(self.hyperparameters['hidden_units']) + 1
            trace = go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Class {i % 6} (AUC = {roc_auc:.2f})')
            fig.add_trace(trace, row=row, col=col)
            fig.update_layout(title='ROC Curves for Different Hyperparameter Combinations',
                              showlegend=False
                              )

        if save_path:
            
            fig.write_html(save_path +'NNNROC_Curvez_AAA3U_.html')
        #fig.show()     
        
        
       # fig.show()



    def plot_hyperparam_heatmap(self, save_path=None):
        best_indices = np.unravel_index(np.argmin(self.mse_grid), self.mse_grid.shape)
        best_hyperparameters = {
            'lr': self.hyperparameters['lr'][best_indices[0]],
            'hidden_units': self.hyperparameters['hidden_units'][best_indices[1]],
            'dropout': self.hyperparameters['dropout'][best_indices[2]],
            'n_estimators': self.hyperparameters['rf_params']['n_estimators'][best_indices[3]],
            'max_depth': self.hyperparameters['rf_params']['max_depth'][best_indices[4]],
        }
        heatmap_data = np.log(self.mse_grid.mean(axis=4)[0, 0])

        # Print and save the best hyperparameters
        print("Best Hyperparameters:")
        print(best_hyperparameters)
        if save_path:
            with open(save_path + 'JJJ_best_hyperparameters.txt', 'w') as f:
                for key, value in best_hyperparameters.items():
                    f.write(f"{key}: {value}\n")

        # Plot the heatmap using Matplotlib
        plt.figure(figsize=(13,9))
        plt.imshow(heatmap_data, cmap='viridis', origin='lower')
        plt.colorbar(label='Log Mean Squared Error')
        plt.xlabel('RF n_estimators')
        plt.ylabel('RF max_depth')

        # Highlight the best hyperparameters
        plt.scatter(best_indices[3], best_indices[4], marker='o', color='red', label='Best Hyperparameters')
        plt.xticks(range(len(self.hyperparameters['rf_params']['n_estimators'])), self.hyperparameters['rf_params']['n_estimators'])
        plt.yticks(range(len(self.hyperparameters['rf_params']['max_depth'])), self.hyperparameters['rf_params']['max_depth'])
        plt.legend()

        plt.title('Hyperparameter Heatmap')

        if save_path:
            plt.savefig(save_path + '_heatmap.pdf')
        
        #plt.show()

        # Save the best model's weighted matrix
        if self.best_model:
            weighted_matrix = self.best_model.get_weights(to_numpy=True)[0]
            print('weighted_matrix:', weighted_matrix)
            if save_path:
                np.savez_compressed(save_path + '_weighted_matrix.npz', weighted_matrix)
        # Save the best model
        #self.best_model.save_model('best_DeepYYY.pth')


        # feed the GS:
X_train = X_tr  # training data
y_train = y_tr  # training labels
X_val = X_va    # validation data
y_val = y_va   # validation labels


# hyperparams:
hyperparameters = {
    'lr': [1e-4],
    'hidden_units': [42],
    'dropout': [0.2],
    'rf_params': {
        'n_estimators': [108],
        'max_depth': [6],
    }
}

search = HyperparameterSearch(hyperparameters)
search.perform_search(X_train, y_train, X_val, y_val)
search.plot_roc_curves(save_path='./NNNfinal_best_model_T3U_')
search.plot_hyperparam_heatmap(save_path='./NNNfinal_best_model_T3U_')
search.plot_loss_vs_hyperparameters(save_path='./NNNfinal_best_model_T3U_')



