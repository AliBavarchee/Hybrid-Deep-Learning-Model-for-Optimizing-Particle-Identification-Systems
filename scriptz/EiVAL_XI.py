#!/usr/bin/env python
# coding: utf-8
# %%


import basf2 as b2


import numpy as np

from os import makedirs
from os.path import join, dirname
from tqdm.auto import tqdm


# %%


import argparse
parser = argparse.ArgumentParser()



import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, dirname
from tqdm.auto import tqdm


import h5py    

import gzip
#from sklearn.decomposition import IncrementalPCA
import seaborn as sns

import pkg_resources
import uproot
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#import plotly.graph_objects as go
import re
import awkward
import pprint
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg as la

from math import *

fname = 'DST08.root'
MATRIX_PATH = 'net_wgt.npy'
# train slim data
data_folder = "/home/alibavar/ali5/data1/slim_dstar.h5"

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


# %%


PARTICLES, PDG_CODES, DETECTORS = _make_const_lists()


# %%


DETECTORS


# %%


PARTICLES


# %%


'''PARTICLES, PDG_CODES, DETECTORS = _make_const_lists()
#
PARTICLES = ["e", "mu", "pi", "K", "p", "d"]
#
PDG_CODES = [11, 13, 211, 321, 2212, 1000010020]
#
DETECTORS = ["SVD", "CDC", "TOP", "ARICH", "ECL", "KLM"]
'''



#data_folder = "/home/alibavar/ali5/data1/slim_dstar.h5"



data5 = h5py.File(data_folder,'r+')


data5.keys()


momentum = data5['p'][()]
theta    = data5['theta'][()]
phi      = data5['phi'][()]


# %%


#plot some hists

print('Plot momento ===>')
plt.hist(momentum, bins=30)
plt.xlabel('Momentum [GeV]', color='green')
plt.show()
print('Plot θ ===>')
plt.hist(theta, bins=30, color='r')
plt.xlabel('θ[rad]')
plt.show()
print('Plot φ ===>')
plt.hist(phi, bins=30)
plt.xlabel('φ [rad]', color='b')
plt.show()



# # First define the columns' names of dataframe

# %%


DST_columns = [f'DST_pi_pidLogLikelihoodOf{pdg}From{det}' for det in DETECTORS for pdg in PDG_CODES]


# %%


parse_clumns = []
for pdg in range(len(PDG_CODES)):
    for det in range(len(DETECTORS)):
        clmns = f'DST_pi_pidLogLikelihoodOf{PDG_CODES[pdg]}From{DETECTORS[det]}'
        parse_clumns.append(clmns)
        df_clmns = pd.DataFrame(parse_clumns, columns=['columns name'])
                    


# %%


print('μ deterc:', parse_clumns[6:11])


# ## Forming dataframe by uproot (Columns are loglikelihoods branches of ntuple): MC data

# %%


#fname = 'DST08.root'
tt = uproot.open(fname)
tt.keys()


# %%


loglikelihoood_DST = tt["Dst;1"].arrays(DST_columns, library="pd")


# %%


loglikelihoood_DST


# ### Replace NaN by 0

# %%


loglikelihoood_DST = loglikelihoood_DST.replace(np.nan, 0)
#Dst_loglikelihoood_pi= Dst_loglikelihoood_pi.replace(np.nan, 0)


# ## Importing Weight Matrix comes out from the model

# %%

#MATRIX_PATH = 'net_wgt.npy'
WMat = np.load(MATRIX_PATH)


# %%


print('Weight MAtrix:')
print(WMat)


# ### Flatten the W_Mat

# %%


WMat_1d = WMat.flatten(order='F')


# %%


print(f"The correction ratio for particle <<{PARTICLES[3]}>> and derector <<{DETECTORS[0]}>>==>", WMat_1d[3])
print(f"The correction ratio for particle <<{PARTICLES[2]}>> and derector <<{DETECTORS[1]}>>==>", WMat_1d[8])


# %%


#WMat_list = WMat_1d.tolist()


# ### the multiple to each {particle from detector}

# %%


Wloglikelihoood_DST = loglikelihoood_DST.apply(lambda x: x * WMat_1d, axis=1)


# ### Convert loglikelihoods (after sum all of them) to likelihoods by exponential 

# %%


sum_clmns = ['SUM_Global_loglikelihoods_e', 'SUM_Global_loglikelihoods_moun', 
            'SUM_Global_loglikelihoods_pi', 'SUM_Global_loglikelihoods_K', 'SUM_Global_loglikelihoods_p',
            'SUM_Global_loglikelihoods_d', 'SUM_Global_loglikelihoods']


# %%


SUMloglikelihoood_DST = loglikelihoood_DST.loc[:,sum_clmns[6]] = loglikelihoood_DST.sum(numeric_only=True, axis=1)#.apply(lambda x: float(x))



# %%


loglikelihoood_DST


# %%


loglikelihoood_DST.loc[:,sum_clmns[0]] = loglikelihoood_DST[parse_clumns[0:5]].sum(numeric_only=True, axis=1)

loglikelihoood_DST.loc[:,sum_clmns[1]] = loglikelihoood_DST[parse_clumns[6:11]].sum(numeric_only=True, axis=1)

loglikelihoood_DST.loc[:,sum_clmns[2]] = loglikelihoood_DST[parse_clumns[12:17]].sum(numeric_only=True, axis=1)

loglikelihoood_DST.loc[:,sum_clmns[3]] = loglikelihoood_DST[parse_clumns[18:23]].sum(numeric_only=True, axis=1)

loglikelihoood_DST.loc[:,sum_clmns[4]] = loglikelihoood_DST[parse_clumns[24:29]].sum(numeric_only=True, axis=1)

loglikelihoood_DST.loc[:,sum_clmns[5]] = loglikelihoood_DST[parse_clumns[30:35]].sum(numeric_only=True, axis=1)


# %%


sum_clmns[0]


# %%


loglikelihoood_DST['likelihood_e'] = loglikelihoood_DST[sum_clmns[0]].apply(lambda x: math.exp(x))

loglikelihoood_DST['likelihood_moun'] = loglikelihoood_DST[sum_clmns[1]].apply(lambda x: math.exp(x))


loglikelihoood_DST['likelihood_pi'] = loglikelihoood_DST[sum_clmns[2]].apply(lambda x: math.exp(x))


loglikelihoood_DST['likelihood_K'] = loglikelihoood_DST[sum_clmns[3]].apply(lambda x: math.exp(x))


loglikelihoood_DST['likelihood_p'] = loglikelihoood_DST[sum_clmns[4]].apply(lambda x: math.exp(x))


loglikelihoood_DST['likelihood_d'] = loglikelihoood_DST[sum_clmns[5]].apply(lambda x: math.exp(x))



loglikelihoood_DST['likelihood_ALL'] = loglikelihoood_DST[sum_clmns[6]].apply(lambda x: math.exp(x))


#loglikelihoood_DST['likelihood_ALL'] = loglikelihoood_DST['SUM_Global_loglikelihoods'].apply(lambda x: math.exp(x))



# %%


loglikelihoood_DST_can = loglikelihoood_DST.drop(DST_columns, inplace=True, axis=1)


# %%


loglikelihoood_DST


# ### Again convert modefied loglikelihoods (after sum all of them) to likelihoods by powerd them 

# %%


#SUMWloglikelihoood_DST = Wloglikelihoood_DST.loc[:,'sum_Total_W_likelihood_pi'] = Wloglikelihoood_DST.sum(numeric_only=True, axis=1)
#SUMWloglikelihoood_K = Wloglikelihoood_K.loc[:,'sum_Total_W_likelihood_K'] = Wloglikelihoood_pi.sum(numeric_only=True, axis=1)


# %%


Wloglikelihoood_DST


# %%


Wloglikelihoood_DST.loc[:,sum_clmns[6]] = Wloglikelihoood_DST.sum(numeric_only=True, axis=1)


# %%


sum_clmns[6]


# %%


Wloglikelihoood_DST.loc[:,sum_clmns[0]] = Wloglikelihoood_DST[parse_clumns[0:5]].sum(numeric_only=True, axis=1)

Wloglikelihoood_DST.loc[:,sum_clmns[1]] = Wloglikelihoood_DST[parse_clumns[6:11]].sum(numeric_only=True, axis=1)

Wloglikelihoood_DST.loc[:,sum_clmns[2]] = Wloglikelihoood_DST[parse_clumns[12:17]].sum(numeric_only=True, axis=1)

Wloglikelihoood_DST.loc[:,sum_clmns[3]] = Wloglikelihoood_DST[parse_clumns[18:23]].sum(numeric_only=True, axis=1)

Wloglikelihoood_DST.loc[:,sum_clmns[4]] = Wloglikelihoood_DST[parse_clumns[24:29]].sum(numeric_only=True, axis=1)

Wloglikelihoood_DST.loc[:,sum_clmns[5]] = Wloglikelihoood_DST[parse_clumns[30:35]].sum(numeric_only=True, axis=1)





# %%


Wloglikelihoood_DST['Wlikelihood_e'] = Wloglikelihoood_DST[sum_clmns[0]].apply(lambda x: math.exp(x))

Wloglikelihoood_DST['Wlikelihood_moun'] = Wloglikelihoood_DST[sum_clmns[1]].apply(lambda x: math.exp(x))


Wloglikelihoood_DST['Wlikelihood_pi'] = Wloglikelihoood_DST[sum_clmns[2]].apply(lambda x: math.exp(x))


Wloglikelihoood_DST['Wlikelihood_K'] = Wloglikelihoood_DST[sum_clmns[3]].apply(lambda x: math.exp(x))


Wloglikelihoood_DST['Wlikelihood_p'] = Wloglikelihoood_DST[sum_clmns[4]].apply(lambda x: math.exp(x))


Wloglikelihoood_DST['Wlikelihood_d'] = Wloglikelihoood_DST[sum_clmns[5]].apply(lambda x: math.exp(x))


Wloglikelihoood_DST['Wlikelihood_ALL'] = Wloglikelihoood_DST[sum_clmns[6]].apply(lambda x: math.exp(x))





# %%


Wloglikelihoood_DST_can = Wloglikelihoood_DST.drop(DST_columns, inplace=True, axis=1)


# %%


Wloglikelihoood_DST


# %%


#Wloglikelihoood_DST['likelihood_ALL'] = Wloglikelihoood_DST['sum_Total_W_likelihood_ALL'].apply(lambda x: math.exp(x))
#Wloglikelihoood_K['likelihood_K'] = Wloglikelihoood_K['sum_Total_W_likelihood_K'].apply(lambda x: math.exp(x))




#Wloglikelihoood_DST['Wlikelihood_ALL']




# plots
plt.hist(loglikelihoood_DST['likelihood_ALL'], bins=36, alpha=0.5, label='DST')
plt.hist(Wloglikelihoood_DST['Wlikelihood_ALL'], bins=36, alpha=0.5, label='Weighted DST')
plt.legend(loc='upper left')
plt.title('Histogram Acc. of all likelihoods of all type of particles before and after applyig weights')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
plt.savefig('fig1.png')



# plots
plt.hist(Wloglikelihoood_DST['Wlikelihood_pi']/Wloglikelihoood_DST['Wlikelihood_K'], bins=20, alpha=0.5, label='Weighted DST binary pid')
plt.hist(loglikelihoood_DST['likelihood_pi']/loglikelihoood_DST['likelihood_K'], bins=20, alpha=0.5, label='DST binary PID')

plt.legend(loc='upper left')
plt.title('Histograms of binary likelihoods of pion respect to Kaon before and after applyig weights')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
plt.savefig('fig2.png')


# %%


# plots
plt.hist(Wloglikelihoood_DST['Wlikelihood_pi'], bins=20, alpha=0.5, label='Weighted DST pion')
plt.hist(loglikelihoood_DST['likelihood_pi'], bins=20, alpha=0.5, label='DST pion')

plt.legend(loc='upper left')
plt.title('Histogram likelihoods of pions from deferents detectors before and after applyig weights')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
plt.savefig('fig3.png')


# %%


# plots
plt.hist(loglikelihoood_DST['SUM_Global_loglikelihoods_K'], bins=36, alpha=0.5, label='K')
plt.hist(Wloglikelihoood_DST['SUM_Global_loglikelihoods_K'], bins=36, alpha=0.5, label='Weighted Kaon')
plt.legend(loc='upper left')
plt.title('Histogram likelihoods of Kaon from deferents detectors before and after applyig weights')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
plt.savefig('fig4.png')


# %%


# Binary PID
RATIO = loglikelihoood_DST['likelihood_pi']/loglikelihoood_DST['likelihood_K']

RATIO.plot.line(x=None, y=None)
# Global PID
RATIOE = loglikelihoood_DST['likelihood_pi']/loglikelihoood_DST['likelihood_ALL']

RATIOE.plot.line(x=None, y=None)


# # Calculating likelihood ratios before and after performing weight matrix

# %%


# Binary PID
WRATIO_B = Wloglikelihoood_DST['Wlikelihood_pi']/Wloglikelihoood_DST['Wlikelihood_K']

# Global PID
WRATIO_G = Wloglikelihoood_DST['Wlikelihood_pi']/Wloglikelihoood_DST['Wlikelihood_ALL']

WRATIO_B.plot.line(x=None, y=None)

WRATIO_G.plot.line(x=None, y=None)

mean_pi = Wloglikelihoood_DST['Wlikelihood_pi'].mean()
mean_all = Wloglikelihoood_DST['Wlikelihood_ALL'].mean()
Rai = mean_pi/mean_all
print('likelihood_ratio=>', Rai)




