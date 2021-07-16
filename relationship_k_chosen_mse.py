# -*- coding: utf-8 -*-
"""
Answer for Q4:
This script can present the figure of the relationship between k_chosen and mse.

"""
"""dat has fields:
* `dat['sresp']`: neurons by stimuli, a.k.a. the neural response data (23589 by 4598)
* `dat['xyz']`: three-dimensional position of each neuron in the brain. 
* `dat['run']`: 1 by stimuli, a.k.a. the running speed of the animal in a.u.
* `dat['istim']`: 1 by stimuli, goes from 0 to 2*np.pi, the orientations shown on each trial
* `dat['u_spont']`: neurons by 128, the weights for the top 128 principal components of spontaneous activity. Unit norm.
* `dat['v_spont']`: 128 by 910, the timecourses for the top 128 PCs of spont activity.
* `dat['u_spont'] @ dat['v_spont']`: a reconstruction of the spontaneous activity for 910 timepoints interspersed throughout the recording.
* `dat['stat']`: 1 by neurons, some statistics for each neuron, see Suite2p for full documentation.
"""

#@title Data retrieval
import os, requests
from matplotlib import rcParams 
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] =15
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True
def regloop(k_chosen,X,y):
    pca = PCA(n_components=k_chosen)
    X_ap = pca.fit_transform(X)
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X_ap,y,test_size = 0.3,random_state = 2)
    reg = linear_model.LinearRegression()
    yp = reg.fit(Xtrain,Ytrain).predict(Xtest)
    mse = mean_squared_error(Ytest, yp)
    return mse

dat = np.load('stringer_orientations.npy',allow_pickle=True).item()
choose_range = list(range(1,dat['sresp'].shape[1],2000))
mse = [regloop(k_chosen,dat['sresp'].T,dat['run'].T) for k_chosen in choose_range]

plt.plot(choose_range,mse)

