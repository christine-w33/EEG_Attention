# Plots out the principle components of the EEG data 

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_channels(data, channels):

    for channel in channels:
        X = data[:, channel]
        plt.plot(X)

    plt.xlabel('Time')
    plt.ylabel('mV')
    plt.axis('tight')
    plt.title('Channels')
    plt.show()
    return

def plot_PCA(principalComponents):

    for r in range(159):
        pc = principalComponents[:,r]
        plt.plot(pc)

    plt.xlabel('Time')
    plt.ylabel('pc')
    plt.axis('tight')
    plt.title('Principal Components')
    plt.show()
    return

def plot_variance(variance):

    y_pos = range(variance.size)
    variance = variance.tolist()
    plt.bar(y_pos, variance)
    plt.ylabel('variance')
    plt.xlabel('PCA')
    plt.title('Variance of Principal Components')
    plt.show()
    return

### Get data
f = h5py.File('/Users/cw/Documents/MATLAB/Goldenberg/JR_setshifting_matlabdata/SetShifting_all/trialdata_032.mat')
data = f['trialdata'][:]

data = data[1000:2000,:]

### Plot EEG channels over time
channels = range(159)
plot_channels(data, channels)
### Perform PCA across channels

X = data[:, channels]
X = np.transpose(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA()
principalComponents = pca.fit_transform(X)
#plot_PCA(principalComponents)
variance = pca.explained_variance_ratio_
#plot_variance(variance)

#print(principalComponents)
#print(principalComponents.shape)
