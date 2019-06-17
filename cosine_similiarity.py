## Calculates the cosine similiarity between nodes and creates a similiarity heatmap.

import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns

trialCosSims = []
for t in range(358):

    trialindex = "%03d"%(t+1)
    f = h5py.File('/Users/cw/Documents/MATLAB/Goldenberg/JR_setshifting_matlabdata/SetShifting_all/trialdata_' + trialindex +'.mat')
    A = f['trialdata'][:]
    A = np.transpose(A)
    X = A[0:158,:] ## <--- change this value to change the nodes being sampled
    CosSims = cosine_similarity(X,X)

    trialCosSims.append(CosSims)

# take the mean of the cosine similiarity across each trial 
similarity_matrix = np.mean(trialCosSims, axis=0)

# plot the heatmap
ax = sns.heatmap(similarity_matrix)
plt.show()
