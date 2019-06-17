## Tests each individual node across trials for lasso logistic regression R squared value.

import scipy.io as spio
import h5py
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

### Extract data
misc = spio.loadmat('/Users/cw/Documents/MATLAB/Goldenberg/JR_setshifting_matlabdata/misc.mat')
misc = misc['misc']
val = misc[0,0]
eventslist = val['eventslist']
eventslist = eventslist[0]

respTimes = []
responses = [] #correct is labelled as 1 and incorrect is labelled as 0
for x in range(1177):
    if eventslist[x][0][0] == u'Correct':
        tresponse = eventslist[x][1][0][0]
        tstim = eventslist[x-1][1][0][0]
        respTime = tresponse - tstim
        respTimes.append(respTime)
        responses.append(1)
    elif eventslist[x][0][0] == u'Incorrect':
        responses.append(0)

respTimes = respTimes[:-1]
# Get trial indexes for correct reponses
index_corrects = [i for i, x in enumerate(responses) if x == 1]
index_corrects = index_corrects[:-1] #remove the last correct response because there was no trial recording for it in setshifting_all

train_scores = []
test_scores = []
nodes = range(159)

# Extract data from one node from each trial and fit lasso regression model 
for n in nodes:

    X = []
    for i in index_corrects:
        trialindex = "%03d"%(i+1)
        f = h5py.File('/Users/cw/Documents/MATLAB/Goldenberg/JR_setshifting_matlabdata/SetShifting_all/trialdata_' + trialindex +'.mat')
        A = f['trialdata'][:]
        A = np.transpose(A)
        A = A[n,1000:2000] # sample the 1000 time samples prior to stimulus for each node

        X.append(A)

    X = np.asarray(X) #change X into numpy ndarray

    ### Split training and testing data

    X_train, X_test, respTimes_train, respTimes_test = train_test_split(X, respTimes, test_size = 0.3)

    ### Fit lasso regression
    lasso = Lasso(alpha = 0.1)
    lasso.fit(X_train,respTimes_train)
    train_score = lasso.score(X_train,respTimes_train)
    test_score = lasso.score(X_test,respTimes_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

plt.plot(nodes, train_scores, 'ro', nodes, test_scores, 'bo')
plt.show()
