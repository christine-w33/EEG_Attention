# Discrete wave transform followed by fitting to lasso logistic regression
# on the 5 channels in each trial with greatest variance

import scipy.io as spio
import h5py
import numpy as np
from sklearn.cross_validation import train_test_split
from pywt import dwt
from sklearn.linear_model import Lasso

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

X = []
for i in index_corrects:
    trialindex = "%03d"%(i+1)
    f = h5py.File('/Users/cw/Documents/MATLAB/Goldenberg/JR_setshifting_matlabdata/SetShifting_all/trialdata_' + trialindex +'.mat')
    A = f['trialdata'][:]
    A = np.transpose(A)

    # Find five traces with biggest range
    ranges = [max(i)-min(i) for i in A]
    nodes_i = sorted(range(len(ranges)), key=lambda i: ranges[i])[-3:]

    # Run discrete wave transform
    trialData = []
    for i in nodes_i:
        n = A[i,:]
        (cA, cD) = dwt(n, 'db1')
        coeffs = np.concatenate((cA, cD), axis = None)
        coeffs = coeffs.tolist()
        trialData.extend(coeffs)

    X.append(trialData)

X = np.asarray(X) #change X into numpy ndarray

### Split training and testing data

X_train, X_test, respTimes_train, respTimes_test = train_test_split(X, respTimes, test_size = 0.3)

### Fit lasso regression
lasso = Lasso()
lasso.fit(X_train,respTimes_train)
train_score = lasso.score(X_train,respTimes_train)
test_score = lasso.score(X_test,respTimes_test)

print "training score:", train_score
print "test score: ", test_score
