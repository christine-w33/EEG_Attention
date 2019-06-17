# To extract the data from MATLAB into Python 

import scipy.io as spio
import numpy as np
import h5py

# 1) go into 'misc.eventslist' and find the "correct" or "incorrect" labels and get their time tables

# Loading misc and getting to eventslist
misc = spio.loadmat('/Users/cw/Documents/MATLAB/Goldenberg/JR_setshifting_matlabdata/misc.mat')
misc = misc['misc']
val = misc[0,0]
eventslist = val['eventslist']
eventslist = eventslist[0]

# Find the time stamps for correct subject responses
tresponses = []
tstims = []
t_to_responses = []
responses = [] #correct is labelled as 1 and incorrect is labelled as 0
for x in range(1177):
    if eventslist[x][0][0] == u'Correct':
        tresponse = eventslist[x][1][0][0]
        tstim = eventslist[x-1][1][0][0]
        t_to_response = tresponse - tstim
        tresponses.append(tresponse)
        tstims.append(tstim)
        t_to_responses.append(t_to_response)
        responses.append(1)
    elif eventslist[x][0][0] == u'Incorrect':
        tresponse = eventslist[x][1][0][0]
        tstim = eventslist[x-1][1][0][0]
        t_to_response = tresponse - tstim
        tresponses.append(tresponse)
        tstims.append(tstim)
        t_to_responses.append(t_to_response)
        responses.append(0)

# 3) find which trials the correct responses occured

index_corrects = [i for i, x in enumerate(responses) if x == 1]
index_corrects = index_corrects[:-1] #remove the last correct response because there was no trial recording for it in setshifting_all

# 5) go into that trial and extract the trace from specific channels from 500ms before stimulus to stimulus

sampleinfo = spio.loadmat('/Users/cw/Documents/MATLAB/Goldenberg/JR_setshifting_matlabdata/sampleinfo.mat')
sampleinfo = sampleinfo['sampleinfo']

n_trials = len(index_corrects)
wavedata = [[] for i in range(n_trials)]
for i in index_corrects:

    # find the index for the time of response within each trial
    t = tresponses[i] - sampleinfo[i][0]
    begin = t - 1500 # designate here the start point for time sample you want to extract

    trialindex = "%03d"%(i+1) #need to add one due to nature of python indexing
    f = h5py.File('/Users/cw/Documents/MATLAB/Goldenberg/JR_setshifting_matlabdata/SetShifting_all/trialdata_' + trialindex +'.mat')
    A = f['trialdata'][:]
    A = A[begin:t, 0] #insert desired time range and channels here
    A = np.transpose(A)
    A.tolist()
    list_counter = index_corrects.index(i)
    wavedata[list_counter] = A
wavedata = np.asarray(wavedata)

print(wavedata)
