# -*- coding: utf-8 -*-
"""
Demo script for the EMPRISE analysis pipeline

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2024-04-04, 11:29: settings, loading, simulate
2024-04-04, 17:58: analyze, processing, visualize
2024-04-05, 09:12: processing, visualize, figures
2025-10-02, 14:49: replaced R²>0.2 by p<0.05 (Bonferroni),
                   added seeding of random number generator
2025-10-08, 16:07: enabled simulation using linear vs. logarithmic
                   and analysis using linear vs. logarithmic
"""


### Introduction ##############################################################

# This Python script demos the numerosity population receptive field (NumpRF)
# estimation pipeline used in the project on "EMergence of PRecISE numerosity
# representations in the human brain" (EMPRISE). The script consists of the
# following steps:
# 1. specification of simulation settings
# 2. loading of onsets and confounds
# 3. simulation of voxel-wise data
# 4. analysis of voxel-wise data
# 5. processing of simulation results
# 6. visualization of simulation results
# 7. recreation of figures analoguos to manuscript


### Step 1: specify settings ##################################################

# import modules
import time
import PySPMs as PySPM
import NumpRF
import EMPRISE
import Figures
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# specify simulation/analysis model
simulate = 'log'                # model for simulation
analyze  = 'log'                # model for analysis
# Explanation: Simulated signals are generated with log(arithmic) or lin(ear)
# tuning functions and analyzed with log(arithmic) or lin(ear) tuning models.

# specify fMRI repetition time
TR = EMPRISE.TR
# Explanation: The TR is set to the TR within the EMPRISE project (2.1 s).

# specify number of voxels
v  = 2000                       # total number of voxels
p  = 0.5                        # voxels with numerosity-related signal
# Explanation: Half of the voxels are set to be responsive to numerosity.

# sample preferred numerosities
mu = np.zeros(v)
for j in range(v):
    if np.random.uniform() <= p:
        mu[j] = np.random.gamma(2,1) + 1
    else:
        mu[j] = 0
# Explanation: If a voxel is responsive to numerosity, its preferred numerosity
# is sampled from a non-central gamma distribution with shape a = 2, rate b = 1
# and non-centrality d = 1. This leads to a distribution with most mass near 2,
# all values larger than 1 and only few values larger than 5.

# sample tuning width
m_xy = 5
n_xy = 0
fwhm = m_xy*mu + n_xy + np.random.normal(0, 2, mu.shape)
fwhm[fwhm<0] = -fwhm[fwhm<0]
fwhm[mu==0]  = 0
# Explanation: Figure 5c from the manuscript shows the linear relationship
# between mu and fwhm "for a representative subject" and suggests fwhm = m*mu+n
# with m = 5 and n = 0. Thus, fwhm was sampled from mu compatibly, adding
# normal random noise with mean mu = 0 and standard deviation sig = 2.

# specify between-voxel variance
s2_k = 1
# Explanation: Each voxel will be simulated seperately and before that,
# regression coefficients will be sampled from normal distributions with
# variance 1 (see below).

# specify between-run variance
s2_j = s2_k/10
# Explanation: The between-run variance is assumed to be one magnitude smaller
# than the between-voxel variance. This means that regression coefficients vary
# considerably between voxels, but less so within one voxel, enforcing a
# reasonable level of stationarity across runs.

# specify autocorrelation
tau  = 0.2
# Explanation: SPM estimates an AR(1) process with AR parameter p = 0.2. This
# corresponds to a correlation of 0.2 between directly adjacent fMRI scans.
# Sources:
# - https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=ind1607&L=SPM&P=R124872
# - https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=ind1610&L=SPM&P=R67251
# Note: The estimation pipeline assumes i.i.d. errors. Generating data with
# serial correlations is therefore a test how robust the analysis pipeline is
# with respect to violation of this assumption (which is likely in fMRI).

# set HRF parameters
hrf_mean = np.array([  6, 16,  1,  1,  6,  0, 32])
hrf_std  = np.array([0.5,  1,0.1,0.1,0.5,  0,  0])
# Explanation: HRF parameters are sampled from N(hrf_mean, hrf_std). Thus,
# 99.7% of values will be within [hrf_mean-3*hrf_std, hrf_mean+3*hrf_std]:
# - delay of response will be approximately 4.5 < p < 7.5
# - delay of undershoot will be 13 < p < 19
# - dispersion of response will be 0.7 < p < 1.3
# - dispersion of undershoot will be 0.7 < p < 1.3
# - ratio of response to undershoot will be 4.5 < p < 7.5
# - onset of function will always be 0
# - length of kernel will always be 32
# Sources:
# - https://statproofbook.github.io/P/norm-probstd
# - https://github.com/SkeideLab/EMPRISE/blob/e7731365c878ce7d6668bc668c3ed113acc31259/data/code/harveymodel.py#L549
# Note: The estimation pipeline assumes the canonical HRF. Generating data with
# variability in the HRF is therefore a test how robust the analysis pipeline is
# with respect to violation of this assumption (which is likely in fMRI).

# preallocate scaling parameters
mu_b = np.zeros(mu.shape)       # mean numerosity signal effect
mu_c = np.zeros(mu.shape)       # mean confound variable effect
s2_i = np.zeros(mu.shape)       # within-run variance
# Explanation: These regression coefficients are here set to zero and then
# later filled with their simulated values (see Step 3).

# specify confound variables
labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
          'white_matter', 'csf', 'global_signal', \
          'cosine00', 'cosine01', 'cosine02']
# Explanation: For this demo/simulation, the same confound variables that were
# used for empirical data analysis are employed.


### Step 2: load logfiles #####################################################

# specify subject/session
sub  = '001'
ses  = 'visual'
sess = EMPRISE.Session(sub, ses)

# load onsets and durations
filenames = []
for run in EMPRISE.runs:
    filename = sess.get_events_tsv(run)
    filename = 'Demo_logfiles/'+filename[filename.find('func/')+5:]
    filenames.append(filename)
ons, dur, stim = sess.get_onsets(filenames)
ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')

# load confound variables
filenames = []
for run in EMPRISE.runs:
    filename = sess.get_confounds_tsv(run)
    filename = 'Demo_logfiles/'+filename[filename.find('func/')+5:]
    filenames.append(filename)
X_c = sess.get_confounds(labels, filenames)
X_c = EMPRISE.standardize_confounds(X_c)


### Step 3: simulate data #####################################################

# prepare simulation
np.random.seed(0)               # initialize RNG
ds  = NumpRF.DataSet(0, ons, dur, stim, TR, X_c)
n   = X_c.shape[0]              # number of scans per run
p   = X_c.shape[1]+2            # number of regressors (= numerosity + confounds + baseline)
r   = len(EMPRISE.runs)         # number of runs

# preallocate simulanda
Y   = np.zeros((n,v,r))         # data matrix
S   = np.zeros((n,v,r))         # numerosity signals
B   = np.zeros((p,v,r))         # regression coefficients
SNR = np.zeros((r,v))           # signal-to-noise ratios
HRF = np.zeros((v,hrf_mean.size))#hemodynamic parameters

# for all voxels
print('\n-> Simulate {} voxels (s2_k = {}, s2_j = {}, tau = {}):'.
      format(v, s2_k, s2_j, tau))
for i in range(v):
    
    # sample task-related scaling
    print('   - Voxel {}: '.format(i+1), end='')
    if mu[i] == 0:
        mu_b[i] = 0
        mu[i]   = 0.001
        fwhm[i] = 0.001
    else:
        mu_b[i] = np.random.normal(5, np.sqrt(s2_k))
    print('mu = {:.2f}, fwhm = {:.2f}, mu_b = {:.2f}, '.format(mu[i], fwhm[i], mu_b[i]), end='')
    # Explanation: If mu is 0, this is a noise voxel. In this case, set the
    # tuning parameters to very small and set the scaling parameter to 0 in
    # order to produce no numerosity-related signal. If mu is not 0, sample
    # the scaling parameter from N(5,1), i.e. approximately within [2, 8].
    
    # sample nuisance scaling
    mu_c[i] = np.random.normal(0, np.sqrt(s2_k))
    print('mu_c = {:.2f}, '.format(mu_c[i]), end='')
    # Explanation: Confound variables are added regardless of tuning parameters.
    # However, confounds are assumed to be zero across voxels. In each voxel, 
    # sample the scaling parameters from N(0,1), i.e. within [-3, +3].
    
    # sample hemodynamic parameters
    HRF[i,:] = np.random.normal(hrf_mean, hrf_std)
    hrf_str  = '['+', '.join(['{:.2f}'.format(x) for x in HRF[i,:]])+']'
    # Explanation: see detailed explanation of hemodynamic parameters above.
    
    # pre-simulate to calculate SNR
    y, s, x, b = ds.simulate(
        mu[[i]], fwhm[[i]], mu_b[i], mu_c[i], 0, 0, 0, tau,
        HRF[i,:], simulate=='lin')
    if mu[i] == 0.001:
        VXb = np.var(x[:,1:,0] @ b[1:,0,0])
    else:
        x[:,0,0] = s[:,0,0]
        VXb = np.var(x[:,:,0] @ b[:,0,0])
    # Explanation: The variance of the true signal X times beta is calculated
    # as the numerator of the signal-to-noise ratio.
    
    # sample within-run variance term
    snr     = np.random.gamma(shape=1.5, scale=1)
    s2_i[i] = VXb/snr
    print('s2_i = {:.2f},'.format(s2_i[i]), end='\n')
    print('     HRF: {}; '.format(hrf_str), end='')
    # Explanation: The signal to noise-ratio is sampled from Gam(1.5,1), such
    # that mean(SNR) = 1.5 and var(SNR) = 1.5. The within-run variance is then
    # calculated as s2_i = Var(Xb)/SNR (because SNR = Var(Xb)/Var(e)).
    
    # sample simulated data
    Y[:,[i],:], S[:,[i],:], X, B[:,[i],:] = ds.simulate(
        mu[[i]], fwhm[[i]], mu_b[i], mu_c[i],
        0, s2_j, s2_i[i], tau, HRF[i,:], simulate=='lin')
    # Explanation: see "NumpRF.DataSet.simulate"; s2_k is set to 0, because
    # it was already applied above and each voxel is simulated separately.
    
    # calculate SNR
    for j in range(r):
        if mu[i] == 0.001:
            Y[:,i,j] = Y[:,i,j] - S[:,i,j] * B[0,i,j]
            SNR[j,i] = np.var(X[:,1:,j] @ B[1:,i,j])/s2_i[i]
        else:
            X[:,0,j] = S[:,i,j]
            SNR[j,i] = np.var(X[:,:,j] @ B[:,i,j])/s2_i[i]
    print('mean SNR = {:.2f}.'.format(np.mean(SNR[:,i])), end='\n')
    # Explanation: see above; SNR = Var(Xb)/Var(e).

# clean up simulation
del hrf_str, ds, y, s, x, b, VXb, snr
t = np.arange(0, EMPRISE.n*EMPRISE.TR, EMPRISE.TR)
print('\n-> Simulated SNRs:')
print('   - no-signal voxels: median SNR = {:.2f};'.format(np.median(SNR[:,mu==0.001])))
print('   - numerosity voxels: median SNR = {:.2f};'.format(np.median(SNR[:,mu!=0.001])))

# save simulated data (Python)
# filename = 'Demo_results/Demo_Data'+'_simulate-'+simulate+'_analyze-'+analyze+'.npz'
# np.savez(filename, Y=Y, S=S, X=X, B=B, mu=mu, fwhm=fwhm, SNR=SNR)

# save simulated data (MATLAB)
settings = {'sub': sub, 'ses': ses, 'labels': labels, \
            'onsets': ons, 'durations': dur, 'stimuli': stim, \
            'mu_b': mu_b, 'mu_c': mu_c, 's2_k': s2_k, 's2_j': s2_j, 's2_i': s2_i, \
            'tau': tau, 'HRF': HRF}
res_dict = {'Y': Y, 'S': S, 'X': X, 'B': B, \
            'mu': mu, 'fwhm': fwhm, 'SNR': SNR, \
            'settings': settings}
filename = 'Demo_results/Demo_Data'+'_simulate-'+simulate+'_analyze-'+analyze+'.mat'
sp.io.savemat(filename, res_dict)


### Step 4: analyze data ######################################################

# specify estimation settings
avg   = [True, False]           # average over runs, but not cycles
corr  = 'iid'                   # assume i.i.d. errors
order = 1                       # use no HRF derivatives
Q_set = None                    # use no covariance components

# specify parameter grids
mu_grid   = np.concatenate((np.arange(0.80, 5.25, 0.05), np.array([20])))
sig_grid  = np.arange(0.05, 3.05, 0.05)
fwhm_grid = None
# Explanation: These are the default settings in "NumpRF.DataSet.estimate_MLE".
# This is also explained in the manuscript: "To this end, we specified a large
# grid of plausible values: mu from 0.8 to 5.2 in steps of 0.05, sig_log from
# 0.05 to 30 in steps of 0.05 (such that 0.12 < fwhm < 34.17 for the lowest
# presented numerosity mu=1 and 0.59 < fwhm < 170.9 for the highest presented
# numerosity mu=5; cf. equation above)."

# initialize data set
ds = NumpRF.DataSet(Y, ons, dur, stim, TR, X_c)
start_time = time.time()

# estimate ML parameters
mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
    ds.estimate_MLE(avg, corr, order, Q_set, mu_grid, sig_grid, fwhm_grid, analyze=='lin')
k_est, k_null, k_const =\
    ds.free_parameters(avg, corr, order)

# calculate estimation time
end_time   = time.time()
difference = end_time - start_time
del start_time, end_time

# save estimated parameters (Python)
# filename = 'Demo_results/Demo_Analysis'+'_simulate-'+simulate+'_analyze-'+analyze+'.npz'
# np.savez(filename, mu_est=mu_est, fwhm_est=fwhm_est, beta_est=beta_est, \
#                    MLL_est=MLL_est, MLL_null=MLL_null, MLL_const=MLL_const)

# save estimated parameters (MATLAB)
settings = {'avg': avg, 'corr': corr, 'order': order, \
            'mu_grid': mu_grid, 'sig_grid': sig_grid, 'fwhm_grid': np.nan}
res_dict = {'mu_est': mu_est, 'fwhm_est': fwhm_est, 'beta_est': beta_est, \
            'MLL_est': MLL_est, 'MLL_null': MLL_null, 'MLL_const': MLL_const, \
            'version': 'V2', 'time': difference, 'settings': settings}
filename = 'Demo_results/Demo_Analysis'+'_simulate-'+simulate+'_analyze-'+analyze+'.mat'
sp.io.savemat(filename, res_dict)


### Step 5: process results ###################################################

# calculate R-squared
Rsq_est = NumpRF.MLL2Rsq(MLL_est, MLL_const, n)
pval    = NumpRF.Rsq2pval(Rsq_est, n, p=4)
sig     = NumpRF.Rsqsig(Rsq_est, n, p=4, alpha=0.05, meth='B')
# Explanation: The number of parameters in the numerosity model is four.
# This is also explained in the manuscript: "To assess statistical significance
# of variance explanation by the tuning model, we performed an F-test of the
# model including the numerosity regressor (...) against a null model including
# only the baseline regressor: F = ((RSS0-RSS)/(p-1)) / (RSS/(n-p)) where
# n = 145 is the number of scans per run and p = 2 is the number of free
# parameters in the numerosity pRF model (beta_0, beta)."

# filter numerosity voxels (according to analysis)
ind = mu_est > 0
ind = np.logical_and(ind, beta_est>0)
ind = np.logical_and(ind, np.logical_and(mu_est>=1, mu_est<=5))
ind = np.logical_and(ind, pval < 0.05/v) # = np.logical_and(ind, sig)
# Explanation: Filtering is used to identify voxels responsive to numerosity.
# This is also explained in the manuscript: "Only vertices that exhibited a
# positive scaling factor (beta>0), an estimated preferred numerosity inside
# the presented stimulus range (1 <= mu <= 5) and statistically significant
# variance explained (p < 0.05, Bonferroni-corrected; see below) were included
# in the following analyses."

# filter numerosity voxels (according to ground truth)
num = mu!=0.001
# Explanation: Voxels whose preferred numerosity was not set to very small
# during the simulation are numerosity-selective voxels (see above).

# store ground truth
mu_true   = mu                  # actual preferred numerosities
fwhm_true = fwhm                # actual FWHM tuning widths
beta_true = mu_b                # actual scaling factors


### Step 6: visualize results #################################################

### Step 6a: signal-to-noise ratios ###########################################

# select voxels
inds   = [np.where(num)[0], np.where(~num)[0]]
titles = ['numerosity-selective voxels', 'no-signal voxels']

# plot signal-to-noise ratios
fig = plt.figure(figsize=(20,10))
axs = fig.subplots(1,len(inds))
for j, ax in enumerate(axs):
    bins = np.arange(0, 10.25, 0.25)
    SNRs = np.mean(SNR[:,inds[j]], axis=0)
    ax.hist(SNRs, bins, color='dodgerblue')
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('signal-to-noise ratio', fontsize=28)
    ax.set_ylabel('number of occurences', fontsize=28)
    ax.set_title(titles[j], fontsize=28)
fig.show()


### Step 6b: numerosity tuning parameters #####################################

# plot tuning parameters
fig = plt.figure(figsize=(20,10))
axs = fig.subplots(1,len(inds))
for j, ax in enumerate(axs):
    ax.plot(mu_true[inds[j]], fwhm_true[inds[j]], 'ob')
    ax.axis([-0.1, 10, -0.1, 40])
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('true preferred numerosity', fontsize=28)
    ax.set_ylabel('true FWHM tuning width', fontsize=28)
    ax.set_title(titles[j], fontsize=28)
fig.show()


### Step 6c: estimated preferred numerosities #################################

# plot tuning parameters
fig = plt.figure(figsize=(20,10))
axs = fig.subplots(1,len(inds))
for j, ax in enumerate(axs):
    ax.plot(mu_true[inds[j]], mu_est[inds[j]], 'ob')
    ax.axis([-0.1, 10, -0.1, 10])
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('true preferred numerosity', fontsize=28)
    ax.set_ylabel('estimated preferred numerosity', fontsize=28)
    ax.set_title(titles[j], fontsize=28)
fig.show()


### Step 6d: estimated preferred numerosities #################################

# plot tuning parameters
fig = plt.figure(figsize=(20,10))
axs = fig.subplots(1,len(inds))
for j, ax in enumerate(axs):
    ax.plot(fwhm_true[inds[j]], fwhm_est[inds[j]], 'ob')
    ax.axis([-0.1, 40, -0.1, 40])
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('true FWHM tuning width', fontsize=28)
    ax.set_ylabel('estimated FWHM tuning width', fontsize=28)
    ax.set_title(titles[j], fontsize=28)
fig.show()


### Step 6e: plot recognition rates ###########################################
# Explanation: This section investigates how many numerosity-selective vertices
# are identified as such ("hits", rather than "misses) and how many vertices
# not responsive to numerosity are identified as such ("correct rejections",
# rather than "false alarms"). This demonstrates that the technique is
# sensitive as well as specific (high statistical power).

# calculate recognition rates
H  = np.sum(np.logical_and(num, ind))/np.sum(num)
M  = np.sum(np.logical_and(num, ~ind))/np.sum(num)
FA = np.sum(np.logical_and(~num, ind))/np.sum(~num)
CR = np.sum(np.logical_and(~num, ~ind))/np.sum(~num)

# open figure
fig   = plt.figure(figsize=(16,9))
ax    = fig.add_subplot(111)
col   = 'b'
names =['hits', 'misses', 'false alarms', 'correct rejections']

# plot recognition rates
ax.plot([0-1, len(names)], [0, 0], '-k', linewidth=1)
ax.bar(np.arange(len(names)), np.array([H, M, FA, CR]), color='b', edgecolor='k')
ax.axis([0-1, len(names), 0-0.1, 1+0.1])
ax.set_xticks(np.arange(len(names)), labels=names, fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel('performance measure', fontsize=28)
ax.set_ylabel('frequency', fontsize=28)
ax.set_title('Identification of numerosity-selective and '+titles[1], fontsize=28)
fig.show()


### Step 7: create paper figures ##############################################

### Step 7a: create analogue of Figure 2a-d ###################################
# Explanation: This section follows "Figures.WP1_Fig1.plot_tuning_function_time_course".

# get session data
# Y              - data matrix
# X_c            - confound variables
# ons, dur, stim - onsets, durations, stimuli
ons0,dur0,stim0  = ons[0], dur[0], stim[0]
ons, dur, stim   = EMPRISE.correct_onsets(ons0, dur0, stim0)

# load model results
mu   = mu_est
fwhm = fwhm_est
beta = beta_est

# calculate vertex-wise R-squared
MLL1 = MLL_est
MLL0 = MLL_const
Rsq  = NumpRF.MLL2Rsq(MLL1, MLL0, n)

# select vertices for plotting
verts = [np.argmax(Rsq + np.logical_and(mu>1, mu<2)),
         np.argmax(Rsq + np.logical_and(mu>4, mu<5))]
if np.isnan(Rsq[verts[0]]) or np.isnan(Rsq[verts[1]]):
    verts = [np.nanargmax(Rsq + np.logical_and(mu>1, mu<2)),
             np.nanargmax(Rsq + np.logical_and(mu>4, mu<5))]

# plot selected vertices
fig = plt.figure(figsize=(24,len(verts)*8))
axs = fig.subplots(len(verts), 2, width_ratios=[4,6])
col = 'b'                   # plot in blue
xr  = EMPRISE.mu_thr        # numerosity range
xm  = xr[1]+1               # maximum numerosity
dx  = 0.05                  # numerosity delta

# Figure 2a/c: estimated tuning functions
for k, vertex in enumerate(verts):

    # compute vertex tuning function
    x  = np.arange(dx, xm+dx, dx)
    xM = mu[vertex]
    if analyze == 'log':
        mu_log, sig_log = NumpRF.lin2log(mu[vertex], fwhm[vertex])
        y  = NumpRF.f_log(x, mu_log, sig_log)
        x1 = np.exp(mu_log - math.sqrt(2*math.log(2))*sig_log)
        x2 = np.exp(mu_log + math.sqrt(2*math.log(2))*sig_log)
        x2 = np.min(np.array([x2,xm]))
    else:
        mu_lin, sig_lin = (mu[vertex], NumpRF.fwhm2sig(fwhm[vertex]))
        y  = NumpRF.f_lin(x, mu_lin, sig_lin)
        x1 = mu[vertex] - fwhm[vertex]/2
        x2 = mu[vertex] + fwhm[vertex]/2
        x1 = np.max(np.array([0,x1]))
        x2 = np.max(np.array([x2,xm]))
    
    # plot vertex tuning function
    hdr  =('truth: mu = {:.2f}, fwhm = {' + ':.{}f'.format([1,2][int(fwhm[vertex]<10)]) + '}'). \
            format(mu_true[vertex], fwhm_true[vertex])
    txt1 = 'vertex {}'.format(vertex)
    txt2 =('mu = {:.2f}\n fwhm = {' + ':.{}f'.format([1,2][int(fwhm[vertex]<10)]) + '}'). \
            format(mu[vertex], fwhm[vertex])
    axs[k,0].plot(x[x<=xr[0]], y[x<=xr[0]], '--'+col, linewidth=2)
    axs[k,0].plot(x[x>=xr[1]], y[x>=xr[1]], '--'+col, linewidth=2)
    axs[k,0].plot(x[np.logical_and(x>=xr[0],x<=xr[1])], y[np.logical_and(x>=xr[0],x<=xr[1])], '-'+col, linewidth=2)
    if k == 0:
        axs[k,0].plot([xM,xM], [0,1], '-', color='gray', linewidth=2)
        axs[k,0].plot([x1,x2], [0.5,0.5], '-', color='gray', linewidth=2)
        axs[k,0].text(xM, 0+0.01, ' mu = preferred numerosity', fontsize=18, \
                      horizontalalignment='left', verticalalignment='bottom')
        axs[k,0].text((x1+x2)/2, 0.5-0.01, 'fwhm = tuning width', fontsize=18, \
                      horizontalalignment='center', verticalalignment='top')
    axs[k,0].axis([0, xm, 0, 1+(1/20)])
    if k == len(verts)-1:
        axs[k,0].set_xlabel('presented numerosity', fontsize=32)
    axs[k,0].set_ylabel('neuronal response', fontsize=32)
    axs[k,0].set_title(hdr, fontsize=32)
    axs[k,0].tick_params(axis='both', labelsize=18)
    axs[k,0].text(xm-(1/20)*xm, 0.85, txt1, fontsize=18,
                  horizontalalignment='right', verticalalignment='top')
    axs[k,0].text(xm-(1/20)*xm, 0.05, txt2, fontsize=18,
                  horizontalalignment='right', verticalalignment='bottom')
    del x, y, xM, x1, x2, hdr, txt1, txt2

# Figure 2b/d: predicted time courses
for k, vertex in enumerate(verts):

    # compute "observed" signal
    y     = EMPRISE.standardize_signals(Y[:,[vertex],:]) - 100
    y_reg = np.zeros(y.shape)
    for j in range(y.shape[2]):
        glm          = PySPM.GLM(y[:,:,j], np.c_[X_c[:,:,j], np.ones((EMPRISE.n,1))])
        b_reg        = glm.OLS()
        y_reg[:,:,j] = glm.Y - glm.X @ b_reg
    
    # get vertex tuning parameters
    if analyze == 'log':
        mu_log, sig_log = NumpRF.lin2log(mu[vertex], fwhm[vertex])
    else:
        mu_lin, sig_lin = (mu[vertex], NumpRF.fwhm2sig(fwhm[vertex]))
    
    # compute predicted signal (run)
    y_run, t = EMPRISE.average_signals(y_reg, None, [True, False])
    if analyze == 'log':
        z, t = NumpRF.neuronal_signals(ons0, dur0, stim0, EMPRISE.TR, EMPRISE.mtr,
                                       np.array([mu_log]), np.array([sig_log]), lin=False)
    else:
        z, t = NumpRF.neuronal_signals(ons0, dur0, stim0, EMPRISE.TR, EMPRISE.mtr,
                                       np.array([mu_lin]), np.array([sig_lin]), lin=True)
    if True:        
        s, t = NumpRF.hemodynamic_signals(z, t, EMPRISE.n, EMPRISE.mtr)
    glm   = PySPM.GLM(y_run, np.c_[s[:,:,0], np.ones((EMPRISE.n, 1))])
    b_run = glm.OLS()
    s_run = glm.X @ b_run
    
    # compute predicted signal (epoch)
    y_avg, t = EMPRISE.average_signals(y_reg, None, [True, True])
    # Note: For visualization purposes, we here apply "avg = [True, True]".
    if analyze == 'log':
        z, t = NumpRF.neuronal_signals(ons, dur, stim, EMPRISE.TR, EMPRISE.mtr,
                                       np.array([mu_log]), np.array([sig_log]), lin=False)
    else:
        z, t = NumpRF.neuronal_signals(ons, dur, stim, EMPRISE.TR, EMPRISE.mtr,
                                       np.array([mu_lin]), np.array([sig_lin]), lin=True)
    if True:
        s, t = NumpRF.hemodynamic_signals(z, t, EMPRISE.scans_per_epoch, EMPRISE.mtr)
    glm   = PySPM.GLM(y_avg, np.c_[s[:,:,0], np.ones((EMPRISE.scans_per_epoch, 1))])
    b_avg = glm.OLS()
    s_avg = glm.X @ b_avg
    
    # assess statistical significance
    Rsq_run = Rsq[vertex]
    Rsq_avg = NumpRF.yp2Rsq(y_avg, s_avg)[0]
    h, p_run, stats = NumpRF.Rsqtest(y_run, s_run, p=4)
    h, p_avg, stats = NumpRF.Rsqtest(y_avg, s_avg, p=4)
    
    # prepare axis limits
    y_min = np.min(y_avg)
    y_max = np.max(y_avg)
    y_rng = y_max-y_min
    t_max = np.max(t)
    xM    = t[np.argmax(s_avg)]
    yM    = np.max(s_avg)
    y0    = b_avg[1,0]
    
    # plot hemodynamic signals signals
    txt = 'beta = {:.2f}\nR² = {:.2f}, {}\n '. \
           format(beta[vertex], Rsq_run, Figures.pvalstr(p_run))
    # Note: For visualization purposes, we here apply "avg = [True, True]".
    axs[k,1].plot(t, y_avg[:,0], ':ok', markerfacecolor='k', markersize=8, linewidth=1, label='measured signal')
    axs[k,1].plot(t, s_avg[:,0], '-'+col, linewidth=2, label='predicted signal')
    for i in range(len(ons)):
        axs[k,1].plot(np.array([ons[i],ons[i]]), \
                      np.array([y_max+(1/20)*y_rng, y_max+(3/20)*y_rng]), '-k')
        axs[k,1].text(ons[i]+(1/2)*dur[i], y_max+(2/20)*y_rng, str(stim[i]), fontsize=18,
                      horizontalalignment='center', verticalalignment='center')
    axs[k,1].plot(np.array([ons[-1]+dur[-1],ons[-1]+dur[-1]]), \
                  np.array([y_max+(1/20)*y_rng, y_max+(3/20)*y_rng]), '-k')
    axs[k,1].plot(np.array([ons[0],ons[-1]+dur[-1]]), \
                  np.array([y_max+(1/20)*y_rng, y_max+(1/20)*y_rng]), '-k')
    if k == 0:
        axs[k,1].plot([xM,xM], [y0,yM], '-', color='gray', linewidth=2)
        axs[k,1].text(xM, (y0+yM)/2, 'beta = scaling factor', fontsize=18, \
                      horizontalalignment='right', verticalalignment='center', rotation=90)
        axs[k,1].legend(loc='lower right', fontsize=18)
    axs[k,1].axis([0, t_max, y_min-(1/20)*y_rng, y_max+(3/20)*y_rng])
    if k == len(verts)-1:
        axs[k,1].set_xlabel('within-cycle time [s]', fontsize=32)
    axs[k,1].set_ylabel('hemodynamic signal [%]', fontsize=32)
    if k == 0:
        axs[k,1].set_title('presented numerosity', fontsize=32)
    axs[k,1].tick_params(axis='both', labelsize=18)
    axs[k,1].text((1/20)*t_max, y_min-(1/20)*y_rng, txt, fontsize=18, \
                  horizontalalignment='left', verticalalignment='bottom')
    del y, y_reg, y_avg, z, s, s_avg, t, b_reg, b_avg, y_min, y_max, y_rng, txt
    
    
### Step 7b: create analogue of Figure 5a #####################################
# Explanation: This section follows "Figures.WP1_Fig4.plot_area_vs_mu".

# specify mu grid
Apv    = 1e4/v                  # area per vertex (here: 10 mm²)
d_mu   = 0.5                    # preferred numerosity bin width
mu_min = EMPRISE.mu_thr[0]
mu_max = EMPRISE.mu_thr[1]
mu_b   = np.arange(mu_min, mu_max+d_mu, d_mu)
mu_c   = np.arange(mu_min+d_mu/2, mu_max+d_mu/2, d_mu)

# preallocate results
area_s = np.zeros(mu_c.shape)
b_a    = np.zeros(2)

# go through numerosity bins
for k in range(mu_c.size):
    ind_k = np.logical_and(mu[ind]>mu_b[k], mu[ind]<mu_b[k+1])
    if np.sum(ind_k) > 0:
        area_s[k] = np.sum(ind_k)*Apv
    else:
        area_s[k] = np.nan

# calculate regression lines
ind_j  =~np.isnan(area_s)
n_a    = np.sum(ind_j)
r_a, p_a, b_a[0], b_a[1] = Figures.simplinreg(area_s[ind_j], mu_c[ind_j])
del ind_k, ind_j

# open figure
fig = plt.figure(figsize=(16,9))
ax  = fig.add_subplot(111)
col = 'darkblue'

# plot area vs. mu
if np.any(area_s):
    lab = 'area vs. mu (r = {:.2f}, {}, n = {})'. \
           format(r_a, Figures.pvalstr(p_a), n_a)
    ax.plot(mu_c, area_s, 'o', \
            color=col, markerfacecolor=col, markersize=10)
    ax.plot([mu_min,mu_max], np.array([mu_min,mu_max])*b_a[0]+b_a[1], '-', \
            color=col, label=lab)
ax.set_xlim(mu_min-d_mu, mu_max+d_mu)
ax.set_ylim(0, (11/10)*np.nanmax(area_s))
ax.legend(loc='upper right', fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel('preferred numerosity', fontsize=28)
ax.set_ylabel('cortical surface area [mm²]', fontsize=28)
fig.show()


### Step 7c: create analogue of Figure 5c #####################################
# Explanation: This section follows "Figures.WP1_Fig4.plot_fwhm_vs_mu".

# preallocate results
fwhm_m  = np.zeros(mu_c.shape)
fwhm_se = np.zeros(mu_c.shape)
b_f     = np.zeros(2)

# go through numerosity bins
for k in range(mu_c.size):
    ind_k = np.logical_and(mu[ind]>mu_b[k], mu[ind]<mu_b[k+1])
    if np.sum(ind_k) > 0:
        fwhm_m[k]  = np.mean(fwhm[ind][ind_k])
        fwhm_se[k] = np.std(fwhm[ind][ind_k])/math.sqrt(np.sum(ind_k))
    else:
        fwhm_m[k]  = np.nan
        fwhm_se[k] = np.nan

# calculate regression lines
ind_j  =~np.isnan(fwhm_m)
n_f    = np.sum(ind_j)
r_f, p_f, b_f[0], b_f[1] = Figures.simplinreg(fwhm_m[ind_j], mu_c[ind_j])
del ind_k, ind_j

# open figure
fig = plt.figure(figsize=(16,9))
ax  = fig.add_subplot(111)
col = 'darkblue'

# plot fwhm vs. mu
if np.any(fwhm_m):
    lab = 'fwhm vs. mu (r = {:.2f}, {}, n = {})'. \
           format(r_f, Figures.pvalstr(p_f), n_f)
    ax.plot(mu_c, fwhm_m, 'o', \
            color=col, markerfacecolor=col, markersize=10)
    ax.errorbar(mu_c, fwhm_m, yerr=fwhm_se, \
                fmt='none', ecolor=col, elinewidth=2)
    ax.plot([mu_min,mu_max], np.array([mu_min,mu_max])*b_f[0]+b_f[1], '-', \
            color=col, label=lab)
ax.set_xlim(mu_min-d_mu, mu_max+d_mu)
ax.set_ylim(0, (11/10)*np.nanmax(fwhm_m+fwhm_se))
ax.legend(loc='upper left', fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel('preferred numerosity', fontsize=28)
ax.set_ylabel('FWHM tuning width', fontsize=28)
fig.show()