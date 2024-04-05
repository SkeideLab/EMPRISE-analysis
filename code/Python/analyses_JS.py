# -*- coding: utf-8 -*-
"""
Analyses of the EMPRISE data set
by Joram Soch <soch@cbs.mpg.de>
"""


# import modules
import os
import time
import shutil
import NumpRF
import EMPRISE
import numpy as np
import scipy as sp
# import pandas as pd
# import nibabel as nib
# import matplotlib.pyplot as plt


### checks for digits/spoken/congruent/incongruent, 11/03/2024 ################

# # check if logfiles can be read
# print()
# sub = EMPRISE.adults[0]
# for ses in ['visual', 'audio', 'digits', 'spoken', 'congruent', 'incongruent']:
#     sess = EMPRISE.Session(sub, ses)
#     ons_ev, dur_ev, stim_ev = sess.get_onsets()
#     ons_bl, dur_bl, stim_bl = EMPRISE.onsets_trials2blocks(ons_ev, dur_ev, stim_ev, 'closed')
#     print('-> Subject "{}", Session "{}": {} events, {} blocks.'. \
#           format(sub, ses, len(ons_ev[0]), len(ons_bl[0])))

# # check number of fMRI volumnes
# print()
# sub = EMPRISE.adults[0]
# for ses in ['visual', 'audio', 'digits', 'spoken', 'congruent', 'incongruent']:
#     sess = EMPRISE.Session(sub, ses)
#     Y = sess.load_surf_data(1, 'L', 'fsnative')
#     print('-> Subject "{}", Session "{}", Run {}: {} scans x {} voxels.'. \
#           format(sub, ses, 1, Y.shape[0], Y.shape[1]))


### NumpRF empirical analysis (digits, spoken), 11/03/2024 ####################

# define analyses
subs   = EMPRISE.adults
sess   = ['digits', 'spoken']
spaces = EMPRISE.spaces
model  = {'avg': [True, False], 'noise': 'iid', 'hrfs': 1}
ver    =  'V2'

# specify folder
targ_dir = EMPRISE.tool_dir + '../../../Analyses/'

# perform analyses
for sub in subs:
    for ses in sess:
        for space in spaces:
                
            # determine results filename
            mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])+'_'+ver+'_'+'new'
            # V2 = new parameter grid; new = re-preprocessed data (2024-01)
            subj_dir = EMPRISE.Model(sub, ses, mod_name, space).get_model_dir() + '/'
            filepath = 'sub-' + sub + '_ses-' + ses + '_model-' + mod_name + '_hemi-' + 'R' + '_space-' + space + '_'
            res_file = filepath + 'numprf.mat'
            para_map = filepath + 'mu.surf.gii'
            
            # if results file already exists, let the user know
            if os.path.isfile(subj_dir+res_file):
                
                # display message
                print('\n\n-> Subject "{}", Session "{}", Model "{}", Space "{}":'. \
                      format(sub, ses, mod_name, space))
                print('   - results file does already exist, model is not estimated!')
                if not os.path.isfile(targ_dir+para_map):
                    shutil.copy(subj_dir+para_map, targ_dir)
            
            # if results file does not yet exist, perform analysis
            else:
            
                # run numerosity model
                try:
                    mod = EMPRISE.Model(sub, ses, mod_name, space)
                    mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver)
                    shutil.copy(subj_dir+para_map, targ_dir)
                except FileNotFoundError:
                    continue


### EMPRISE classical GLM analysis (group), 11/03/2024 ########################

# # prepare analyses
# subs  = EMPRISE.adults
# sess  =['visual', 'audio']
# model = 'False_iid'
# ana   = 'False_iid_con_0002'

# # specify model
# y = 'con_0002.surf.gii'
# X = np.ones((len(subs),1))

# # define contrasts
# cons  = [{'type': 'F', 'c': np.array([[1]])}, \
#          {'type': 't', 'c': np.array([+1])},  \
#          {'type': 't', 'c': np.array([-1])}]
# alpha = 0.001

# # perform GLM analysis
# for ses in sess:
#     grp     = EMPRISE.Group('all', ses, model, ana, subs)
#     results = grp.run_GLM_analysis_group(y, X, cons)
#     mod_dir = EMPRISE.deri_out + 'pyspm' + '/sub-' + grp.sub + '/ses-' + grp.ses + '/model-' + grp.ana
#     for k in range(len(cons)):
#         fig   = grp.threshold_SPM(k+1, alpha)
#         p_str = '{:1.2e}'.format(alpha)
#         filename = mod_dir + '/spm{}_{:04d}'.format(cons[k]['type'].upper(), k+1) + '_p-' + p_str + '.png'
#         fig.savefig(filename, dpi=150, transparent=True)


### EMPRISE classical GLM analysis (fsaverage), 11/03/2024 ####################

# # prepare analyses
# subs  = EMPRISE.adults
# sess  =['visual', 'audio']
# space = 'fsaverage'
# model = 'False_iid'

# # define contrasts
# cons = [{'type': 'F', 'c': np.array([[1/5, 1/5, 1/5, 1/5, 1/5, -1]]).T},  \
#         {'type': 't', 'c': np.array([+1/5, +1/5, +1/5, +1/5, +1/5, -1])}, \
#         {'type': 't', 'c': np.array([-1/5, -1/5, -1/5, -1/5, -1/5, +1])}]

# # perform GLM analysis
# for sub in subs:
#     for ses in sess:
#         mod = EMPRISE.Model(sub, ses, model, space)
#         mod.run_GLM_analysis(False, 'iid', cons)


### EMPRISE classical GLM analysis (all subs), 04/03/2024 #####################

# # prepare analyses
# subs  = EMPRISE.adults
# sess  =['visual', 'audio']
# space = 'fsnative'
# model = 'False_iid'

# # define contrasts
# cons = [{'type': 'F', 'c': np.array([[1/5, 1/5, 1/5, 1/5, 1/5, -1]]).T},  \
#         {'type': 't', 'c': np.array([+1/5, +1/5, +1/5, +1/5, +1/5, -1])}, \
#         {'type': 't', 'c': np.array([-1/5, -1/5, -1/5, -1/5, -1/5, +1])}]
# thrs = [0.001, NumpRF.Rsq2pval(Rsq=0.2, n=145, p=4)]

# # perform GLM analysis
# for sub in subs:
#     for ses in sess:
#         mod = EMPRISE.Model(sub, ses, model, space)
#         mod.run_GLM_analysis(False, 'iid', cons)

# # perform contrasts
# for ses in sess:
#     for sub in subs:    
#         mod = EMPRISE.Model(sub, ses, model, space)
#         mod_dir = EMPRISE.deri_out + 'pyspm' + '/sub-' + mod.sub + '/ses-' + mod.ses + '/model-' + mod.model
#         for k in range(len(cons)):
#             for p in thrs:
#                 fig      = mod.threshold_SPM(k+1, p)
#                 thr_str  = '{:1.2e}'.format(p)
#                 filename = mod_dir + '/spm{}_{:04d}'.format(cons[k]['type'].upper(), k+1) + '_p-' + thr_str + '.png'
#                 fig.savefig(filename, dpi=150, transparent=True)


### EMPRISE classical GLM analysis (sub-001), 29/01/2024 ######################

# # prepare analyses
# sub   = '001'
# sess  =['visual', 'audio']
# space = 'fsnative'
# model = 'False_iid'

# # define contrasts
# cons = [{'type': 'F', 'c': np.array([[1/5, 1/5, 1/5, 1/5, 1/5, -1]]).T},  \
#         {'type': 't', 'c': np.array([+1/5, +1/5, +1/5, +1/5, +1/5, -1])}, \
#         {'type': 't', 'c': np.array([-1/5, -1/5, -1/5, -1/5, -1/5, +1])}]
# thrs = [0.001, NumpRF.Rsq2pval(Rsq=0.2, n=145, p=4)]

# # perform GLM analysis
# for ses in sess:
#     mod = EMPRISE.Model(sub, ses, model, space)
#     mod.run_GLM_analysis(False, 'iid', cons)

# # perform contrasts
# for ses in sess:
#     mod = EMPRISE.Model(sub, ses, model, space)
#     mod_dir = EMPRISE.deri_out + 'pyspm' + '/sub-' + mod.sub + '/ses-' + mod.ses + '/model-' + mod.model
#     for k in range(len(cons)):
#         for p in thrs:
#             fig      = mod.threshold_SPM(k+1, p)
#             thr_str  = '{:1.2e}'.format(p)
#             filename = mod_dir + '/spm{}_{:04d}'.format(cons[k]['type'].upper(), k+1) + '_p-' + thr_str + '.png'
#             fig.savefig(filename, dpi=150, transparent=True)


### NumpRF empirical re-analysis (True_False_iid_1_V2_new), 26/01/2024 ########

# # define analyses
# subs   = EMPRISE.adults
# sess   = ['visual', 'audio']
# spaces = EMPRISE.spaces
# model  = {'avg': [True, False], 'noise': 'iid', 'hrfs': 1}
# ver    =  'V2'

# # specify folder
# targ_dir = EMPRISE.tool_dir + '../../../Analyses/'

# # perform analyses
# for sub in subs:
#     for ses in sess:
#         for space in spaces:
                
#             # determine results filename
#             mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])+'_'+ver+'_'+'new'
#             subj_dir = EMPRISE.Model(sub, ses, mod_name, space).get_model_dir() + '/'
#             filepath = 'sub-' + sub + '_ses-' + ses + '_model-' + mod_name + '_hemi-' + 'R' + '_space-' + space + '_'
#             res_file = filepath + 'numprf.mat'
#             para_map = filepath + 'mu.surf.gii'
            
#             # if results file already exists, let the user know
#             if os.path.isfile(subj_dir+res_file):
                
#                 # display message
#                 print('\n\n-> Subject "{}", Session "{}", Model "{}", Space "{}":'. \
#                       format(sub, ses, mod_name, space))
#                 print('   - results file does already exist, model is not estimated!')
#                 if not os.path.isfile(targ_dir+para_map):
#                     shutil.copy(subj_dir+para_map, targ_dir)
            
#             # if results file does not yet exist, perform analysis
#             else:
            
#                 # run numerosity model
#                 try:
#                     mod = EMPRISE.Model(sub, ses, mod_name, space)
#                     mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver)
#                     shutil.copy(subj_dir+para_map, targ_dir)
#                 except FileNotFoundError:
#                     continue