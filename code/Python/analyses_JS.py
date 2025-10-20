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


### linear-model cvR² (visual, audio), 10/07/2024 #############################

# define analyses
subs   = EMPRISE.adults
sess   = ['visual', 'audio']
spaces = EMPRISE.spaces
model  = 'True_False_iid_1_V2-lin_new'

# calculate R-squared
for sub in subs:
    for ses in sess:
        for space in spaces:
            mod  = EMPRISE.Model(sub, ses, model, space)
            maps = mod.calculate_Rsq(folds=['all', 'odd', 'even', 'cv'])


### linear-model NumpRF analysis (visual, audio), 03/07/2024 ##################

# # import modules
# import EMPRISE

# # define analyses
# subs   = EMPRISE.adults
# # subs.remove('003')
# # subs.remove('009')
# sess   = ['visual', 'audio']
# spaces = EMPRISE.spaces
# model  = {'avg': [True, False], 'noise': 'iid', 'hrfs': 1}
# ver    =  'V2-lin'

# # perform analyses
# for sub in subs:
#     for ses in sess:
#         for space in spaces:
            
#             # specify model name
#             mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])+'_'+ver+'_'+'new'
#             # V2 = new parameter grid; lin = linear model; new = re-preprocessed data (2024-01)
            
#             # run numerosity model
#             mod = EMPRISE.Model(sub, ses, mod_name, space)
#             mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver, sh=False)
#             mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver, sh=True)


### linear-model NumpRF analysis (visual, audio), 01/07/2024 ##################

# # define analyses
# subs   = ['003', '009']
# sess   = ['visual', 'audio']
# spaces = EMPRISE.spaces
# model  = {'avg': [True, False], 'noise': 'iid', 'hrfs': 1}
# ver    =  'V2-lin'

# # specify folder
# targ_dir = EMPRISE.tool_dir + '../../../Analyses/'

# # perform analyses
# for sub in subs:
#     for ses in sess:
#         for space in spaces:
                
#             # determine results filename
#             mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])+'_'+ver+'_'+'new'
#             # V2 = new parameter grid; lin = linear model; new = re-preprocessed data (2024-01)
#             subj_dir = EMPRISE.Model(sub, ses, mod_name, space).get_model_dir() + '/'
#             filepath = 'sub-' + sub + '_ses-' + ses + '_model-' + mod_name + '_hemi-' + 'R' + '_space-' + space + '_runs-' + 'even' + '_'
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
#                     mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver, sh=False)
#                     mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver, sh=True)
#                     shutil.copy(subj_dir+para_map, targ_dir)
#                 except FileNotFoundError:
#                     continue


### extract numbers of vertices (pilot analyses), 24/06/2024 ##################

# # define analyses
# subs   = EMPRISE.adults
# sess   = ['visual', 'audio']
# hemis  = ['L', 'R']
# spaces = EMPRISE.spaces
# model  = 'True_False_iid_1_V2_new'

# # preallocate results
# v = np.zeros((len(subs),len(sess),len(hemis),len(spaces)), dtype=np.int32)

# # numbers of vertices
# for l, space in enumerate(spaces):
#     print('\n-> Numbers of vertices ({}):'.format(space))
#     for i, sub in enumerate(subs):
#         print('   - Subject {}: '.format(sub), end='')
#         for j, ses in enumerate(sess):
#             print('{}: '.format(ses), end='')
#             for k, hemi in enumerate(hemis):
#                 mod        = EMPRISE.Model(sub, ses, model, space)
#                 res_file   = mod.get_results_file(hemi)
#                 NpRF       = sp.io.loadmat(res_file)
#                 v[i,j,k,l] = np.squeeze(NpRF['mu_est']).size
#                 print(v[i,j,k,l], end=['/','; '][int(hemi=='R')])
#         print()

# # calculate averages
# print('\n-> Average number of vertices per hemisphere:')
# for l, space in enumerate(spaces):
#     for j, ses in enumerate(sess):
#         print('   - {}, {}: '.format(space, ses), end='')
#         print('{}.'.format(int(np.mean(v[:,j,:,l]))))


### cross-validated R-squared (visual, audio), 10/06/2024 #####################

# # define analyses
# subs   = EMPRISE.adults
# sess   = ['visual', 'audio']
# spaces = EMPRISE.spaces
# model  = 'True_False_iid_1_V2_new'

# # calculate R-squared
# for sub in subs:
#     for ses in sess:
#         for space in spaces:
#             mod  = EMPRISE.Model(sub, ses, model, space)
#             maps = mod.calculate_Rsq(folds=['all', 'odd', 'even', 'cv'])


### cross-validated R-squared (pilot analysis), 21/05/2024 ####################

# # define analyses
# sub_visual = '003'
# sub_audio  = '009'
# spaces     = EMPRISE.spaces
# model      = 'True_False_iid_1_V2_new'

# # calculate R-squared
# for space in spaces:
#     mod  = EMPRISE.Model(sub_visual, 'visual', model, space)
#     maps = mod.calculate_Rsq(folds=['all', 'odd', 'even', 'cv'])


### split-half NumpRF analysis (visual, audio), 24/05/2024 ####################

# # import modules
# import EMPRISE

# # define analyses
# subs   = EMPRISE.adults
# subs.remove('003')
# subs.remove('009')
# sess   = ['visual', 'audio']
# spaces = EMPRISE.spaces
# model  = {'avg': [True, False], 'noise': 'iid', 'hrfs': 1}
# ver    =  'V2'
# sh     =  True

# # perform analyses
# for sub in subs:
#     for ses in sess:
#         for space in spaces:
                
#             # specify model name
#             mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])+'_'+ver+'_'+'new'
#             # V2 = new parameter grid; new = re-preprocessed data (2024-01)
            
#             # run numerosity model
#             mod = EMPRISE.Model(sub, ses, mod_name, space)
#             mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver, sh=sh)


### split-half NumpRF analysis (visual, audio), 14/05/2024 ####################

# # define analyses
# subs   = ['003', '009']
# sess   = ['visual', 'audio']
# spaces = EMPRISE.spaces
# model  = {'avg': [True, False], 'noise': 'iid', 'hrfs': 1}
# ver    =  'V2'
# sh     =  True

# # specify folder
# targ_dir = EMPRISE.tool_dir + '../../../Analyses/'

# # perform analyses
# for sub in subs:
#     for ses in sess:
#         for space in spaces:
                
#             # determine results filename
#             mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])+'_'+ver+'_'+'new'
#             # V2 = new parameter grid; new = re-preprocessed data (2024-01)
#             subj_dir = EMPRISE.Model(sub, ses, mod_name, space).get_model_dir() + '/'
#             filepath = 'sub-' + sub + '_ses-' + ses + '_model-' + mod_name + '_hemi-' + 'R' + '_space-' + space + '_runs-' + 'even' + '_'
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
#                     mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver, sh=sh)
#                     shutil.copy(subj_dir+para_map, targ_dir)
#                 except FileNotFoundError:
#                     continue


### split-half NumpRF analysis (pilot analysis), 14/05/2024 ###################

# # define analysis
# sub   =  '001'
# ses   =  'visual'
# space =  'fsnative'
# model = {'avg': [True, False], 'noise': 'iid', 'hrfs': 1}
# ver   =  'V2'

# # perform analysis
# mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])+'_'+'test'
# mod      = EMPRISE.Model(sub, ses, mod_name, space)
# results1 = mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver, sh=False)
# results2 = mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver, sh=True)
# print(results1)
# print(results2)


### NumpRF empirical analysis (congruent, incongruent), 24/04/2024 ############

# # define analyses
# subs   = EMPRISE.adults
# subs.remove('007')
# subs.remove('009')
# sess   = ['congruent', 'incongruent']
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
#             # V2 = new parameter grid; new = re-preprocessed data (2024-01)
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


### sanity check for congruent/incongruent, 24/04/2024 ########################

# # specify subject/session
# sub  = EMPRISE.adults[0]
# ses  = 'incongruent'

# # load onsets and confounds
# sess = EMPRISE.Session(sub, ses)
# ons, dur, stim = sess.get_onsets()
# ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
# X_c  = sess.get_confounds(EMPRISE.covs)
# X_c  = EMPRISE.standardize_confounds(X_c)
# #X_c = np.zeros((EMPRISE.n,0,len(ons)))

# # simulate data
# ds   = NumpRF.DataSet(0, ons, dur, stim, EMPRISE.TR, X_c)
# mu   = np.array([1,3,5])
# fwhm = np.array([5,10,15])
# Y, S, X, B = ds.simulate(mu, fwhm)

# # visualize data
# fig = plt.figure(figsize=(32,18))
# ds.plot_signals_figure(fig, mu, fwhm, avg=[False, False], title='EMPRISE "'+ses+'"')
# fig.savefig('Figure_outputs/'+'NumpRF_simulate_'+ses+'.png', dpi=150)


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

# # define analyses
# subs   = EMPRISE.adults
# sess   = ['digits', 'spoken']
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
#             # V2 = new parameter grid; new = re-preprocessed data (2024-01)
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