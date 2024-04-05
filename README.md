# EMPRISE-analysis

**EMPRISE - EMergence of PRecISE numerosity representations in the human brain**

This code belongs to the [EMPRISE project](https://docs.google.com/document/d/1NplHdKIxAtiP9eXvE_wwQogRfu9pO_3gWW6BLUdUZak/edit) within the [Skeide Lab](https://www.skeidelab.com/) at the [Max Planck Institute for Human Cognitive and Brain Sciences](https://www.cbs.mpg.de/en) (MPI CBS) in Leipzig, Germany and allows for population receptive field (pRF) analysis of functional MRI data collected during experiments with visual and auditory sensory numerosity (conditions "visual" and "audio") as well as visual and auditory symbolic numbers (conditions "digits" and "spoken").

Results on visual and auditory numerosity population receptive field (NumpRF) organization in adult subjects are reported in a paper entitled *"Population coding for visual and auditory quantity in human numerotopic maps"* which is currently in peer-review.

* Paper: *submitted, currently under review*
* Preprint: https://github.com/SkeideLab/preprints/blob/main/EMPRISE/EMPRISE_WP1_Manuscript.pdf
* Supplement: https://github.com/SkeideLab/preprints/blob/main/EMPRISE/EMPRISE_WP1_Supplement.pdf
* Code: https://github.com/SkeideLab/EMPRISE-analysis/
* Data: TBA

PsychoPy code for stimulus presentation can be found in `/experiment/`. Functional MRI acquisition parameters can be found in `/documentation/`. The following descriptions mainly address the NumpRF estimation pipeline contained in `/code/`.


## Installation

### System requirements

This code requires the following software to be installed:

* [Python 3.10](https://www.python.org/downloads/release/python-3100/)
* numpy 1.24.3, scipy 1.10.1, pandas 2.1.1, statsmodels 0.14.0
* matplotlib 3.7.1, surfplot 0.2.0, nibabel 5.1.0
* [AFNI](https://github.com/afni/afni) 23.2.03 (only if using `EMPRISE.threshold_AFNI_cluster`)

This code has been tested with, but is not dependent on the following software:

* [Microsoft Windows 10.0.19045](https://learn.microsoft.com/en-us/windows/release-health/release-information)
* [Linux 5.10.0-28-amd64](https://packages.debian.org/bullseye/linux-image-5.10.0-28-amd64)
* [Spyder 5.4.3](https://github.com/spyder-ide/spyder/releases/tag/v5.4.3)

No non-standard hardware is required to run this code.

### Installation guide

To use this software, install Python, Version 3.10 or higher. Then, install the required Python packages in the specified (or their current) version number using `pip install <package-name>` or `conda install <package-name>` (e.g. if using [Anaconda](https://www.anaconda.com/)). If you're using the function `threshold_AFNI_cluster` from the `EMPRISE` module (unlikely), then [AFNI](https://github.com/afni/afni) has to be installed and available on your Linux path.

The typical install time on a "normal" desktop computer will be around 10 minutes.

### Software demo

To access the demonstration, open `Demo.py` in Python and run this script. This program contains a little demonstration of the numerosity population receptive field (NumpRF) estimation pipeline and consists of the following steps: **1.** specification of simulation settings; **2.** loading of onsets and confounds; **3.** simulation of voxel-wise data; **4.** analysis of voxel-wise data; **5.** processing of simulation results; **6.** visualization of simulation results; **7.** recreation of figures analoguos to manuscript. It contains extensive comments explaining each step.

The demo run time on a "normal" desktop computer will be less than 5 minutes.

Expected output of the demonstration can be found in `Demo.pdf`.

### Instructions for use

The demo uses all four major Python modules from this repository: `PySPMs`, `NumpRF`, `EMPRISE` and `Figures`. Details on how to use those modules can be found below. When the data set is released for public use, figures from the Manuscript and Supplementary Material can be reproduced using code in `Figures.py`.


## Python pipeline

### Overview

The sub-folder `/code/Python/` contains the main Python code for numerosity population receptive field modelling (NumpRF). The Python pipeline consists of four modules:

* `PySPMs.py`: routines for performing [SPM](https://www.fil.ion.ucl.ac.uk/spm/)-style operations and analyses
* `NumpRF.py`: routines for numerosity population receptive field modelling
* `EMPRISE.py`: routines for dealing with EMPRISE functional MRI data
* `Figures.py`: routines for generating Figure shown in the EMPRISE paper

When working with Python, open `/code/Python/EMPRISE.py`, edit the study directory ([line 73](https://github.com/SkeideLab/EMPRISE/blob/5139987bb2893bcb4956abbe40dd971f3dd6e18c/code/Python/EMPRISE.py#L73)) and the tools directory ([line 77](https://github.com/SkeideLab/EMPRISE/blob/5139987bb2893bcb4956abbe40dd971f3dd6e18c/code/Python/EMPRISE.py#L77)) and run `import EMPRISE`.

### NumpRF

`NumpRF` is a module with tools for general mathematical and statistical operations related to (numerosity) receptive field estimation. It contains:

* some basic functions, such as `f_log` for a logarithmic numerosity tuning function, `lin2log` for conversion of tuning parameters from linear to logarithmic stimulus space and `Rsqtest` for statistical significance testing of the coefficient of determination ("R-squared");
* more specialized functions, such as `neuronal_signals` for implementing a neuronal model for numerosity processing and `hemodynamic_signals` for transforming neural to hemodynamic signals, assuming some hemodynamic response function (HRF);
* a class `DataSet`, initialized with measured fMRI data, experimental timing and nuisance variables, with methods for data simulation (`simulate`), parameter estimation (`estimate_MLE`) and plotting (`plot_signals_axis|figure`).

To use the `NumpRF` module, e.g. run the following code:

```python
# load modules
import NumpRF
import EMPRISE

# initialize a dataset with data matrix, onsets/durations and confound variables
ds = NumpRF.DataSet(Y, ons, dur, stim, EMPRISE.TR, X_c)
# Y    - an n x v x r array of run-wise (r) and voxel-wise (v) fMRI time series (n)
# ons  - a list with r elements where each entry is a vector of onset times
# dur  - a list with r elements where each entry is a vector of durations
# stim - a list with r elements where each entry is a vector of stimuli (= presented numerosity)
# X_c  - an n x c x r array of run-wise (r) confound variable (c) time series (n)
# for details how to obtain empirical Y, ons, dur, stim, X_c, see next section

# estimate numerosity tuning parameters (mu, fwhm, beta) from data set
mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =
    ds.estimate_MLE(avg=[True, False], corr='iid', order=1)
k_est, k_null, k_const =    # number of free parameters in model
    ds.free_parameters(avg=[True, False], corr='iid', order=1)

# assess statistical significance of tuning model fit
AIC_est = -2*MLL_est  + 2*k_est
AIC_null= -2*MLL_null + 2*k_null
Rsq_est = MLL2Rsq(MLL_est, MLL_null, n=EMPRISE.n)
p_val   = Rsq2pval(Rsq_est, n=EMPRISE.n, p=4)
```

For more information, see function help texts inside the module source code.

### EMPRISE

`EMPRISE` is a module with tools for loading, processing, modelling and saving data from the EMPRISE paradigm. It contains:

* some basic functions, such as `average_signals` for averaging data over sessions and/or cycles, `standardize_signals|confounds` for mean-centering and variance-scaling of measured data or confound variables;
* a number of constants describing the EMPRISE paradigm, e.g. session, run, space and mesh identifiers (`sess|runs|spaces|meshs`), repetition time (`TR`), microtime resolution/onset (`mtr|mto`), number of scans per run (`n`) etc.;
* a class `Session`, initialized via a subject and a session ID, with methods for loading data (`load_(surf_)data_all`) or extracting experimental timing information (`get_onsets`) and confound variables (`get_confounds`);
* a class `Model`, initialized via subject, session, space ID and model name, with methods for estimating NumpRF models (`analyze_numerosity`), thresholding parameter maps (`threshold_maps`) and surface-based clustering (`threshold_AFNI_cluster`);
* a class `Group`, initialized via session and model name as well as subject IDs, with methods for estimating group-level GLMs (`run_GLM_analysis_group`) and thresholding statistical parametric maps (`threshold_SPM`).

To use the `EMPRISE` module, e.g. run the following code:

```python
# load modules
import EMPRISE

# initialize session object with subject and session ID
subj_id = '001'
sess_id = 'visual'
sess = EMPRISE.Session(subj_id, sess_id)

# load fMRI data from this subject/session, left hemisphere
Y = sess.load_surf_data_all('L', 'fsnative')

# extract onsets from this session, convert trials to blocks
ons, dur, stim = sess.get_onsets()
ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')

# load and standardize confound variables using standard confounds
X_c = sess.get_confounds(EMPRISE.covs)
X_c = EMPRISE.standardize_confounds(X_c)

# initialize model object with session/session/model name
mod = EMPRISE.Model(subj_id, sess_id, mod_name='default', space_id='fsnative')

# estimate numerosity tuning parameters for subject/session/model
results = mod.analyze_numerosity(avg=[True, False], corr='iid', order=1, ver='V2')

# threshold tuning parameter maps according to selected criteria
crit = 'Rsqmb,0.2'    # Rsq>0.2 ("Rsq"), 1<mu<5 ("m"), beta>0 ("b")
maps = mod.threshold_maps(crit)

# perform AFNI surface clustering and return supra-threshold vertices and triangles
mesh = 'pial'         # perform surface clustering on pial mesh
verts, trias = mod.threshold_AFNI_cluster(crit, mesh)
```

### Summary

So, in order to analyze the data from all subjects and all sessions, one could e.g. run the following code:

```python
# load modules
import EMPRISE

# define analysis
subs   = EMPRISE.adults + EMPRISE.childs
sess   = ['visual', 'audio', 'digits', 'spoken']
spaces = EMPRISE.spaces    # "fsnative" & "fsaverage"
model  = {'avg': [True, False], 'noise': 'iid', 'hrfs': 1}

# for all subjects, sessions and spaces
for sub in subs:
    for ses in sess:
        for space in spaces:

            # initialize and estimate NumpRF model
            mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])
            mod      = EMPRISE.Model(sub, ses, mod_name, space)
            mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver='V2')
```

For more information, see function help texts inside the module source code.

### Figures

`Figures` is a module with tools for generating Figures for the EMPRISE paper, once models have been estimated. It contains:

* one function for each Table (`WP1_Tab0`, `WP1_Tab1`, `WP1_Tab2`);
* one function for each Figure (`WP1_Fig1`, `WP1_Fig2`, `WP1_Fig3`, `WP1_Fig4`);
* one function for generating illustration figures for the NumpRF approach (`Talk_Figs`).

To use the `Figures` module, adapt the variable `Figures` in the if-name-main section of the module:

```python
# select Figures
Figures = ['1', 'S1', 'S2']

# available Figures
# Figures = ['1', '2A', '2B', '2C', '2Bp', \
#            '3', '3A', '3B', '3C', '3D', '3E', \
#            '4', '4A', '4B', '4C', '4D', '4E', \
#            'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', \
#            'T0', 'T1', 'T2', 'A1', 'A2']
```

This variable must be a list of strings and specifies the figures that you want to plot. By calling the above-mentioned functions with input arguments, it also allows to plot supplementary figures (e.g. `WP1_Fig4('S5')`). For more information, see the if-name-main section of the module.


## Further resources

### LaTeX technical report

The sub-folder `/code/LaTeX/` contains LaTeX source files for a technical report that is intended to be a complete mathematical/statistical description of the currently implemented Python pipeline for numerosity receptive field modelling. The [most recent version](https://github.com/SkeideLab/EMPRISE/blob/JoramSoch/code/LaTeX/NumpRF.pdf) is `/code/LaTeX/NumpRF.pdf`, [previous versions](https://github.com/SkeideLab/EMPRISE/tree/JoramSoch/code/LaTeX/Version_history) can be found in `/code/LaTeX/Version_history/`.

### Shell scripts for AFNI

The sub-folder `/code/Shell/` contains Shell scripts that perform AFNI operations which are called from the Python pipeline. More precisely, the AFNI functions `SurfClust` and `ConvertDset` are used for surface clustering of thresholded tuning parameter map obtained from numerosity pRF modelling. This routine exists for FreeSurfer spaces [fsnative](https://github.com/SkeideLab/EMPRISE/blob/JoramSoch/code/Shell/cluster_surface.sh) and [fsaverage](https://github.com/SkeideLab/EMPRISE/blob/JoramSoch/code/Shell/cluster_surface_fsa.sh) and is called from the [EMPRISE](https://github.com/SkeideLab/EMPRISE/blob/JoramSoch/code/Python/EMPRISE.py) module function `threshold_AFNI_cluster`. For more information, see `cluster_surface.sh`:

```bash
# AFNI surface clustering for NumpRF tuning parameter maps
# AFNI <full_path_to_this_script> <sub> <ses> <model> <img> <anat>
# 
#     <full_path_to_this_script>
#             = /data/hu_soch/ownCloud/MPI/EMPRISE/tools
#               /EMPRISE/code/Shell/cluster_surface.sh
#     <sub>   - subject ID (e.g. "001")
#     <ses>   - session ID (e.g. "visual")
#     <model> - model name (e.g. "True_False_iid_1")
#     <img>   - image suffix (e.g. "space-fsnative_mu_thr-Rsq")
#     <anat>  - anat prefix in fMRIprep derivatives folder (e.g.
#               "ses-visual/anat/sub-001_ses-visual_acq-mprageised_")
```

### MATLAB legacy scripts

The sub-folder `/code/MATLAB/` contains MATLAB code that was used in an earlier attempt to employ [BayespRF](https://github.com/pzeidman/BayespRF), an SPM toolbox for Bayesian population receptive field modelling ([Zeidman et al., 2018](https://doi.org/10.1016/j.neuroimage.2017.09.008)). When working with MATLAB, open `/code/MATLAB/project_directories.m`, edit the study directory ([line 10](https://github.com/SkeideLab/EMPRISE/blob/JoramSoch/code/MATLAB/project_directories.m#L10)) and the tools directory ([line 14](https://github.com/SkeideLab/EMPRISE/blob/JoramSoch/code/MATLAB/project_directories.m#L14)) and run the script.

The MATLAB subfolder of the repository contains the following functions:

* `create_onset_files.m`: saves names, onsets and durations for first-level fMRI analysis in SPM format
* `create_mult_regs.m`: extracts confound variables for first-level fMRI analysis in SPM format 
* `create_stats_batch.m`: creates an SPM batch editor job for first-level fMRI analysis
* `BpRF_run_first_level.m`: extracts time series and prepares BayespRF analysis
* `BpRF_run_run_pRF_analysis.m`: performs numerosity pRF analysis using BayespRF
* `BpRF_run_run_pRF_simulation.m`: simulates numerosity pRF analysis using BayespRF
* `spm_prf_analyse_JS.m`: a modification of `spm_prf_analyse.m` for including covariates
* `spm_prf_fcn_numerosity.m`: the template pRF response function for numerosity analysis

To estimate a first-level GLM in SPM, run:

```matlab
create_onset_files(subj_id, sess_id, 0);
create_mult_regs(subj_id, sess_id, 0);
create_stats_batch(subj_id, sess_id, 'pRF', true);
```

To run voxel-wise BayespRF analysis, run:

```matlab
BpRF_run_first_level(subj_id, sess_id, 'pRF');
BpRF_run_pRF_analysis(subj_id, session, 'pRF', []);
```

As model estimation using BayespRF took very long (practically infeasible for whole-brain pRF modelling) and gave inaccurate results in simulation analysis, the BayespRF toolbox is not currently used for the present project.
