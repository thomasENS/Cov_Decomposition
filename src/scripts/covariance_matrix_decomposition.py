# %% Imports & Constants
from scipy.optimize import minimize
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy
import os, nibabel, sys

## Fix __init__.py that does not work with my VSCode
sys.path.append(os.path.join(os.getcwd(), 'src'))

from utils.utils import (_check_folder)
from utils.maskers import (_extract_mask, _load_stream_mask, Streams)
from utils.regressors import (_compute_regressors)
from utils.minimization import (objective_function, constraints, matrix_r2_score)
from args.constants import (nConditions)
from args.args import (mDir, subj, regressors_mode)


rDir = os.path.join(mDir, 'derivatives', f'subj{subj:02d}')
_check_folder(rDir)

# %% Creating the Regressors Matrices : Either from super-ordinate categories (Num, Obj, Bg) or Load them from the LLS Regressor Matrices
Regressors_list, nModalities = _compute_regressors(regressors_mode, subj)
nRegressors = nModalities - 1 # We do not take into account the constant regressor for the LOCO

## Load beta estimates for the given subject
beta_estimates = nibabel.load(os.path.join(mDir, 'data', f'subj{subj:02d}', 'fmri_estimates', f'visual_conditions_across_all_concatenated_runs_z_scores.nii.gz'))
betas = beta_estimates.get_fdata()

Explained_Variance, Weights = [], []
for stream in Streams.keys():

    ## Extracting the beta estimates for the given roi
    mask = _load_stream_mask(stream, subj)
    betas_stream = _extract_mask(betas, mask) # shape (nConditions, nVoxelsPerStream)
    nVoxelsPerStream = betas_stream.shape[1] 
    
    ## Voxelwise centering across the conditions
    betas_stream -= betas_stream.mean(axis=0)

    ## Creating the target neural representation similarity matrix (nRSM)
    nRSM = np.zeros((nConditions, nConditions))
    for i in range(nConditions):
        for j in range(i+1, nConditions):
            corr = pearsonr(betas_stream[i], betas_stream[j])[0]
            nRSM[i,j], nRSM[j,i] = corr, corr

    ## Standardizing the nRSM by its frobenius norm
    nRSM /= np.linalg.norm(nRSM)

    ## Initial weight estimates using the full list of regressors
    w0 = np.ones(nModalities) / nModalities  # Initial guess
    result = minimize(objective_function, w0, args=(nRSM, Regressors_list), constraints=constraints)
    Weights.append(deepcopy(result.x))

    ## Prediction & Explained variance for the "full" problem
    nRSM_pred = sum(result.x[j] * Regressors_list[j] for j in range(nModalities))
    r2_full = matrix_r2_score(nRSM_pred, nRSM)

    ## Perform the minimization problem by leaving-out-one-regressor
    Unique_var = []
    for k in range(nRegressors):

        Constrained_Regressors_List = [Regressor for (i, Regressor) in enumerate(Regressors_list) if i != k]
        w0 = np.ones(nRegressors) / nRegressors  # Initial guess
        result = minimize(objective_function, w0, args=(nRSM, Constrained_Regressors_List), constraints=constraints)
        
        ## Prediction & unique variance for the left-out regressor
        nRSM_pred = sum(result.x[j] * Constrained_Regressors_List[j] for j in range(nRegressors))
        unique_var = r2_full - matrix_r2_score(nRSM_pred, nRSM)
        Unique_var.append(unique_var)

    ## compute the shared variance & then store it as the last value of the explained variance
    shared_var = r2_full - np.sum(Unique_var)
    Unique_var.append(shared_var)

    ## Store the explained variance per regressor for each stream
    Explained_Variance.append(deepcopy(Unique_var))


## Saving the nRSM weights decomposition upon regressor matrices across stream as well as the unique and share explained variance
np.save(os.path.join(rDir, f'nRSM_Decomposition_per_Stream_with_{regressors_mode.upper()}_Regressors.npy'), np.stack(Weights))
np.save(os.path.join(rDir, f'nRSM_Explained_Variance_per_Stream_with_{regressors_mode.upper()}_Regressors.npy'), np.stack(Explained_Variance))