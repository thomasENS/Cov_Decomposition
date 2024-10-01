import numpy as np
import os

from args.constants import (nObjects, nBackgrounds, Numerosity, nNumerosity, nConditions)


def _compute_regressors(regressors_mode, subj=10):
    '''
        Fct that create the regressor similarity matrices (equivalent to rank 1 covariance matrices)
        
            The similarity matrices S are based on :
                S = L @ L.T with L = (Li - mean(L))_i for continous-valued regressor
                S = (delta(Li) * delta(Lj))_ij for categorical regressor
    '''

    if regressors_mode == 'num_obj_bg':
        
        Modalities = ['Numerosity', 'Objects', 'Backgrounds']; nModalities = len(Modalities)

        ## One regressor for numerosity (continuous-valued) and one for object or background (indicator function)
        Regressors = {Modality:np.zeros((nConditions, nConditions)) for Modality in Modalities}

        for idx_num_A in range(nNumerosity):
            for idx_obj_A in range(nObjects):
                for idx_bg_A in range(nBackgrounds):

                    ## Compute the idx of the current regressor : (Run, Numerosity, Object, Background) + Response
                    idx_reg_A = (idx_num_A * nObjects * nBackgrounds) + (idx_obj_A * nBackgrounds) + idx_bg_A
                    
                    for idx_num_B in range(nNumerosity):
                        for idx_obj_B in range(nObjects):
                            for idx_bg_B in range(nBackgrounds):

                                idx_reg_B = (idx_num_B * nObjects * nBackgrounds) + (idx_obj_B * nBackgrounds) + idx_bg_B

                                ## Regressor vectors based on the ordering (Run, Numerosity, Object, Background)      
                                Regressors['Objects'][idx_reg_A, idx_reg_B]     = 1 if idx_obj_A == idx_obj_B else 0
                                Regressors['Backgrounds'][idx_reg_A, idx_reg_B] = 1 if idx_bg_A == idx_bg_B else 0

    elif regressors_mode == 'lls':

        LLS = ['Mean_Luminance', 'Std_Luminance', 'Agg_Fourier_Mag', 'Energy_High_SF', 'Texture_Similarity', 'Image_Complexity']
        Modalities  = ['Numerosity'] + LLS; nModalities = len(Modalities)

        uDir = os.path.join(mDir, 'data', f'subj{subj}', 'statistics')

        ## One regressor for numerosity and each low-level statistics (continuous-valued)
        Regressors = {Modality:np.zeros((nConditions, nConditions)) for Modality in Modalities}

        for i in range(1, nModalities):

            ## Each (Numerosity, Object, Background) condition has been viewed 8 times
            ## across the 4 runs by the participant (2 Sp x 2 SzA x 2 versions)
            regressor_vector = np.load(os.path.join(uDir, f'{Modalities[i]}_per_Conditions.npy'))
            regressor_vector = np.array([np.mean(regressor_vector[idx_cond]) for idx_cond in range(nConditions)]).reshape(-1, 1)
            regressor_vector -= np.mean(regressor_vector)

            Regressors[Modalities[i]] = regressor_vector @ regressor_vector.T

    ## Creating the numerosity regressor "similarity matrix"
    numerosity_vector  = np.array([np.log(Numerosity[idx_num]) for idx_num in range(nNumerosity) for idx_obj in range(nObjects) for idx_bg in range(nBackgrounds)]).reshape(-1, 1)
    numerosity_vector -= np.mean(numerosity_vector)
    Regressors['Numerosity'] = numerosity_vector @ numerosity_vector.T

    ## Adding the Constant Regressor
    Modalities += ['Constant']; nModalities = len(Modalities)
    Regressors['Constant'] = np.eye(nConditions)

    ## Divide by the frobenius norm to have regressors spanning the same scale
    for Modality in Modalities:
        Regressors[Modality] /= np.linalg.norm(Regressors[Modality])

    Regressors_list = list(Regressors.values())

    return Regressors_list, nModalities