import nibabel, os

## Useful Constants for Individual Subject's Brain Masking
hemispheres = ['left', 'right']

## Wang Atlas Regions-of-Interest (symetric for both hemispheres)
ROIs = [
    'FEF',   'IPS5',  'TO2',  'V3d',
    'hV4',   'LO1',   'V1d',  'V3v',
    'IPS0',  'LO2',   'V1v',  'VO1',
    'IPS1',  'PHC1',  'V2d',  'VO2',
    'IPS2',  'PHC2',  'V2v',
    'IPS3',  'SPL1',  'V3A',
    'IPS4',  'TO1',   'V3B'
]

Streams = {
    'V1':['V1d', 'V1v'],
    'V2':['V2d', 'V2v'],
    'V3':['V3d', 'V3v'],
    'IPS0':['IPS0'],
    'IPS12':['IPS1', 'IPS2'],
    'IPS345':['IPS3', 'IPS4', 'IPS5'],
    'hV4':['hV4'],
    'VO12':['VO1', 'VO2'],
    'PHC12':['PHC1', 'PHC2'],
    'V3AB':['V3A', 'V3B'],
    'LO12':['LO1', 'LO2'],
    'TO12':['TO1', 'TO2'],
}; nStreams = len(Streams.keys())

## Useful Methods for Individual Subject's Brain Masking
def _load_roi_mask(hemi, roi, subj=2):
    '''
        Fct that returns the np.array boolean mask associated with a given ROI from the Wang atlas.
    '''

    assert hemi in hemispheres, f'Given hemisphere is not valid, it should be in: {", ".join(hemispheres)}'
    assert roi in ROIs, f'Given roi is not part of the Wang atlas, valid input are: {", ".join(ROIs)}'


    sDir = f'/neurospin/unicog/protocols/IRMf/ObjectNumberComplexity_ChapalainEger_2022/Imaging_Data/mri_surface/sub{subj:02d}/mask_func'

    return nibabel.load(os.path.join(sDir, f'{hemi[0]}h.wang2015atlas_{roi}.nii')).get_fdata().astype(bool)

def _extract_mask(array, mask):
    '''
        Fct that returns the masked array using a given ROI.

        -Args :
            array: np.array of beta estimates, it should be of shape (nX, nY, nZ, nSamples)
            mask:  np.array (boolean) of shape (nX, nY, nZ)

        -Return :
        masked_array: np.array of shape (nSamples, nSizeROI)

    '''

    assert array.shape[:3] == mask.shape, f'Given array shape: {array.shape[:3]} does not with the ROI shape: {mask.shape} !'

    return array[mask, :].T

def _load_stream_mask(stream, subj=2):
    '''
        Fct that returns the np.array boolean mask associated with a given Stream of ROIs from the Wang atlas.
    '''

    assert stream in Streams.keys(), f'Given roi is not part of the Wang atlas, valid input are: {", ".join(Streams.keys())}'


    sDir = f'/neurospin/unicog/protocols/IRMf/ObjectNumberComplexity_ChapalainEger_2022/Imaging_Data/mri_surface/sub{subj:02d}/mask_func'

    mask = None
    for hemi in hemispheres:
        for roi in Streams[stream]:
            
            if mask is None:
                mask  = nibabel.load(os.path.join(sDir, f'{hemi[0]}h.wang2015atlas_{roi}.nii')).get_fdata()
            else:
                mask += nibabel.load(os.path.join(sDir, f'{hemi[0]}h.wang2015atlas_{roi}.nii')).get_fdata()

    return mask.astype(bool)