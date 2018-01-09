"""Handy utilities"""

__all__ = ['get_resource_path','get_mni_template','get_n_slices','get_ta','get_slice_order','get_n_volumes','get_vox_dims','get_subject_data','split_text']
__author__ = ["Eshin Jolly", "Luke Chang"]
__license__ = "MIT"

from os.path import dirname, join, pardir, sep as pathsep
import nibabel as nib
import os
from bids.grabbids import BIDSLayout

def get_resource_path():
    """ Get path to nltools resource directory. """
    return join(dirname(__file__), 'resources') + pathsep

def get_mni_template(mm='2'):
    """ Get MNI template image in specified resolution. """
    if mm == '1':
        file_path = os.path.join(get_resource_path(),'MNI152_T1_1mm.nii.gz')
    elif mm == '2':
        file_path = os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz')
    elif mm == '3':
        file_path = os.path.join(get_resource_path(),'MNI152_T1_3mm.nii.gz')
    else:
        raise ValueError("Uknown MNI resolution provided!")
    return file_path

def get_ants_templates():
    imgs = [os.path.join(get_resource_path(),img) for img in [
        'OASIS_template.nii.gz',
        'OASIS_BrainCerebellumProbabilityMask.nii.gz',
        'OASIS_BrainCerebellumRegistrationMask.nii.gz']]
    return imgs

def get_ants_settings():
    settings_files = [os.path.join(get_resource_path(),f) for f in [
        'ANTS_normalization_settings.json',
        'ANTS_applyTransform_settings.json'
    ]]
    return settings_files

def get_n_slices(volume):
    """ Get number of volumes of image. """

    import nibabel as nib
    nii = nib.load(volume)
    return nii.get_shape()[2]

def get_ta(tr, n_slices):
    """ Get slice timing. """

    return tr - tr/float(n_slices)

def get_slice_order(volume):
    """ Get order of slices """

    import nibabel as nib
    nii = nib.load(volume)
    n_slices = nii.get_shape()[2]
    return range(1,n_slices+1)

def get_n_volumes(volume):
    """ Get number of volumes of image. """

    import nibabel as nib
    nii = nib.load(volume)
    if len(nib.shape)<4:
        return 1
    else:
        return nii.shape[-1]

def get_vox_dims(volume):
    """ Get voxel dimensions of image. """

    import nibabel as nib
    if isinstance(volume, list):
        volume = volume[0]
    nii = nib.load(volume)
    hdr = nii.get_header()
    voxdims = hdr.get_zooms()
    return [float(voxdims[0]), float(voxdims[1]), float(voxdims[2])]

def get_subject_data(data_dir, subject_id, session = None, extension='nii.gz'):
    """
    Complete subject query from BIDS dataset. Borrowed and modified from fmriprep/niworkflows.
    https://github.com/poldracklab/fmriprep/blob/master/fmriprep/utils/bids.py
    Called by workflow builder.
    """
    layout = BIDSLayout(data_dir)
    assert subject_id in layout.get_subjects(), "Subject not found in BIDS directory!"

    if session:
        queries = {
            'fmap': {'subject': subject_id,'session':session, 'modality': 'fmap',
                     'extensions': [extension]},
            'bold': {'subject': subject_id,'session':session, 'modality': 'func', 'type': 'bold',
                     'extensions': [extension]},
            'sbref': {'subject': subject_id,'session':session, 'modality': 'func', 'type': 'sbref',
                      'extensions': [extension]},
            't2w': {'subject': subject_id,'session':session, 'type': 'T2w',
                    'extensions': [extension]},
            't1w': {'subject': subject_id,'session':session, 'type': 'T1w',
                    'extensions': [extension]},
        }
    else:
        queries = {
            'fmap': {'subject': subject_id, 'modality': 'fmap',
                     'extensions': [extension]},
            'bold': {'subject': subject_id,'modality': 'func', 'type': 'bold',
                     'extensions': [extension]},
            'sbref': {'subject': subject_id,'modality': 'func', 'type': 'sbref',
                      'extensions': [extension]},
            't2w': {'subject': subject_id,'type': 'T2w',
                    'extensions': [extension]},
            't1w': {'subject': subject_id,'type': 'T1w',
                    'extensions': [extension]},
        }

    data = {modality: [x.filename for x in layout.get(**query)]
            for modality, query in queries.items()}
    if len(data['t1w']) == 0:
        raise IOError("Subject anatomical not found!")
    elif len(data['t1w']) > 1:
        raise NotImplementedError("Cannot current handle multiple subject anatomicals!")
    assert len(data['bold']) > 0, "Subject functional(s) not found!"

    return data

def split_text(fname):
    """
    Borrowed from fmriprep:
    https://github.com/poldracklab/fmriprep/blob/master/fmriprep/interfaces/bids.py
    """
    fname, ext = os.path.splitext(os.path.basename(fname))
    if ext == '.gz':
        fname, ext2 = os.path.splitext(fname)
        ext = ext2 + ext
    return fname, ext
