"""Handy utilities"""

__all__ = ['get_resource_path','get_mni_template','get_n_slices','get_ta','get_slice_order','get_n_volumes','get_vox_dims']
__author__ = ["Luke Chang"]
__license__ = "MIT"

from os.path import dirname, join, pardir, sep as pathsep
import nibabel as nib
import os

def get_resource_path():
    """ Get path to nltools resource directory. """
    return join(dirname(__file__), 'resources') + pathsep

def get_mni_template(mm='2'):
    """ Get MNI template image in specified resolution. """
    if mm == '1':
        file_path =  nib.load(os.path.join(get_resource_path(),'MNI152_T1_1mm.nii.gz'))
    elif mm == '2':
        file_path =  nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz'))
    elif mm == '3':
        file_path =  nib.load(os.path.join(get_resource_path(),'MNI152_T1_3mm.nii.gz'))
    else:
        raise ValueError("Uknown MNI resolution provided!")
    return file_path

def get_ants_templates():
    imgs = [nib.load(os.path.join(get_resource_path(),img)) for img in [
        'OASIS_BrainCerebellumExtractionMask.nii.gz',
        'OASIS_BrainCerebellumProbabilityMask.nii.gz',
        'OASIS_BrainCerebellumRegistrationMask.nii.gz']]
    return imgs

def get_ants_settings():
    return


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
