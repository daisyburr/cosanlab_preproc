from __future__ import division

'''
Preprocessing Workflow Runner
=============================

'''

__all__ = [
    'workflow_builder'
]
__author__ = ["Eshin Jolly"]
__license__ = 'MIT'

import os
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import IdentityInterface
from bids.grabbids import BIDSLayout
from cosanlab_preproc.utils import get_subject_data, get_mni_template
from cosanlab_preproc.workflows import anatomical_wf, functional_wf, confounds_wf, normalization_wf
from cosanlab_preproc.interfaces import BIDSInput, BIDSOutput

def workflow_builder(
    subject_id,
    data_dir,
    output_dir,
    work_dir = None,
    session = None,
    segment = False,
    low_pass = None,
    high_pass = None,
    down_sample_precision = True,
    discorr = False,
    reference_vol = 'mean',
    smooth = 6,
    trim = 0,
    template = '3mm',
    extension='nii.gz',
    ants_threads = 12,
    name = 'cosan_complete_wf'):
    """
    Workflow builder flexibly connects sub-workflows together to construct a complete preprocessing pipeline. This meta-flow works by connecting several sub-workflows together and using input arguments when this function is called, specifically:

    BIDS input (node)
    Anatomical workflow (sub-wf)
    Functional workflow (sub-wf)
    Normalization workflow (sub-wf)
    Confounds workflow (sub-wf)
    BIDS output (node)

    This makes it possible to run any of these connections completely separately of each other provided their basic inputs are met (i.e. paths to files).

    Inputs:
        subject_id (str): subject folder name to preprocess
        data_dir (str): path to base BIDS formatted directory
        output_dir (str)L path to store preprocessed data
        work_dir (str): path to store intermediary files for each step; defaults to subject's own data folder under 'Preprocessing'
        session (int): what subject session to process; defaults to None assuming only 1 session exists
        segment (bool): whether to perform T1 tissue segmentation using FSL FAST
        low_pass (float): cut-off frequency for low pass filtering using nilearn; defaults to no filtering
        high_pass (float: cut-off frequency for high pass filtering using nilearn; defaults to no filtering)
        down_sample_precision (bool): save final preprocessed data as int16 to reduce file sizes; all operations are done in floats prior to this; defaults to True
        discorr (bool): whether to perform distortion correction using FSL TOPUP and APPLYTOPUP; default is True
        reference_vol (str): what volume to use as the reference for hmc; default is mean of the run
        smooth (int or list): smoothing kernel in mm to apply; can provide a list to create multiple smoothed outputs; default is 6mm
        trim (int): number of volumes (TRs) to trim at the beginning of each functional run; default is 0
        template (str): '1mm', '2mm' or '3mm' MNI template for normalization or path to another template; default is 3mm
        extension (str): file extension to search; default is nii.gz
        ants_threads (int): number of multi-proc threads ANTS should use; default is 12
        name (str): name to give this meta-workflow; 'cosan_complete_wf' is default

    Outputs:
        anat/
            extracted_brain (file): skull stripped brain in subject space
            normalized_brain (file): skull stripped brain in template space
            segmented_brain (file): T1 segmentation in subject space
            segmented_normalized_brain (file): T1 segmentation in template space
        distortions/
            dist_field (file): estimated EPI distortions from TOPUP
            bias_field (file): estimated T1 bias field from FAST
        func/
            bold_preproc (file): fully preprocessed time-series
            mean_bold_preproc (file): average fully preprocessed time-series image
        covs/
            covariates (file): csv file including expanded motion regressors, outlier spikes, average WM, average CSF time series
    """

    assert os.path.exists(data_dir), "Data directory not found!"
    if not os.path.exists(output_dir):
        os.make_dirs(output_dir)

    if 'sub' in subject_id:
        subject_id = subject_id[4:]

    complete_wf = Workflow(name=name)
    complete_wf.config['execution'] = {'crashfile_format':'txt'}

    #BIDS INPUT
    bids_input = Node(BIDSInput(),name='bids_input')
    bids_input.inputs.data_dir = data_dir
    bids_input.inputs.subject_id = subject_id

    #BIDS OUTPUT
    bids_output = Node(BIDSOutput(),name='bids_output')
    bids_output.inputs.base_directory=output_dir


    #ANATOMICAL WF
    anat_wf = anatomical_wf(template_image = template,
                            num_threads = ants_threads,
                            segment = segment)

    #FUNCTIONAL WF
    func_wf = functional_wf(discorr = discorr,
                            trim = trim,
                            reference_vol = reference_vol)

    #CONFOUNDS WF

    #NORMALIZATION WF

    #BIDS INPUT -> ANATOMICAL WF
    complete_wf.connect([
        (bids_input,anat_wf,[('t1w','inputnode.anat_image')])
    ])


    #BIDS INPUT -> FUNCTIONAL WF
    complete_wf.connect([
        (bids_input,func_wf,
        [('bold','inputnode.func_image'),
         ('fmap','inputnode.fmap')])
    ])

    #ANATOMICAL WF -> NORMALIZATION WF

    #FUNCTIONAL WF -> NORMALIZATION WF

    #FUNCTIONAL WF -> CONFOUNDS WF

    #ANATOMICAL WF -> BIDS OUTPUT

    #FUNCTIONAL WF -> BIDS OUTPUT

    #CONFOUNDS WF -> BIDS OUTPUT

    #NORMALIZATION WF -> BIDS OUTPUT

    #Run functional workflow with or without
        #distortion correction - should handle None
            #Look at Readsidecarjson in fmriprep for handling encoding directions
        #trimming - should handle None
        #smoothing - should convert to list and create iterable node or handle None
        #low_pass_filter - should create additional file not replace

    return complete_wf
