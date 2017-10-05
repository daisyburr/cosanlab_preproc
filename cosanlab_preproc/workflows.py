from __future__ import division

'''
Preprocessing Workflows
==================================

'''

__all__ = [
    'anatomical_wf',
    'functional_wf',
    'confounds_wf',
    'normalization_wf'
]
__author__ = ['Eshin Jolly']
__license__ = 'MIT'

import os
from cosanlab_preproc.utils import get_ants_settings, get_mni_template, get_ants_templates
from nipype.interfaces.ants.segmentation import BrainExtraction
from nipype.interfaces.ants.registration import Registration
from nipype.interfaces.ants.resampling import ApplyTransforms
from nipype.interfaces.fsl.preprocess import FAST
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import IdentityInterface

def anatomical_wf(
    anat_image = None,
    template_image = '3mm',
    segment = True,
    ants_normalization_settings = get_ants_settings()[0],
    ants_transform_settings = get_ants_settings()[1],
    ants_brain_template = get_ants_templates()[0],
    ants_brainprob_mask = get_ants_templates()[1],
    ants_regprob_mask = get_ants_templates()[2],
    num_threads = 12):

    """
    Workflow to preprocess anatomical images only.
    Steps include:
    1) Skull stripping (ANTS)
    2) Bias field correction and segmentation (FSL) [optional]
    3) MNI Normalization estimation and transform (ANTS)
    4) MNI Normalization transform of tissue segmentation (ANTS) [optional]

    Inputs:
        anat_image (file): anatomical nifti image path or None if connected to another input node (default)
        template_image (file): brain-only MNI template to normalize to; 3mm default
        segment (bool): whether to perform segmentation using FSL FAST
        ants_settings (file): json file with ANTS settings or none to override
        ants_extraction_template (file): ants OASIS brain template_image
        ants_brainprob_template (file): ants brain probability mask
        ants_regprob_template (file): ants registration probability mask
        num_threads (int): number of threads to use for workflow

    Outputs:
        normalized_brain (file): skull stripped normalized brain
        extracted_brain (file): skull stripped brain
        normalized_segmentation (file): normalized tissue segmentation
        partial_volume_files (list): partial volume files
        partial_volume_map (file): single image probability volume
        probability_maps (list): tissue probability files
        tissue_class_files(list): tissue class files
        tissue_class_map (file): single image segmented volume
        bias_field (file): estimated bias field image
        normalization_transform_matrix (file): transform matrix to go from T1 to template space

    """

    anatomical_wf = Workflow(name='anatomical_wf')
    anatomical_wf.config['execution'] = {'crashfile_format':'txt'}

    #Define Input Node
    inputnode = Node(
        interface = IdentityInterface(
            fields = [
                'anat_image',
                'template_image']),
            name = 'inputspec')

    if anat_image:
        inputnode.inputs.anat_image = anat_image

    if template_image[0] in ['1', '2', '3']:
        template_image = get_mni_template(template_image[0])
    else:
        if not os.path.exists(template_image):
            raise IOError("Non MNI template path not found: {}".format(template_image))

    inputnode.inputs.template_image = template_image

    #Define Outpute Node
    outputnode = Node(
        interface = IdentityInterface(
            fields = [
                'normalized_brain',
                'extracted_brain',
                'normalized_segmentation',
                'partial_volume_files',
                'partial_volume_map',
                'probability_maps',
                'tissue_class_files',
                'tissue_class_map',
                'bias_field',
                'normalization_transform_matrix']),
            name = 'outputspec')

    #Define Brain Extraction Node
    brain_extraction = Node(BrainExtraction(),name='brain_extraction')
    brain_extraction.inputs.dimension = 3
    brain_extraction.inputs.use_floatingpoint_precision = 1
    brain_extraction.inputs.num_threads = num_threads
    brain_extraction.inputs.brain_probability_mask = ants_brainprob_mask
    brain_extraction.inputs.keep_temporary_files = 0
    brain_extraction.inputs.brain_template = ants_brain_template
    brain_extraction.inputs.extraction_registration_mask = ants_regprob_mask

    anatomical_wf.connect([
        (inputnode, brain_extraction, [('anat_image','anatomical_image')]),
        (brain_extraction, outputnode, [('BrainExtractionBrain','extracted_brain')])
    ])

    #Define Template Normalization Node
    normalization = Node(Registration(
        from_file=ants_normalization_settings,
        num_threads = num_threads),
        name='normalization')

    anatomical_wf.connect([
        (brain_extraction,normalization, [('BrainExtractionBrain','moving_image')]),
        (inputnode, normalization, [('template_image','fixed_image')]),
        (normalization,outputnode,
        [('composite_transform','normalization_transform_matrix'),
         ('warped_image','normalized_brain')])
    ])

    if segment:
        #Define Brain Segmentation Node
        segmentation = Node(FAST(),name='segmentation')
        segmentation.inputs.number_classes = 3
        segmentation.inputs.output_biascorrected = True
        segmentation.inputs.output_biasfield = True
        segmentation.inputs.probability_maps = True

        #Define Segmented Tissue Normalization Node
        tissue_transform = Node(ApplyTransforms(
            from_file=ants_transform_settings,
            dimension = 3,
            ),name='tissue_transform')
        tissue_transform.inputs.num_threads = num_threads

    anatomical_wf.connect([
        (brain_extraction, segmentation, [('BrainExtractionBrain','in_files')]),
        (segmentation, outputnode, [('partial_volume_files','partial_volume_files'),
         ('partial_volume_map','partial_volume_map'),
         ('probability_maps','probability_maps'),
         ('tissue_class_files','tissue_class_files'),
         ('tissue_class_map','tissue_class_map'),
         ('bias_field','bias_field')]),
        (normalization,tissue_transform,[('composite_transform','transforms')]),
        (segmentation,tissue_transform, [('tissue_class_map','input_image')]),
        (inputnode,tissue_transform,[('template_image','reference_image')]),
        (tissue_transform,outputnode,[('output_image','normalized_segmentation')])
    ])

    return anatomical_wf

def functional_wf(
    func_images = None,
    discorr = True,
    trim = 0,
    reference_vol = 'mean'):

    """
    Workflow to perform basic BOLD preprocesing only.
    Steps include:
    1) Distortion correction (FSL) [optional]
    2) Trimming (Nipy) [optional]
    3) Realignment/motion correction (FSL)
    4) Normalize motiona parameters to ensure same orientation (Nipype)
    5) Compute mean of motion corrected series
    6) Compute high contrast mean of motion corrected series for use in coregistration?
    5) Masking of epi (nilearn) ?

    Useful:
    EstimateReferenceImage https://github.com/poldracklab/niworkflows/blob/master/niworkflows/interfaces/registration.py
    https://github.com/poldracklab/fmriprep/blob/master/fmriprep/workflows/bold.py
    https://github.com/poldracklab/fmriprep/blob/master/fmriprep/workflows/util.py


    Inputs:
        func_images (list): nifti images or None if connected to another input node (default)
        discorr (bools): whether to perform distortion correction using FSL's TOPUP and APPLYTOPUS
        trim (int): how many volumes to trim at the beginning of the  series; default 0
        reference_vol (str): what volume to use as the reference of motion correction; default is the mean of the run; options include first, mean, median, last

    Outputs:
        bold_hmc (file): motion corrected time series
        hmc_pars (file): motion correction parameters
        mean_bold (file): mean epi

    """

    functional_wf = Workflow(name='functional_wf')
    functional_wf.config['execution'] = {'crashfile_format':'txt'}

    #Define Input Node
    inputnode = Node(
        interface = IdentityInterface(
            fields = [
                'func_image']),
            name = 'inputspec')
    inputnode.iterables = ('func_image',func_images)

    if trim > 0:
        #Define and connect trimming node
        pass

    #Define mcflirt node
    #Define normalize motion params
    #Compute high resolution mean epi
        #N4 bias field?
        #

    return functional_wf

def confounds_wf():
    pass

def normalization_wf():
    pass
