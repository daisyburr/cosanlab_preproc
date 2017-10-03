from __future__ import division

'''
Anatomical Preprocessing Workflows
==================================

'''

__all__ = [
    'anatomical_workflow'
]
__author__ = ['Eshin Jolly']
__license__ = 'MIT'

import os
from cosan_preproc.utils import get_ants_settings, get_mni_template, get_ants_templates
from nipype.interfaces.ants.segmentation import BrainExtraction, N4BiasFieldCorrection
from niypep.interfaces.ants.resampling import Registration, ApplyTransforms
from nipype.interfaces.fsl.preprocess import FAST
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import IdentityInterface

def anatomical_workflow(
    anat_image,
    template_image = get_mni_template('3'),
    ants_settings = get_ants_settings(),
    ants_extraction_template = get_ants_templates()[0],
    ants_brainprob_template = get_ants_templates()[1],
    ants_regprob_template = get_ants_templates()[2],
    num_threads = 12):

    """
    Workflow to preprocess anatomical images only.
    Steps include:
    1) Skull stripping (ANTS)
    2) Bias field correction and segmentation (FSL)
    3) Normalization estimation (ANTS)
    4) Normalization transform (ANTS)

    Inputs:
        anat_image (file): anatomical nifti image
        template_image (file): MNI template to normalize to 3mm default
        ants_settings (file): json file with ANTS settings or none to override
        ants_extraction_template (file): ants OASIS brain template_image
        ants_brainprob_template (file): ants brain probability mask
        ants_regprob_template (file): ants registration probability mask
        num_threads (int): number of threads to use for workflow

    Outputs:
        normalized_brain (file): skull stripped normalized brain
        extracted_brain (file): skull stripped brain
        partial_volume_files (list): partial volume files
        partial_volume_map (file): single image probability volume
        probability_maps (list): tissue probability files
        tissue_class_files(list): tissue class files
        tissue_class_map (file): single image segmented volume
        bias_field (file): estimated bias field image

    """

    anatomical_wf = Workflow(name='anatomical_wf')

    inputnode = Node(
        interface = IdentityInterface(
            fields = [
                'anat_image',
                'template_image']),
            name = 'inputspec')

    outputnode = Node(
        interface = IdentityInterface(
            fields = [
                'normalized_brain',
                'extracted_brain',
                'partial_volume_files',
                'partial_volume_map',
                'probability_maps',
                'tissue_class_files',
                'tissue_class_map',
                'bias_field']),
            name = 'outputspec')

    brain_extraction = Node(BrainExtraction(),name='brain_extraction')
    brain_extraction.inputs.dimension = 3
    brain_extraction.inputs.use_floatingpoint_precision = 1
    brain_extraction.inputs.num_threads = num_threads
    brain_extraction.inputs.brain_probability_mask = ants_brainprob_template
    brain_extraction.inputs.keep_temporary_files = 0
    brain_extraction.inputs.brain_template = ants_extraction_template
    brain_extraction.inputs.extraction_registration_mask = ants_regprob_template

    anatomical_wf.connect([
        (inputnode, brain_extraction, [('anat_image','anatomical_image')]),
        (brain_extraction, outputnode, [('BrainExtractionBrain','extracted_brain')])
    ])

    segmentation = Node(FAST(),name='segmentation')
    segmentation.inputs.number_classes = 3
    segmentation.inputs.output_biascorrected = True
    segmentation.inputs.output_biasfield = True
    segmentation.inputs.output_probability_maps = True

    anatomical_wf.connect([
        (brain_extraction, segmentation, [('BrainExtractionBrain','in_files')])
    ])


    return anatomical_wf
