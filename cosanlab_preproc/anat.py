from __future__ import division

'''
Anatomical Preprocessing Workflows
==================================

'''

__all__ = [
    'anatomical_wf'
]
__author__ = ['Eshin Jolly']
__license__ = 'MIT'

from cosanlab_preproc.utils import get_ants_settings, get_mni_template, get_ants_templates
from nipype.interfaces.ants.segmentation import BrainExtraction
from nipype.interfaces.ants.registration import Registration
from nipype.interfaces.ants.resampling import ApplyTransforms
from nipype.interfaces.fsl.preprocess import FAST
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.utility import IdentityInterface

def anatomical_wf(
    anat_image,
    template_image = get_mni_template('3'),
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
    2) Bias field correction and segmentation (FSL)
    3) MNI Normalization estimation and transform (ANTS)
    4) MNI Normalization transform of tissue segmentation (ANTS)

    Inputs:
        anat_image (file): anatomical nifti image
        template_image (file): brain-only MNI template to normalize to; 3mm default
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

    #Define Input Node
    inputnode = Node(
        interface = IdentityInterface(
            fields = [
                'anat_image',
                'template_image']),
            name = 'inputspec')

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

    #Define Brain Segmentation Node
    segmentation = Node(FAST(),name='segmentation')
    segmentation.inputs.number_classes = 3
    segmentation.inputs.output_biascorrected = True
    segmentation.inputs.output_biasfield = True
    segmentation.inputs.probability_maps = True

    anatomical_wf.connect([
        (brain_extraction, segmentation, [('BrainExtractionBrain','in_files')]),
        (segmentation, outputnode, [('partial_volume_files','partial_volume_files'),
         ('partial_volume_map','partial_volume_map'),
         ('probability_maps','probability_maps'),
         ('tissue_class_files','tissue_class_files'),
         ('tissue_class_map','tissue_class_map'),
         ('bias_field','bias_field')])
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

    #Define Segmented Tissue Normalization Node
    tissue_transform = Node(ApplyTransforms(
        from_file=ants_transform_settings,
        dimension = 3,
        ),name='tissue_transform')
    tissue_transform.inputs.num_threads = num_threads

    anatomical_wf.connect([
        (normalization,tissue_transform,[('composite_transform','transforms')]),
        (segmentation,tissue_transform, [('tissue_class_map','input_image')]),
        (inputnode,tissue_transform,[('template_image','reference_image')]),
        (tissue_transform,outputnode,[('output_image','normalized_segmentation')])
    ])

    return anatomical_wf
