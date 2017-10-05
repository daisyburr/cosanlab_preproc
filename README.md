# Cosanlab Preprocessing Tools

[Nipype](http://nipype.readthedocs.io/en/latest/) based preprocessing tools used in the [Cosanlab](http://cosanlab.com/).  

These tools are partially inspired by the Poldrack Lab's [niworkflows](https://github.com/poldracklab/niworkflows) but are not designed to be as encompassing as   [fMRIprep](http://fmriprep.readthedocs.io/en/stable/).  

This package is also pre-installed within our [Docker based analysis container](https://github.com/cosanlab/cosanToolsDocker) and HPC friendly [Singularity analyis container](https://github.com/cosanlab/cosanToolsSingularity).

Data is expected to be the [BIDS](http://bids.neuroimaging.io/) standard format. At minimum a user simply needs to provide:
- the root BIDS directory
- an output directory to save results
- the id of a subject to process

##Simple usage  

###Performs  
- Brain extraction, registration, normalization to MNI152 3mm (ANTS)
- HMC (FSL)
- Smoothing at 6mm
- Confound generation plots and covariates
```
from cosanlab_preproc.runner import workflow_builder
wf = workflow_builder(
    subject_id = 'my_subject',
    data_dir = '/my_data',
    output_dir = '/preprocessed_data',
    )
wf.run()
```
##More customized usage  

###Performs  
- Brain extraction, registration, normalization to MNI152 *2mm* (ANTS)
- Segmentation (WM, CSF, GM also normalized to MNI152) (FSL)
- Distortion correction (FSL)
- Trimming 10 early volumes
- HMC to first volume of run (FSL)
- High-pass filtering
- Smoothing at 4mm

```
from cosanlab_preproc.runner import workflow_builder
wf = workflow_builder(
    subject_id = 'my_subject',
    data_dir = '/my_data',
    output_dir = '/preprocessed_data',
    segment = True,
    trim = 10,
    template = '2mm',
    high_pass = 128,
    discorr = True,
    smooth = 4,
    reference_vol = 'first',
    )
wf.run()
```
