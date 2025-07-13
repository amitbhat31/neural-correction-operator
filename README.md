# Neural Correction Operator

This is the implementation for "Neural Correction Operator: A Reliable and Fast Approach for Electrical Impedance Tomography". While the code in this repository is focused on the datasets in the paper, it can be adapted for use with other datasets.

## Workflow Overview
The neural correction operator framework for the EIT inverse problem is performed in two main steps:
1. **Offline data generation**: Generate the ground truth medium and use L-BFGS with a finite number of iterations to obtain a rough reconstruction of the original medium.
2. **Neural correction**: Use a deep learning method to refine the L-BFGS reconstruction into the true medium.

## Installation
The relevant Python packages for this project can be installed via pip and are:
- numpy, scipy, matplotlib
- torch, torchvision (we recommend installing with CUDA support)
- numba, pypardiso
- hydra-core, omegaconf (for using the config .yaml files)
- click


## Offline data generation
The ```generate_circle_estimates.py``` and ```generate_SL_estimates.py``` in the src/data folder generate the training datasets for the Four Circles and Shepp-Logan datasets as described in the paper. We provide the mesh_125_h05.mat file for generating the finite element mesh. The data in the paper can be generated via

```python generate_circle_estimates.py --img-size 64 --num-samples 5000 --noise 0.0 --num-iters 350 --data-root /neural-correction-operator/data/ --mesh-file mesh_128_h05.mat```

```python generate_SL_estimates.py --img-size 64 --num-samples 5000 --noise 0.0 --num-iters 150 --original-size 256 --pad-size 4 --data-root /neural-correction-operator/data/ --mesh-file mesh_128_h05.mat```

## Neural correction 

We provide implementations for both ResNet and conditional DDPM models as our correction models. After generating the data, it suffices to train and sample via
```python train_resnet.py```
and 
```python sample_resnet.py```

and running the analogous files for DDPM. Parameters can be adjusted in the config .yaml files, and we provide config files corresponding to the experiments run in the paper.



