# MetaAI

## 1. Installation:
> Download the code at https://zenodo.org/records/15167559 and configure the running environment.

Software dependencies and operating systems: 

* Operating systems: Windows 11
* Environments: Refer to file **environment.yml**
* Software Version: 1.0
* Non-standard Hardware: None

Source code package: 
* WANGYS-truth/Current-Diffusion-Model-v1.0.zip

Model weights:
* vq-VAE: vqvae_fullsize_currents_mag_symetricEhance_120.pt
* Single-layer MetaAI: fullsize_cos_mag_tsqrt_smag_MVDT_605000.pt
* Two-layer MetaAI: doubleJmagSmag_attnTsqrtCos_MVDT_1530000.pt


## 2. Demo:
Test Data format:
> Mag(|S21|) at 19 points spans 2-20 GHz with interval 1 GHz.

Instructions to run on test data:
> Single-layer MetaAI: run **sample_TestS.py**
> Two-layer MetaAI: run **sample_TestSDoubleJ.py**

Expected output:
> For each generation of one target:
> * single-layer MetaAI generates one array with size (19, 2, 128, 128)
> * two-layer MetaAI generates two arrays with size (19, 2, 128, 128)
