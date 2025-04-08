# MetaAI

## 1. Introduction & Installation:
We introduce a video diffusion transformer to design metasurfaces with a given Eletromagnetic response via generating current distributions at different frequencies.
The code is developed based on several works, including [vq-VAE](https://github.com/rosinality/vq-vae-2-pytorch), [VDT](https://github.com/RERV/VDT), and [videoMetamaterials](https://github.com/jhbastek/VideoMetamaterials)
To use the pretained model generating currents distributions from a EM response, start by cloning this repository via
```
git clone https://github.com/WANGYS-truth/Current-Diffusion-Model
```
Next, download the data and models at https://zenodo.org/records/15167559 and put them into the work space. We provide the pretrained model weights of both one layer and two layers. The generating currents distributions will be stored in `samples` folder and intermediate results of diffusion process will be stored in `sampleStep` folder.
```
.
├── doubleJmagSmag_attnTsqrtCos_MVDT_1530000.pt(model weight of two-layers)
├── fullsize_cos_mag_tsqrt_smag_MVDT_605000.pt(model weight of one-layer)
├── vqvae_fullsize_currents_mag_symetricEhance_120.pt(model weight of vq-VAE)
├── samples
│   ├── *nameOfModel*
└── sampleStep
    └── *samplingStep*
```

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
