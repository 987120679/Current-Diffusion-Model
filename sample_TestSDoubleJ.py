# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models_attn import VDT_models
# from models_strAttn import VDT_models
# from models_fullAttn import VDT_models
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms, utils
from vqvae import VQVAE
from dataset import TestSDataset
from einops import rearrange, reduce, repeat
import h5py
import numpy as np


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch.distributed as dist
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math


def create_npz_from_sample_folder(sample_dir, channels, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        for cc in range(channels):
            sample_pil = Image.open(f"{sample_dir}/{i:06d}_channel{cc:06d}.png")
            print(sample_pil.shape)
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
    samples = np.stack(samples)
    print(samples.shape)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    num = args.num_generation
    for i in range(0,num):
        args = parser.parse_args()
        parser.set_defaults(global_seed=i)
        args = parser.parse_args()
        sample(args)


def sample(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    # dist.init_process_group("nccl")
    # initialize distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    # dist.init_process_group(backend='gloo', init_method='env://', rank = 0, world_size = 1)
    dist.init_process_group("gloo", rank=0, world_size=1)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 4
    model = VDT_models[args.model](
        input_size=latent_size,
        in_channels = args.latent_channels*2,
        num_frames = args.num_frames
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    # model.load_state_dict(torch.load(ckpt_path))
    model.eval()  # important!
    diffusion = create_diffusion(args.num_sampling_steps,noise_schedule = args.noise_schedule)
    vae = VQVAE(in_channel=args.in_channels,channel=128,n_res_block=4,n_res_channel=64,embed_dim=args.latent_channels,n_embed=2048,decay=0.99)
    vae.load_state_dict(torch.load(args.vae_name))
    vae=vae.to(device)
    vae.eval()
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0
    using_cfg = False

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    special_discription = args.special_discription
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-" \
                  f"cfg-{args.cfg_scale}-samplingStep-{args.num_sampling_steps}-{special_discription}/-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    # prepare the dataloader
    dataset = TestSDataset(args.data_path, args.num_frames)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        # shuffle=True,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=n,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    for _ in pbar:
        # Sample inputs:
        for y_mag, f, current_subfolders in loader:
            y = y_mag.to(device)
            f = f.to(device)
            z = torch.randn(n, model.num_frames, model.in_channels, latent_size, latent_size, device=device)
            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_f = y[:,:,1].unsqueeze(-1).to(y.device)
                y_null_s = torch.ones(y_f.shape).to(y.device)
                y_null = torch.cat([y_null_s,y_f],dim=-1).to(y.device)
                y = torch.cat([y, y_null], 0)
                f = torch.cat([f, f], 0)
                model_kwargs = dict(y=y, f=f, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y, f=f)
                sample_fn = model.forward
            noise_path = f"./sampleStep/{args.num_sampling_steps}/step"
            os.makedirs(noise_path, exist_ok=True)
            origin_noise = z[0,0,0,:,:].squeeze()
            utils.save_image(
                origin_noise,
                f"{noise_path}/origin_noise.png",
                normalize=True,
            )
                        
            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            # samples = samples[:,:,0:args.latent_channels//2,:,:]
            samples1 = samples[:,:,0:args.latent_channels,:,:]
            samples2 = samples[:,:,args.latent_channels:,:,:]
            B, T, C, h, w = samples1.shape
            samples1 = rearrange(samples1, 'b f c h w -> (b f) c h w')
            samples2 = rearrange(samples2, 'b f c h w -> (b f) c h w')
            samples1_out = vae.decode_quant_VDT(samples1)
            samples2_out = vae.decode_quant_VDT(samples2)
            samples1_out = rearrange(samples1_out, '(b f) c h w -> b f c h w', b=B)
            samples2_out = rearrange(samples2_out, '(b f) c h w -> b f c h w', b=B)
            samples1 = rearrange(samples1, '(b f) c h w -> b f c h w', b=B)
            samples2 = rearrange(samples2, '(b f) c h w -> b f c h w', b=B)
            B, T, c, H, W = samples1_out.shape
            # print(samples.shape)
            # samples = samples.view(-1, C, H, W).to(device=device)
            # save_image(x, "input.png", nrow=16, normalize=True, value_range=(-1, 1))
            # print(x.shape)
            # samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            
            for i, samples1_out in enumerate(samples1_out):
                # index = i * dist.get_world_size() + rank + total
                os.makedirs(f"{sample_folder_dir}/{current_subfolders[i]}/FirstLayer", exist_ok=True)
                os.makedirs(f"{sample_folder_dir}/{current_subfolders[i]}/SecondLayer", exist_ok=True)
                f1=h5py.File(f"{sample_folder_dir}/{current_subfolders[i]}/FirstLayer/surfj1.hdf5","w")
                surfj1_data = samples1_out.permute(1,0,2,3).cpu().numpy()
                f1.create_dataset("surfj1",data=surfj1_data)
                f1.close()
                f2=h5py.File(f"{sample_folder_dir}/{current_subfolders[i]}/SecondLayer/surfj2.hdf5","w")
                surfj2_data = samples2_out[i,:,:,:,:].permute(1,0,2,3).cpu().numpy()
                f2.create_dataset("surfj2",data=surfj2_data)
                f2.close()
                for cc in range(c):
                    samples1_out_cc = samples1_out[:,cc,:,:]
                    samples2_out_cc = samples2_out[i,:,cc,:,:]
                    samples1_out_cc = samples1_out_cc.contiguous().view(-1, 1, H, W).to(device=device)
                    samples2_out_cc = samples2_out_cc.contiguous().view(-1, 1, H, W).to(device=device)
                    os.makedirs(f"{sample_folder_dir}/{current_subfolders[i]}/FirstLayer", exist_ok=True)
                    os.makedirs(f"{sample_folder_dir}/{current_subfolders[i]}/SecondLayer", exist_ok=True)
                    save_image(samples1_out_cc, f"{sample_folder_dir}/{current_subfolders[i]}/FirstLayer/surface1_channel{cc:06d}.png", nrow=args.num_frames, normalize=True, value_range=(0, 1.5))
                    save_image(samples2_out_cc, f"{sample_folder_dir}/{current_subfolders[i]}/SecondLayer/surface2_channel{cc:06d}.png", nrow=args.num_frames, normalize=True, value_range=(0, 1.5))
            # also save latent samples for review
            for j, samples1 in enumerate(samples1):
                os.makedirs(f"{sample_folder_dir}/{current_subfolders[j]}/FirstLayer", exist_ok=True)
                os.makedirs(f"{sample_folder_dir}/{current_subfolders[j]}/SecondLayer", exist_ok=True)
                # index = j * dist.get_world_size() + rank + total
                for CC in range(C):
                    samples1_cc = samples1[:,CC,:,:]
                    samples2_cc = samples2[j,:,CC,:,:]
                    samples1_cc = samples1_cc.contiguous().view(-1, 1, h, w).to(device=device)
                    samples2_cc = samples2_cc.contiguous().view(-1, 1, h, w).to(device=device)
                    save_image(samples1_cc, f"{sample_folder_dir}/{current_subfolders[j]}/FirstLayer/latent1_channel{CC:06d}.png", nrow=args.num_frames, normalize=True, value_range=(-1.3, 1.3))
                    save_image(samples2_cc, f"{sample_folder_dir}/{current_subfolders[j]}/SecondLayer/latent2_channel{CC:06d}.png", nrow=args.num_frames, normalize=True, value_range=(-1.3, 1.3))
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # create_npz_from_sample_folder(sample_folder_dir, args.image_channels, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-M/2")
    parser.add_argument("--vae",  type=str, default="vqvae2")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=[64, 128, 256, 512], default=128)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--data-path", type=str, default='./vae/vip/tl')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    
    parser.add_argument("--num_frames", type=int, default=19)
    parser.add_argument("--num_freqs", type=int, default=19)

    parser.add_argument("--ckpt", type=str, default="doubleJmagSmag_attnTsqrtCos_MVDT_1530000.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--vae-name",  type=str, default="vqvae_fullsize_currents_mag_symetricEhance_120.pt")
    parser.add_argument("--in-channels", type=int, default=2)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--noise-schedule", type=str, default="squaredcos_cap_v2") #linear,squaredcos_cap_v2
    parser.add_argument("--special-discription",  type=str, default="viptl")
    parser.add_argument("--num-fid-samples", type=int, default=1)
    parser.add_argument("--num-sampling-steps", type=str, default="10,4,4,4,4,4,10,20,50,100")
    parser.add_argument("--cfg-scale",  type=float, default=4)
    parser.add_argument("--num-generation", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
