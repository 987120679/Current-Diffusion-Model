import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist

from dataset import SurfJDataset
from einops import rearrange 


def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, img in enumerate(loader):
        model.zero_grad()

        batch_size = img.shape[0]
        img = rearrange(img, '... b f c h w -> ... (b f) c h w')

        img = img.to(device)

        out, latent_loss = model(img)
        # quant_t, quant_b, diff, id_t, id_b = model.encode(img)
        # print(quant_t.shape)
        # print(quant_b.shape)
        # print(diff.shape)
        # print(id_t.shape)
        # print(id_b.shape)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 1000 == 0:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                sample = sample[:,0:1,:,:]
                out = out[:,0:1,:,:]

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    value_range=(-1, 1),
                )

                model.train()


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = SurfJDataset(args.path)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = VQVAE(in_channel=args.in_channel,channel=128,n_res_block=2,n_res_channel=32,embed_dim=4,n_embed=512,decay=0.99).to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--path", type=str, default='./data')
    parser.add_argument("--in_channel", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
