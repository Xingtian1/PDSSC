# -*- coding: utf-8 -*-
"""
Stage III: channel-adaptive bitrate controller training.

This script freezes the codec + flow completion stack from Stage II and learns
an external predictor that selects the smallest feasible transmission depth N
whose reconstruction distortion is sufficiently close to the best feasible one.
"""

import argparse
import logging
import os
import random
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speechtokenizer.model import SpeechTokenizer
from speechtokenizer.flow import BitrateController, FlowMatchingModel, PretrainedSpeakerEncoder
from speechtokenizer.flow.dataset import LibriSpeechFlowDataset
from scripts.eval_utils import _latent_linear_interp, flow_sample


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs = {}
        if backend == "nccl":
            init_kwargs["device_id"] = local_rank
        dist.init_process_group(backend=backend, **init_kwargs)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return distributed, local_rank, world_size


def cleanup_distributed():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    wavs = torch.stack([item[0] for item in batch], dim=0)
    ref_wavs = torch.stack([item[1] for item in batch], dim=0)
    return wavs, ref_wavs


def extract_obs_features(latent: torch.Tensor) -> torch.Tensor:
    """
    Build lightweight quality-aware statistics from the received partial latent.
    latent: (1, D, T)
    return: (6,)
    """
    x = latent.squeeze(0).float()  # (D, T)
    frame_energy = x.pow(2).mean(dim=0).sqrt()
    frame_var = x.var(dim=0, unbiased=False)
    if x.shape[-1] > 1:
        delta = (x[:, 1:] - x[:, :-1]).abs().mean()
    else:
        delta = x.new_tensor(0.0)
    feat = torch.stack([
        x.abs().mean(),
        x.std(unbiased=False),
        x.abs().amax(),
        frame_energy.mean(),
        frame_var.mean(),
        delta,
    ], dim=0)
    return feat


def build_quality_preserving_targets(
    dist_mat: torch.Tensor,
    feasible: torch.Tensor,
    candidate_layers,
    tol_mode: str,
    rel_margin: float,
    abs_margin: float,
):
    """
    Build pseudo-labels for controller training.

    dist_mat: (B, A), distortion for each candidate depth
    feasible: (B, A), whether the candidate satisfies R_N <= Rmax

    Returns:
      target_idx: (B,), smallest feasible depth whose distortion is close to
        the best feasible distortion
      best_dist: (B,), best feasible distortion
      thresh: (B,), distortion threshold used to construct the target
    """
    inf = torch.full_like(dist_mat, float("inf"))
    feasible_dist = torch.where(feasible, dist_mat, inf)
    best_dist = feasible_dist.min(dim=1).values

    if tol_mode == "abs":
        thresh = best_dist + abs_margin
    else:
        thresh = best_dist * (1.0 + rel_margin)

    within_tol = feasible & (dist_mat <= thresh.unsqueeze(1))
    target_idx = within_tol.float().argmax(dim=1)

    if not bool(within_tol.any(dim=1).all()):
        raise RuntimeError("failed to build controller pseudo-labels: no feasible target within tolerance")

    return target_idx.long(), best_dist, thresh


@torch.no_grad()
def build_candidate_latents(st_model, wav: torch.Tensor, candidate_layers, p_loss: float, device):
    """
    Encode once, then construct cumulative channel-impaired latents for all
    candidate transmission depths under one shared packet-loss realization.
    wav: (1, T)
    return:
      latent_by_n: dict[n_layers] -> (1, D, T_enc)
    """
    x = wav.unsqueeze(0).to(device)  # (1, 1, T)
    codes_all = st_model.encode(x)   # (L, 1, T_enc)
    t_enc = codes_all.shape[2]
    dim = st_model.quantizer.dimension

    def safe_decode(idx: int) -> torch.Tensor:
        vq_l = st_model.quantizer.vq.layers[idx]
        dec = vq_l.decode(codes_all[idx])
        if dec.shape[-1] == dim:
            dec = dec.permute(0, 2, 1)
        return dec.contiguous()

    decoded_layers = [safe_decode(i) for i in range(max(candidate_layers))]

    q1_recv = (torch.rand(t_enc, device=device) >= p_loss)
    latent_q1 = decoded_layers[0]
    if not bool(q1_recv.all()):
        latent_q1 = _latent_linear_interp(latent_q1, q1_recv)

    latent_running = latent_q1.clone()
    latent_by_n = {1: latent_running.clone()}
    for l in range(1, max(candidate_layers)):
        recv_mask = (torch.rand(t_enc, device=device) >= p_loss).float()
        latent_running = latent_running + decoded_layers[l] * recv_mask.unsqueeze(0).unsqueeze(0)
        latent_by_n[l + 1] = latent_running.clone()
    return latent_by_n


def build_models(args, device):
    st_model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    st_model.eval().to(device)
    for p in st_model.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(args.flow_ckpt, map_location="cpu")
    saved_args = argparse.Namespace(**ckpt.get("args", {}))
    flow_model = FlowMatchingModel(
        latent_dim=st_model.quantizer.dimension,
        base_ch=getattr(saved_args, "base_ch", 512),
        ch_mults=tuple(getattr(saved_args, "ch_mults", [1, 1, 2])),
        cond_dim=getattr(saved_args, "cond_dim", 512),
        spk_dim=getattr(saved_args, "spk_dim", 256),
        time_dim=getattr(saved_args, "time_dim", 128),
        n_res=getattr(saved_args, "n_res", 2),
        n_mid_res=getattr(saved_args, "n_mid_res", 2),
    ).to(device)
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["flow_model"].items()})
    flow_model.eval()
    for p in flow_model.parameters():
        p.requires_grad_(False)

    spk_encoder = PretrainedSpeakerEncoder(
        emb_dim=getattr(saved_args, "spk_dim", 256),
        save_dir=os.path.join(os.path.dirname(args.flow_ckpt), "spkrec-ecapa"),
    ).to(device)
    spk_encoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["spk_encoder"].items()}, strict=False)
    spk_encoder.eval()
    for p in spk_encoder.parameters():
        p.requires_grad_(False)

    controller = BitrateController(
        num_actions=len(args.n_layers_list),
        hidden_dim=args.hidden_dim,
        in_dim=2 + args.obs_feat_dim,
    ).to(device)
    return st_model, flow_model, spk_encoder, controller


def train(args):
    distributed, local_rank, world_size = setup_distributed()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_controller.log")
    if is_main_process():
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == file_handler.baseFilename
                   for h in logger.handlers):
            logger.addHandler(file_handler)
    logger.propagate = True
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if is_main_process():
        logger.info("device=%s", device)
        logger.info("distributed=%s world_size=%d local_rank=%d", distributed, world_size, local_rank)
        logger.info("log_file=%s", log_path)

    st_model, flow_model, spk_encoder, controller = build_models(args, device)
    if distributed:
        controller = DDP(
            controller,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    dataset = LibriSpeechFlowDataset(
        data_dir=args.data_dir,
        split=args.split,
        segment_sec=args.segment_sec,
        sample_rate=st_model.sample_rate,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=args.lr * 0.1
    )

    candidate_layers = list(args.n_layers_list)
    best_loss = float("inf")
    no_improve = 0
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        controller_to_load = controller.module if isinstance(controller, DDP) else controller
        controller_to_load.load_state_dict(ckpt["controller"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        best_loss = float(ckpt.get("best_loss", best_loss))
        if is_main_process():
            logger.info("resumed from %s", args.resume)

    try:
        for epoch in range(args.max_epochs):
            controller.train()
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_pred_rate = 0.0
            epoch_target_rate = 0.0

            for step, (wavs, ref_wavs) in enumerate(loader):
                wavs = wavs.to(device)
                ref_wavs = ref_wavs.to(device)
                if args.plr_values:
                    plr_choices = torch.tensor(args.plr_values, device=device, dtype=torch.float32)
                    plr_scalar = plr_choices[torch.randint(low=0, high=len(args.plr_values), size=(1,), device=device)].item()
                    plr = torch.full((wavs.size(0),), float(plr_scalar), device=device)
                else:
                    plr_scalar = float(torch.empty(1, device=device).uniform_(args.plr_min, args.plr_max).item())
                    plr = torch.full((wavs.size(0),), plr_scalar, device=device)

                if args.r_max_values:
                    r_choices = torch.tensor(args.r_max_values, device=device, dtype=torch.float32)
                    r_scalar = r_choices[torch.randint(low=0, high=len(args.r_max_values), size=(1,), device=device)].item()
                    r_max = torch.full((wavs.size(0),), float(r_scalar), device=device)
                else:
                    r_max = torch.full((wavs.size(0),), float(args.r_max), device=device)

                with torch.no_grad():
                    spk_emb = spk_encoder(ref_wavs.squeeze(1))

                candidate_losses = []
                obs_feat_list = []
                anchor_n = candidate_layers[0]
                for b in range(wavs.size(0)):
                    sample_losses = []
                    wav_b = wavs[b : b + 1]
                    spk_b = spk_emb[b : b + 1]
                    plr_b = float(plr[b].item())
                    with torch.no_grad():
                        latent_by_n = build_candidate_latents(
                            st_model,
                            wav_b.squeeze(0),
                            candidate_layers=candidate_layers,
                            p_loss=plr_b,
                            device=device,
                        )
                        obs_feat_list.append(extract_obs_features(latent_by_n[anchor_n]))
                    for n_layers in candidate_layers:
                        with torch.no_grad():
                            lat_in = latent_by_n[n_layers]
                            lat_8 = flow_sample(flow_model, lat_in, spk_b, n_steps=args.n_steps, n_layers=n_layers)
                            lat_8 = lat_8[..., :lat_in.shape[-1]]
                            wav_rec = st_model.decoder(lat_8).squeeze(1)
                            wav_ref = wav_b.squeeze(1)
                            T = min(wav_rec.shape[-1], wav_ref.shape[-1])
                            wav_rec = wav_rec[..., :T]
                            wav_ref = wav_ref[..., :T]
                            recon_dist = F.mse_loss(wav_rec, wav_ref, reduction="mean")
                        sample_losses.append(recon_dist)
                    candidate_losses.append(torch.stack(sample_losses))

                obs_feat = torch.stack(obs_feat_list, dim=0)
                score = controller(plr.unsqueeze(-1), r_max=r_max.unsqueeze(-1), obs_feat=obs_feat)
                dist_mat = torch.stack(candidate_losses, dim=0)  # (B, A)
                rate_vec = torch.tensor([n * 0.5 for n in candidate_layers], device=device, dtype=torch.float32)
                feasible = rate_vec.unsqueeze(0) <= (r_max.unsqueeze(1) + 1e-6)
                masked_score = score.masked_fill(~feasible, -1e9)
                target_idx, best_dist, dist_thresh = build_quality_preserving_targets(
                    dist_mat=dist_mat,
                    feasible=feasible,
                    candidate_layers=candidate_layers,
                    tol_mode=args.tol_mode,
                    rel_margin=args.rel_margin,
                    abs_margin=args.abs_margin,
                )
                pred_idx = masked_score.argmax(dim=1)
                pred_rate = rate_vec[pred_idx]
                target_rate = rate_vec[target_idx]
                pred_dist = dist_mat.gather(1, pred_idx.unsqueeze(1)).squeeze(1)
                target_dist = dist_mat.gather(1, target_idx.unsqueeze(1)).squeeze(1)
                loss = F.cross_entropy(masked_score, target_idx)
                acc = (pred_idx == target_idx).float().mean()
                avg_pred_rate = pred_rate.float().mean()
                avg_target_rate = target_rate.float().mean()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += float(loss.item())
                epoch_acc += float(acc.item())
                epoch_pred_rate += float(avg_pred_rate.item())
                epoch_target_rate += float(avg_target_rate.item())

                if is_main_process() and step % args.log_every == 0:
                    pred_layers = ",".join(
                        f"N{candidate_layers[i]}:{c}" for i, c in enumerate(
                            torch.bincount(pred_idx.detach().cpu(), minlength=len(candidate_layers)).tolist()
                        ) if c > 0
                    )
                    target_layers = ",".join(
                        f"N{candidate_layers[i]}:{c}" for i, c in enumerate(
                            torch.bincount(target_idx.detach().cpu(), minlength=len(candidate_layers)).tolist()
                        ) if c > 0
                    )
                    sample_cand = ",".join(
                        f"N{n}:{dist_mat[0, i].item():.4f}" for i, n in enumerate(candidate_layers)
                        if bool(feasible[0, i].item())
                    )
                    logger.info(
                        "epoch=%d step=%d/%d loss=%.6f acc=%.4f best=%.6f thr=%.6f target_mse=%.6f pred_mse=%.6f target_rate=%.3f avg_rate=%.3f plr=%.3f rmax=%.3f target=[%s] pred=[%s] cand=[%s]",
                        epoch, step, len(loader), loss.item(),
                        float(acc.item()),
                        float(best_dist.mean().item()),
                        float(dist_thresh.mean().item()),
                        float(target_dist.mean().item()),
                        float(pred_dist.mean().item()),
                        float(avg_target_rate.item()),
                        float(avg_pred_rate.item()),
                        float(plr.mean().item()),
                        float(r_max.mean().item()),
                        target_layers,
                        pred_layers
                        ,
                        sample_cand
                    )

            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            epoch_acc_tensor = torch.tensor(epoch_acc, device=device)
            epoch_pred_rate_tensor = torch.tensor(epoch_pred_rate, device=device)
            epoch_target_rate_tensor = torch.tensor(epoch_target_rate, device=device)
            if distributed:
                dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(epoch_acc_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(epoch_pred_rate_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(epoch_target_rate_tensor, op=dist.ReduceOp.SUM)
            avg_loss = epoch_loss_tensor.item() / max(1, len(loader) * world_size)
            avg_acc = epoch_acc_tensor.item() / max(1, len(loader) * world_size)
            avg_pred_rate_epoch = epoch_pred_rate_tensor.item() / max(1, len(loader) * world_size)
            avg_target_rate_epoch = epoch_target_rate_tensor.item() / max(1, len(loader) * world_size)
            scheduler.step(avg_loss)
            if is_main_process():
                logger.info(
                    "epoch=%d avg_loss=%.4f avg_acc=%.4f avg_target_rate=%.3f avg_rate=%.3f lr=%.2e",
                    epoch,
                    avg_loss,
                    avg_acc,
                    avg_target_rate_epoch,
                    avg_pred_rate_epoch,
                    optimizer.param_groups[0]["lr"],
                )

            improved = avg_loss < best_loss
            if improved:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1

            controller_state = controller.module.state_dict() if isinstance(controller, DDP) else controller.state_dict()
            train_ckpt = {
                "epoch": epoch,
                "controller": controller_state,
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
            }
            if improved and is_main_process():
                torch.save(train_ckpt, os.path.join(args.save_dir, "best_controller_train.pt"))
                infer_ckpt = {
                    "epoch": epoch,
                    "controller": controller_state,
                    "best_loss": best_loss,
                    "args": vars(args),
                }
                torch.save(infer_ckpt, os.path.join(args.save_dir, "best_controller.pt"))

            if no_improve >= args.patience:
                if is_main_process():
                    logger.info("early stopping")
                break
    finally:
        cleanup_distributed()


def parse_args():
    p = argparse.ArgumentParser(description="Stage III bitrate controller training")
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--flow_ckpt", default="output/flow_checkpoints_stage3/best_train.pt")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--split", default="train-clean-100")
    p.add_argument("--save_dir", default="output/controller_checkpoints")
    p.add_argument("--resume", default="")
    p.add_argument("--n_layers_list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--segment_sec", type=float, default=4.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--obs_feat_dim", type=int, default=6)
    p.add_argument("--r_max", type=float, default=3.0)
    p.add_argument("--r_max_values", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    p.add_argument("--plr_min", type=float, default=0.0)
    p.add_argument("--plr_max", type=float, default=0.30)
    p.add_argument("--plr_values", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    p.add_argument("--tol_mode", choices=["rel", "abs"], default="rel")
    p.add_argument("--rel_margin", type=float, default=0.02)
    p.add_argument("--abs_margin", type=float, default=0.0)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except Exception:
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[train_controller][rank={rank} local_rank={local_rank}] unhandled exception", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
