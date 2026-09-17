# -*- coding: utf-8 -*-
"""
Rectified Flow 生成模型 阶段3 训练脚本

相对 train_flow_v2.py（阶段2）的改进：
  1. N 从 2 开始（去掉 0.5kbps），难样本加权公式对应调整
       N=2→2.0x, N=3→1.75x, N=4→1.5x, N=5→1.25x, N>=6→1.0x
  2. STFT 损失 = log-幅度谱 L1（PESQ 代理）+ 谱收敛项（SC，额外 PESQ 信号）
  3. 低 N 噪声增强（N<=3 在 x_t 上加高斯噪声），提升低码率鲁棒性 → 改善 STOI
  4. ReduceLROnPlateau patience=10（阶段2 为 5），min_lr=lr*0.1（阶段2 为 0.01）
       防止 LR 过早死掉导致假收敛
  5. Early stopping patience=30（阶段2 为 20）
  6. stft_every 默认=2（阶段2 为 4），更频繁的感知梯度信号
  7. 支持从阶段2 checkpoint（--resume_v2）热启动

用法（从阶段2 best.pt 热启动）:
  python scripts/train_flow_v3.py \\
      --config_path model_hub/speechtokenizer_hubert_avg/config.json \\
      --ckpt_path   model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt \\
      --data_dir    /path/to/LibriSpeech \\
      --split       train-clean-100 \\
      --save_dir    output/flow_checkpoints_stage3 \\
      --resume_v2   output/flow_checkpoints_stage2/best.pt

用法（从阶段3 checkpoint 续训）:
  python scripts/train_flow_v3.py \\
      ... \\
      --resume output/flow_checkpoints_stage3/best_train.pt
"""

import os
import argparse
import random
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from speechtokenizer import SpeechTokenizer
from speechtokenizer.flow import FlowMatchingModel, PretrainedSpeakerEncoder
from speechtokenizer.flow.dataset import LibriSpeechFlowDataset


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    wavs     = [item[0] for item in batch]
    ref_wavs = [item[1] for item in batch]
    wavs     = torch.stack(wavs,     dim=0)
    ref_wavs = torch.stack(ref_wavs, dim=0)
    return wavs, ref_wavs


# ─────────────────────────────────────────────────────────────────────────────
# 多分辨率 STFT 损失（log-L1 + 谱收敛）
# ─────────────────────────────────────────────────────────────────────────────

# (n_fft, hop_length, win_length)
_STFT_CONFIGS = [
    (512,  128, 512),
    (1024, 256, 1024),
    (2048, 512, 2048),
]

_HANN_WINDOWS = {}


def _get_hann_window(win_length, device):
    key = (win_length, str(device))
    if key not in _HANN_WINDOWS:
        _HANN_WINDOWS[key] = torch.hann_window(win_length, periodic=True, device=device)
    return _HANN_WINDOWS[key]


def multi_res_stft_loss(wav_pred, wav_ref, lambda_sc=0.5):
    """
    多分辨率 STFT 损失：log-幅度谱 L1 + 谱收敛（Spectral Convergence）。
      wav_pred, wav_ref : (B, T) float32，值域 [-1, 1]
      lambda_sc         : 谱收敛项权重（相对于 log-L1 而言）
    返回标量 loss。

    两项作用：
      - log-L1  : 对 PESQ（感知语音质量）灵敏，关注幅度谱形状
      - SC      : 对 STOI（语音可懂度）更灵敏，关注能量分布的整体误差
    """
    total_log = wav_pred.new_zeros(1).squeeze()
    total_sc  = wav_pred.new_zeros(1).squeeze()

    for n_fft, hop, win in _STFT_CONFIGS:
        window = _get_hann_window(win, wav_pred.device)
        s_pred = torch.stft(
            wav_pred, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True,
        ).abs().clamp(min=1e-5)   # (B, F, T_frames)
        s_ref = torch.stft(
            wav_ref, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True,
        ).abs().clamp(min=1e-5)

        # log-幅度谱 L1
        total_log = total_log + F.l1_loss(s_pred.log(), s_ref.log())

        # 谱收敛：||S_pred - S_ref||_F / ||S_ref||_F，按 batch 平均
        diff_norm = (s_pred - s_ref).norm(dim=(-2, -1))          # (B,)
        ref_norm  = s_ref.norm(dim=(-2, -1)).clamp(min=1e-8)     # (B,)
        total_sc  = total_sc + (diff_norm / ref_norm).mean()

    n = len(_STFT_CONFIGS)
    return total_log / n + lambda_sc * total_sc / n


# ─────────────────────────────────────────────────────────────────────────────
# 批量获取 (latent_received, latent_8) 对
# N 从 2 开始（去掉 0.5kbps）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def prepare_training_batch(st_model, wavs, device, p_loss_values=None,
                            low_rate_prob=0.25):
    if p_loss_values is None:
        p_loss_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    B = wavs.shape[0]

    codes_all = st_model.encode(wavs.to(device))       # (8, B, T_enc)
    latent_8  = st_model.quantizer.decode(codes_all)   # (B, D, T_enc)

    T_enc = codes_all.shape[2]
    D     = latent_8.shape[1]

    # 偏置采样：low_rate_prob 概率采 N=2（1kbps），其余均匀采 N=[3,8]
    # 注意：不超过 0.3，避免过度压制其他码率（原均匀分布每个 N 各占 1/7≈14%）
    use_low   = torch.rand(B) < low_rate_prob
    n_low     = torch.full((B,), 2)
    n_high    = torch.randint(3, 9, (B,))
    n_choices = torch.where(use_low, n_low, n_high)

    # PLR 对所有 N 统一采样，N=2 也保留丢包训练
    # （若 N=2 强制 PLR=0，推理时遇丢包模型会失效，PLR 曲线会塌）
    plr_vec = torch.tensor(p_loss_values)[
        torch.randint(len(p_loss_values), (B,))
    ]

    def batch_decode_layer(layer_idx):
        vq_l = st_model.quantizer.vq.layers[layer_idx]
        flat = codes_all[layer_idx].reshape(1, B * T_enc)
        vecs = vq_l.decode(flat)
        bv   = vecs.squeeze(0)
        if bv.shape[0] == D:
            bv = bv.T
        return bv.reshape(B, T_enc, D).permute(0, 2, 1).contiguous()   # (B, D, T_enc)

    # Q1 始终可靠传输
    latent_received = batch_decode_layer(0)

    # Q2..Q8
    for l in range(1, 8):
        decoded      = batch_decode_layer(l)
        layer_active = (n_choices >= l + 1).float().to(device)          # (B,)
        recv_mask    = (torch.rand(B, T_enc) >= plr_vec.unsqueeze(1)).float().to(device)
        recv         = (layer_active.unsqueeze(1) * recv_mask).unsqueeze(1)
        latent_received = latent_received + decoded * recv

    T = min(latent_received.shape[-1], latent_8.shape[-1])
    return latent_received[..., :T], latent_8[..., :T], n_choices


# ─────────────────────────────────────────────────────────────────────────────
# 训练主循环
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: {}".format(device))

    # ── 加载 SpeechTokenizer（冻结）──────────────────────────────────
    logger.info("加载 SpeechTokenizer...")
    st_model = SpeechTokenizer.load_from_checkpoint(
        args.config_path, args.ckpt_path
    )
    st_model.eval().to(device)
    for p in st_model.parameters():
        p.requires_grad_(False)

    latent_dim = st_model.quantizer.dimension
    logger.info("  latent_dim = {}".format(latent_dim))

    # ── 说话人编码器 ─────────────────────────────────────────────────
    logger.info("加载预训练说话人编码器...")
    spk_encoder = PretrainedSpeakerEncoder(
        emb_dim  = args.spk_dim,
        save_dir = os.path.join(args.save_dir, "spkrec-ecapa"),
    ).to(device)

    # ── Flow 模型 ─────────────────────────────────────────────────────
    logger.info("初始化 Flow 模型...")
    flow_model = FlowMatchingModel(
        latent_dim = latent_dim,
        base_ch    = args.base_ch,
        ch_mults   = tuple(args.ch_mults),
        cond_dim   = args.cond_dim,
        spk_dim    = args.spk_dim,
        time_dim   = args.time_dim,
        n_res      = args.n_res,
        n_mid_res  = args.n_mid_res,
    ).to(device)

    n_params = sum(p.numel() for p in flow_model.parameters()) / 1e6
    n_spk    = sum(p.numel() for p in spk_encoder.parameters()) / 1e6
    logger.info("  FlowModel  参数量: {:.1f}M".format(n_params))
    logger.info("  SpeakerEnc 参数量: {:.1f}M".format(n_spk))

    # ── 数据集 ────────────────────────────────────────────────────────
    logger.info("加载数据集...")
    dataset = LibriSpeechFlowDataset(
        data_dir    = args.data_dir,
        split       = args.split,
        segment_sec = args.segment_sec,
        sample_rate = st_model.sample_rate,
    )
    loader = DataLoader(
        dataset,
        batch_size         = args.batch_size,
        shuffle            = True,
        num_workers        = args.num_workers,
        collate_fn         = collate_fn,
        pin_memory         = device.type == "cuda",
        drop_last          = True,
        persistent_workers = args.num_workers > 0,
    )

    # ── 优化器 ────────────────────────────────────────────────────────
    params    = list(flow_model.parameters()) + list(spk_encoder.proj.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr           = args.lr,
        betas        = (0.9, 0.99),
        weight_decay = 1e-4,
    )
    # patience=10（阶段2 为 5），min_lr=lr*0.1（阶段2 为 0.01）
    # 防止 LR 过早降到底，导致假收敛
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 10,
        min_lr   = args.lr * 0.1,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 恢复检查点 ────────────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")

    def _strip_compile_prefix(sd):
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # 优先：从阶段3 checkpoint 续训
    if args.resume and os.path.isfile(args.resume):
        logger.info("[Resume] 从阶段3 checkpoint 恢复: {}".format(args.resume))
        ckpt    = torch.load(args.resume, map_location="cpu")
        missing, _ = flow_model.load_state_dict(
            _strip_compile_prefix(ckpt["flow_model"]), strict=False)
        if missing:
            logger.info("[Resume] 新增参数（随机初始化）: {}".format(missing))
        spk_encoder.load_state_dict(
            _strip_compile_prefix(ckpt["spk_encoder"]), strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            logger.info("[Resume] ✓ 优化器状态已恢复，LR 重置为 {:.2e}".format(args.lr))
        else:
            logger.info("[Resume] ⚠ 无优化器状态，使用新优化器，LR={:.2e}".format(args.lr))
        best_loss = float("inf")
        logger.info("[Resume] ✓ best_loss 重置为 inf，仅与本轮训练内比较")
        start_epoch = ckpt["epoch"] + 1
        logger.info("[Resume] ✓ 从 epoch {} 继续，best_loss={:.4f}".format(
            start_epoch, best_loss))
        del ckpt

    # 其次：从阶段2 best.pt 热启动（只迁移权重）
    elif args.resume_v2 and os.path.isfile(args.resume_v2):
        logger.info("[Resume-V2] 从阶段2 checkpoint 热启动: {}".format(args.resume_v2))
        ckpt = torch.load(args.resume_v2, map_location="cpu")
        missing, unexpected = flow_model.load_state_dict(
            _strip_compile_prefix(ckpt["flow_model"]), strict=False)
        if missing:
            logger.info("[Resume-V2] 新增参数（随机初始化）: {}".format(missing))
        if unexpected:
            logger.info("[Resume-V2] 忽略多余参数: {}".format(unexpected))
        spk_encoder.load_state_dict(
            _strip_compile_prefix(ckpt["spk_encoder"]), strict=False)
        del ckpt
        logger.info("[Resume-V2] ✓ 权重迁移完成，优化器重置，LR={:.2e}".format(args.lr))

    if hasattr(torch, "compile"):
        logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
        logging.getLogger("torch._inductor").setLevel(logging.WARNING)
        flow_model = torch.compile(flow_model)
        logger.info("torch.compile 已启用")

    # ── 训练循环 ──────────────────────────────────────────────────────
    loss_fn_none = nn.MSELoss(reduction="none")
    no_improve   = 0

    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        flow_model.train()
        spk_encoder.train()

        epoch_loss = 0.0
        t0 = time.time()

        for step, (wavs, ref_wavs) in enumerate(loader):
            # 1. 获取 (x0, x1, n_layers_batch)，N ~ U[2,8]
            x0, x1, n_layers_batch = prepare_training_batch(
                st_model       = st_model,
                wavs           = wavs,
                device         = device,
                p_loss_values  = args.p_loss_values,
                low_rate_prob  = args.low_rate_prob,
            )
            n_layers_batch = n_layers_batch.to(device)   # (B,) int

            # 2. 说话人嵌入
            ref     = ref_wavs.to(device)
            spk_emb = spk_encoder(ref.squeeze(1))         # (B, spk_dim)

            # 3. Rectified Flow 插值
            B    = x0.shape[0]
            t    = torch.rand(B, device=device)
            t_bc = t[:, None, None]
            xt   = (1 - t_bc) * x0 + t_bc * x1           # (B, D, T)

            # 阶段3 改进③：分级噪声增强
            #   N=2 → 更大噪声（noise_std_n2），专攻 1kbps 鲁棒性
            #   N=3 → 标准噪声（noise_std）
            if args.noise_std_n2 > 0:
                mask_n2 = (n_layers_batch == 2).float().to(device)[:, None, None]
                xt = xt + torch.randn_like(xt) * args.noise_std_n2 * mask_n2
            if args.noise_std > 0:
                mask_n3 = (n_layers_batch == 3).float().to(device)[:, None, None]
                xt = xt + torch.randn_like(xt) * args.noise_std * mask_n3

            # 4. 目标速度场
            target = x1 - x0                              # (B, D, T)

            # 5. 前向预测
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                v_pred = flow_model(xt, t, spk_emb, n_layers_batch)   # (B, D, T)

                # 阶段3 改进①：难样本加权 Flow 损失（N 从 2 开始）
                # N=2→2.0x, N=3→1.75x, N=4→1.5x, N=5→1.25x, N>=6→1.0x
                # 再对 N=2 额外加 n2_extra_weight，专攻 1kbps
                n_weight  = ((10.0 - n_layers_batch.float()) / 4.0).clamp(min=1.0)
                n2_boost  = (n_layers_batch == 2).float().to(device) * args.n2_extra_weight
                n_weight  = (n_weight + n2_boost)[:, None, None]      # (B,1,1)
                L_flow    = (loss_fn_none(v_pred, target) * n_weight).mean()

            # 阶段3 改进②：多分辨率 STFT 损失（log-L1 + 谱收敛）
            L_stft      = torch.tensor(0.0, device=device)
            decoder_ran = False
            wav_pred_c  = None

            if args.lambda_stft > 0 and step % args.stft_every == 0:
                x1_pred = (x0 + v_pred).float()
                with torch.backends.cudnn.flags(enabled=False):
                    wav_pred = st_model.decoder(x1_pred)   # (B, 1, T_wav)
                wav_ref_raw = wavs.to(device).float()
                T_wav       = min(wav_pred.shape[-1], wav_ref_raw.shape[-1])
                wav_pred_c  = wav_pred[..., :T_wav].squeeze(1).clamp(-1.0, 1.0)
                wav_ref_c   = wav_ref_raw[..., :T_wav].squeeze(1)
                decoder_ran = True
                L_stft = multi_res_stft_loss(wav_pred_c, wav_ref_c,
                                             lambda_sc=args.lambda_sc)

            if not torch.isfinite(L_stft):
                L_stft = torch.tensor(0.0, device=device)

            loss = L_flow + args.lambda_stft * L_stft

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm=1.0)

            if not torch.isfinite(loss):
                logger.warning("[step {}] loss=nan/inf，跳过参数更新".format(step))
                optimizer.zero_grad()
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % args.log_every == 0:
                lr_cur   = optimizer.param_groups[0]["lr"]
                stft_val = L_stft.item() if decoder_ran else float("nan")
                logger.info(
                    "[epoch {:3d} | step {:5d}/{}] "
                    "loss={:.4f}  flow={:.4f}  stft={:.4f}  lr={:.2e}".format(
                        epoch, step, len(loader),
                        loss.item(), L_flow.item(), stft_val, lr_cur,
                    )
                )

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        scheduler.step(avg_loss)
        lr_cur = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch {:3d} 完成 | avg_loss={:.4f} | lr={:.2e} | 耗时={:.0f}s".format(
                epoch, avg_loss, lr_cur, elapsed
            )
        )

        improved = avg_loss < best_loss

        if (epoch + 1) % args.save_every == 0 or improved:
            ckpt_path = os.path.join(
                args.save_dir,
                "flow_epoch{:03d}_loss{:.4f}.pt".format(epoch, avg_loss),
            )
            torch.save(
                {
                    "epoch"       : epoch,
                    "flow_model"  : flow_model.state_dict(),
                    "spk_encoder" : spk_encoder.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "avg_loss"    : avg_loss,
                    "args"        : vars(args),
                },
                ckpt_path,
            )
            if improved:
                best_loss = avg_loss
                # best_train.pt：含优化器，用于续训
                torch.save(
                    {
                        "epoch"       : epoch,
                        "flow_model"  : flow_model.state_dict(),
                        "spk_encoder" : spk_encoder.state_dict(),
                        "optimizer"   : optimizer.state_dict(),
                        "avg_loss"    : avg_loss,
                        "args"        : vars(args),
                    },
                    os.path.join(args.save_dir, "best_train.pt"),
                )
                # best.pt：不含优化器，用于推理/评估
                torch.save(
                    {
                        "epoch"       : epoch,
                        "flow_model"  : flow_model.state_dict(),
                        "spk_encoder" : spk_encoder.state_dict(),
                        "avg_loss"    : avg_loss,
                        "args"        : vars(args),
                    },
                    os.path.join(args.save_dir, "best.pt"),
                )
            logger.info("已保存: {}".format(ckpt_path))

        if improved:
            no_improve = 0
        else:
            no_improve += 1
            logger.info("[EarlyStopping] 无改善 {}/{} 轮".format(no_improve, args.patience))
            if no_improve >= args.patience:
                logger.info("[EarlyStopping] 已连续 {} 轮无改善，停止训练".format(args.patience))
                break

    logger.info("训练完成。最优 loss={:.4f}".format(best_loss))
    logger.info("最优检查点: {}".format(os.path.join(args.save_dir, "best.pt")))


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TimbreFlow 阶段3 训练")

    # 路径
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path",   default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--split",       default="train-clean-100")
    p.add_argument("--save_dir",    default="output/flow_checkpoints_stage3")
    p.add_argument("--resume",      default="",
                   help="从阶段3 checkpoint 续训（含优化器状态，best_loss 自动重置）")
    p.add_argument("--resume_v2",   default="",
                   help="从阶段2 best.pt 热启动（只迁移权重，优化器重置）")

    # 模型结构（与阶段2保持一致，从 checkpoint 加载权重）
    p.add_argument("--base_ch",   type=int,   default=512)
    p.add_argument("--ch_mults",  type=int,   nargs="+", default=[1, 1, 2])
    p.add_argument("--cond_dim",  type=int,   default=512)
    p.add_argument("--spk_dim",   type=int,   default=256)
    p.add_argument("--time_dim",  type=int,   default=128)
    p.add_argument("--n_res",     type=int,   default=2)
    p.add_argument("--n_mid_res", type=int,   default=2)

    # 训练超参
    p.add_argument("--max_epochs",  type=int,   default=300)
    p.add_argument("--patience",    type=int,   default=30,
                   help="early stopping patience（阶段3 为 30，比阶段2 的 20 更宽松）")
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--segment_sec", type=float, default=4.0)
    p.add_argument("--lr",          type=float, default=2e-5,
                   help="阶段3 学习率，比阶段2 结束 LR 高一档以重新激活训练")
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log_every",   type=int,   default=20)
    p.add_argument("--save_every",  type=int,   default=5)

    # 损失参数
    p.add_argument("--lambda_stft", type=float, default=1.0,
                   help="多分辨率 STFT 损失总权重")
    p.add_argument("--lambda_sc",   type=float, default=0.5,
                   help="STFT 损失内谱收敛项权重（相对于 log-L1）")
    p.add_argument("--stft_every",  type=int,   default=2,
                   help="每隔多少步计算一次 STFT 损失（阶段3 为 2，阶段2 为 4）")

    # 低 N 噪声增强（分级）
    p.add_argument("--noise_std",     type=float, default=0.02,
                   help="N=3 的 latent 噪声标准差")
    p.add_argument("--noise_std_n2",  type=float, default=0.05,
                   help="N=2（1kbps）的 latent 噪声标准差，比 N=3 更大")

    # 1kbps 专攻
    p.add_argument("--low_rate_prob",    type=float, default=0.4,
                   help="采样时 N=2 的概率（原均匀分布仅 1/7≈14%%，默认提升到 40%%）")
    p.add_argument("--n2_extra_weight",  type=float, default=1.0,
                   help="N=2 在加权 flow loss 上的额外加成（叠加在基础权重 2.0x 上）")

    # 信道参数
    p.add_argument("--p_loss_values", type=float, nargs="+",
                   default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
