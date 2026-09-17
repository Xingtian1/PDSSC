# -*- coding: utf-8 -*-
"""
Rectified Flow 生成模型 阶段2 训练脚本（在 train_flow.py 基础上改进）

相对 train_flow.py（阶段1）的改进：
  1. 难样本加权损失：低 N（少层传输）的样本损失权重更高，缓解低码率质量崩溃
  2. 低 N 噪声增强：N<=2 时对 x_t 加高斯噪声，增强鲁棒性
  3. 多分辨率 STFT 损失：用 3 个尺度的频谱 L1 损失替代/补充单一 Mel 损失，
     直接优化短时谱一致性（对应 STOI 和 UTMOS 改善）
  4. n_emb 正态初始化：让模型从训练开始就感知码率差异
  5. 支持从阶段1 checkpoint（--resume_v1）热启动，继承 flow_model 和 spk_encoder 权重

用法（从头训练）:
  python scripts/train_flow_v2.py \\
      --config_path model_hub/speechtokenizer_hubert_avg/config.json \\
      --ckpt_path   model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt \\
      --data_dir    data \\
      --split       train-clean-100 \\
      --save_dir    output/flow_checkpoints_v2 \\
      --batch_size  8

用法（从阶段1 best.pt 热启动）:
  python scripts/train_flow_v2.py \\
      ... \\
      --resume_v1 output/flow_checkpoints/best.pt

用法（从阶段2 checkpoint 续训）:
  python scripts/train_flow_v2.py \\
      ... \\
      --resume output/flow_checkpoints_v2/best.pt
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
# 多分辨率 STFT 损失
# ─────────────────────────────────────────────────────────────────────────────

# (n_fft, hop_length, win_length)
_STFT_CONFIGS = [
    (512,  128, 512),
    (1024, 256, 1024),
    (2048, 512, 2048),
]



# 预建 Hann 窗（避免每步重建，设备无关，用时移到目标设备）
_HANN_WINDOWS = {}


def _get_hann_window(win_length, device):
    key = (win_length, str(device))
    if key not in _HANN_WINDOWS:
        _HANN_WINDOWS[key] = torch.hann_window(win_length, periodic=True, device=device)
    return _HANN_WINDOWS[key]


def multi_res_stft_loss(wav_pred, wav_ref):
    """
    多分辨率 STFT L1 损失（log 幅度谱）。
    wav_pred, wav_ref : (B, T) float32，值域 [-1, 1]
    返回标量 loss。
    """
    total = wav_pred.new_zeros(1).squeeze()   # 保留设备和 dtype，初始值 0
    for n_fft, hop, win in _STFT_CONFIGS:
        window = _get_hann_window(win, wav_pred.device)
        s_pred = torch.stft(
            wav_pred, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True,
        ).abs().clamp(min=1e-5)   # (B, F, T_frames)
        s_ref = torch.stft(
            wav_ref,  n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True,
        ).abs().clamp(min=1e-5)
        total = total + F.l1_loss(s_pred.log(), s_ref.log())
    return total / len(_STFT_CONFIGS)


# ─────────────────────────────────────────────────────────────────────────────
# 核心：批量获取 (latent_received, latent_8) 对（与阶段1相同，不做修改）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def prepare_training_batch(st_model, wavs, device, p_loss_values=None):
    if p_loss_values is None:
        p_loss_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    B = wavs.shape[0]

    codes_all = st_model.encode(wavs.to(device))
    latent_8  = st_model.quantizer.decode(codes_all)

    T_enc = codes_all.shape[2]
    D     = latent_8.shape[1]

    n_choices = torch.randint(1, 9, (B,))
    plr_vec   = torch.tensor(p_loss_values)[
        torch.randint(len(p_loss_values), (B,))
    ]

    def batch_decode_layer(layer_idx):
        vq_l = st_model.quantizer.vq.layers[layer_idx]
        flat = codes_all[layer_idx].reshape(1, B * T_enc)
        vecs = vq_l.decode(flat)
        bv   = vecs.squeeze(0)
        if bv.shape[0] == D:
            bv = bv.T
        return bv.reshape(B, T_enc, D).permute(0, 2, 1).contiguous()

    latent_received = batch_decode_layer(0)

    for l in range(1, 8):
        decoded      = batch_decode_layer(l)
        layer_active = (n_choices >= l + 1).float().to(device)
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

    # ── 阶段2 改进①：n_emb 正态初始化（让模型从训练开始感知码率差异）──
    nn.init.normal_(flow_model.n_emb.weight, std=0.02)
    logger.info("  n_emb 已用正态分布初始化（std=0.02）")

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
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = device.type == "cuda",
        drop_last        = True,
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 5,
        min_lr   = args.lr * 0.01,
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 恢复检查点 ────────────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")

    def _strip_compile_prefix(sd):
        """剥离 torch.compile 在 state_dict key 上添加的 '_orig_mod.' 前缀"""
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # 优先：从阶段2 checkpoint 续训（完整恢复，包含优化器状态）
    if args.resume and os.path.isfile(args.resume):
        logger.info("[Resume] 从阶段2 checkpoint 恢复: {}".format(args.resume))
        ckpt = torch.load(args.resume, map_location="cpu")
        flow_sd = _strip_compile_prefix(ckpt["flow_model"])
        missing, _ = flow_model.load_state_dict(flow_sd, strict=False)
        if missing:
            logger.info("[Resume] 新增参数（随机初始化）: {}".format(missing))
        spk_sd = _strip_compile_prefix(ckpt["spk_encoder"])
        spk_encoder.load_state_dict(spk_sd, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            # 用命令行指定的 LR 覆盖 checkpoint 里保存的旧 LR
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            logger.info("[Resume] ✓ 优化器状态已恢复，LR 重置为 {:.2e}".format(args.lr))
        else:
            logger.info("[Resume] ⚠ 无优化器状态（best.pt），使用新优化器，LR={:.2e}".format(args.lr))
        best_loss   = ckpt.get("avg_loss", float("inf"))  # 保留原 best_loss，避免第一个 epoch 覆盖更好的 ckpt
        start_epoch = ckpt["epoch"] + 1
        logger.info("[Resume] ✓ 从 epoch {} 继续，best_loss={:.4f}".format(start_epoch, best_loss))
        del ckpt

    # 其次：从阶段1 checkpoint 热启动（只迁移权重，优化器重置）
    elif args.resume_v1 and os.path.isfile(args.resume_v1):
        logger.info("[Resume-V1] 从阶段1 checkpoint 热启动: {}".format(args.resume_v1))
        ckpt    = torch.load(args.resume_v1, map_location="cpu")
        flow_sd = _strip_compile_prefix(ckpt["flow_model"])
        missing, unexpected = flow_model.load_state_dict(flow_sd, strict=False)
        if missing:
            logger.info("[Resume-V1] 新增参数（随机初始化）: {}".format(missing))
        if unexpected:
            logger.info("[Resume-V1] 忽略多余参数: {}".format(unexpected))
        spk_sd = _strip_compile_prefix(ckpt["spk_encoder"])
        spk_encoder.load_state_dict(spk_sd, strict=False)
        del ckpt
        # 热启动后重新初始化 n_emb（阶段1 是 zeros，阶段2 需要正态）
        nn.init.normal_(flow_model.n_emb.weight, std=0.02)
        logger.info("[Resume-V1] ✓ 权重迁移完成，n_emb 重新正态初始化，优化器重置")
        # start_epoch = 0，best_loss = inf（阶段2 重新计算）

    if hasattr(torch, "compile"):
        logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
        logging.getLogger("torch._inductor").setLevel(logging.WARNING)
        flow_model = torch.compile(flow_model)
        logger.info("torch.compile 已启用")

    # ── 训练循环 ──────────────────────────────────────────────────────
    # reduction='none' 以便按 N 加权
    loss_fn_none = nn.MSELoss(reduction="none")
    no_improve   = 0

    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        flow_model.train()
        spk_encoder.train()

        epoch_loss = 0.0
        t0 = time.time()

        for step, (wavs, ref_wavs) in enumerate(loader):
            # 1. 获取 (x0, x1, n_layers_batch)
            x0, x1, n_layers_batch = prepare_training_batch(
                st_model      = st_model,
                wavs          = wavs,
                device        = device,
                p_loss_values = args.p_loss_values,
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

            # 4. 目标速度场
            target = x1 - x0                              # (B, D, T)

            # 5. 前向预测
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                v_pred = flow_model(xt, t, spk_emb, n_layers_batch)   # (B, D, T)

                # ── 阶段2 改进③：难样本加权 Flow 损失 ──────────────
                # 只提升低 N 权重，高 N 保持 1.0（不惩罚 N=8 的高码率性能）
                # N=1→2.0, N=2→1.75, N=3→1.5, N=4→1.25, N>=5→1.0
                n_weight = ((9.0 - n_layers_batch.float()) / 4.0).clamp(min=1.0)  # (B,)
                n_weight = n_weight[:, None, None]                                 # (B,1,1)
                L_flow   = (loss_fn_none(v_pred, target) * n_weight).mean()

            # ── 阶段2 改进④：多分辨率 STFT 损失 ────────────────────
            # 每 stft_every 步计算一次（减少 Decoder 调用开销）
            L_stft         = torch.tensor(0.0, device=device)
            decoder_ran    = False   # 标记本步是否已经跑过 Decoder
            wav_pred_c     = None
            wav_ref_c      = None

            if args.lambda_stft > 0 and step % args.stft_every == 0:
                x1_pred = (x0 + v_pred).float()
                with torch.backends.cudnn.flags(enabled=False):
                    wav_pred = st_model.decoder(x1_pred)   # (B, 1, T_wav)
                wav_ref_raw = wavs.to(device).float()
                T_wav       = min(wav_pred.shape[-1], wav_ref_raw.shape[-1])
                wav_pred_c  = wav_pred[..., :T_wav].squeeze(1).clamp(-1.0, 1.0)
                wav_ref_c   = wav_ref_raw[..., :T_wav].squeeze(1)
                decoder_ran = True
                L_stft = multi_res_stft_loss(wav_pred_c, wav_ref_c)

            # NaN 防护
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

            if step % args.log_every == 0 and decoder_ran:
                lr_cur = optimizer.param_groups[0]["lr"]
                logger.info(
                    "[epoch {:3d} | step {:5d}/{}] loss={:.4f}  flow={:.4f}  stft={:.4f}  lr={:.2e}".format(
                        epoch, step, len(loader),
                        loss.item(), L_flow.item(), L_stft.item(), lr_cur
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
                # best_train.pt：含优化器状态，用于续训
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
    p = argparse.ArgumentParser(description="TimbreFlow 阶段2 训练")

    # 路径
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path",   default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--split",       default="train-clean-100")
    p.add_argument("--save_dir",    default="output/flow_checkpoints_v2",
                   help="阶段2 checkpoint 保存目录（与阶段1分开）")
    p.add_argument("--resume",      default="",
                   help="从阶段2 checkpoint 续训（完整恢复含优化器状态）")
    p.add_argument("--resume_v1",   default="",
                   help="从阶段1 best.pt 热启动（只迁移权重，优化器重置）")

    # 模型结构（默认比阶段1更大）
    p.add_argument("--base_ch",   type=int,   default=512)
    p.add_argument("--ch_mults",  type=int,   nargs="+", default=[1, 2, 4],
                   help="阶段2 默认 [1,2,4]（比阶段1 [1,1,2] 更深）")
    p.add_argument("--cond_dim",  type=int,   default=512)
    p.add_argument("--spk_dim",   type=int,   default=256)
    p.add_argument("--time_dim",  type=int,   default=128)
    p.add_argument("--n_res",     type=int,   default=3,
                   help="每块 ResBlock 数量，阶段2 默认 3（阶段1 为 2）")
    p.add_argument("--n_mid_res", type=int,   default=4,
                   help="瓶颈 ResBlock 数量，阶段2 默认 4（阶段1 为 2）")

    # 训练超参
    p.add_argument("--max_epochs",  type=int,   default=500)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=8,
                   help="模型更大，batch_size 默认 8（阶段1 为 12）")
    p.add_argument("--segment_sec", type=float, default=4.0)
    p.add_argument("--lr",          type=float, default=5e-5,
                   help="阶段2 学习率默认 5e-5（比阶段1 1e-4 小，微调用）")
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log_every",   type=int,   default=20)
    p.add_argument("--save_every",  type=int,   default=5)

    # 阶段2 新增损失参数
    p.add_argument("--lambda_stft", type=float, default=1.0,
                   help="多分辨率 STFT 损失权重（阶段2 主力感知损失）")
    p.add_argument("--stft_every",  type=int,   default=4,
                   help="每隔多少步计算一次 STFT 损失")
    p.add_argument("--lambda_mel",  type=float, default=0.0,
                   help="Mel 谱损失权重（阶段2 默认关闭，由 STFT 覆盖）")
    p.add_argument("--mel_every",   type=int,   default=4)

    # 信道参数
    p.add_argument("--p_loss_values", type=float, nargs="+",
                   default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
