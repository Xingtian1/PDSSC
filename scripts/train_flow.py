"""
Rectified Flow 生成模型训练脚本

训练目标（重要）:
  x_0 = latent_received  (接收端 partial latent，信道损伤后插值恢复)
  x_1 = latent_8         (完整 8 层反量化之和，目标)
  x_t = (1-t)*x_0 + t*x_1
  loss = || v_θ(x_t, t, spk_emb) - (x_1 - x_0) ||^2

推理时：从 x_0 出发，沿预测速度场以欧拉法积分到 x_1，再用 Decoder 重建音频。

用法:
  python scripts/train_flow.py \\
      --config_path model_hub/speechtokenizer_hubert_avg/config.json \\
      --ckpt_path   model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt \\
      --data_dir    data \\
      --split       train-clean-100 \\
      --save_dir    output/flow_checkpoints \\
      --batch_size  4 \\
      --max_epochs  50
"""

import os
import json
import argparse
import random
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TAT
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """将变长音频 pad 到最大长度后组批。"""
    wavs     = [item[0] for item in batch]
    ref_wavs = [item[1] for item in batch]
    # 数据集已经统一 segment_len，直接 stack
    wavs     = torch.stack(wavs,     dim=0)  # (B, 1, T)
    ref_wavs = torch.stack(ref_wavs, dim=0)  # (B, 1, T)
    return wavs, ref_wavs


# ─────────────────────────────────────────────────────────────────────────────
# 核心：批量获取 (latent_received, latent_8) 对
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def prepare_training_batch(
    st_model      : SpeechTokenizer,
    wavs          : torch.Tensor,    # (B, 1, T)
    device        : torch.device,
    p_loss_values : list = None,     # 离散 PLR 列表，如 [0.05, 0.10, ..., 0.30]
) -> tuple:
    """
    全向量化信道模拟，无 Python 逐样本循环：
      1. 批量 encode 一次获得全部 codes
      2. 每样本从离散 PLR 列表中随机选一个，生成 Bernoulli 丢包掩码
      3. 批量 decode 所有层（一次 reshape+decode，无逐帧循环）

    返回: (latent_received, latent_8), 形状均为 (B, D, T_enc)
    """
    if p_loss_values is None:
        p_loss_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    B = wavs.shape[0]

    # ── 批量 encode ──────────────────────────────────────────────────
    codes_all = st_model.encode(wavs.to(device))      # (8, B, T_enc)
    latent_8  = st_model.quantizer.decode(codes_all)  # (B, D, T_enc)

    T_enc = codes_all.shape[2]
    D     = latent_8.shape[1]

    # ── 每样本随机选传输层数 N（1~8）和丢包率 ──────────────────────
    n_choices  = torch.randint(2, 9, (B,))            # N ~ U[2,8]，每样本独立（最低1.0kbps）
    plr_vec    = torch.tensor(p_loss_values)[
        torch.randint(len(p_loss_values), (B,))
    ]                                                  # PLR 离散均匀采样

    # ── 批量解码单层 ─────────────────────────────────────────────────
    def batch_decode_layer(layer_idx: int) -> torch.Tensor:
        """返回 (B, D, T_enc)，在 device 上"""
        vq_l  = st_model.quantizer.vq.layers[layer_idx]
        flat  = codes_all[layer_idx].reshape(1, B * T_enc)
        vecs  = vq_l.decode(flat)
        bv    = vecs.squeeze(0)
        if bv.shape[0] == D:
            bv = bv.T
        return bv.reshape(B, T_enc, D).permute(0, 2, 1).contiguous()  # (B, D, T_enc)

    # ── Q1 始终可靠传输 ───────────────────────────────────────────────
    latent_received = batch_decode_layer(0)

    # ── Q2..Q8：按各样本的 N 决定是否传输，传输的层按 PLR 随机丢包 ──
    for l in range(1, 8):
        decoded = batch_decode_layer(l)                          # (B, D, T_enc)
        # 该层是否被传输：n_choices[b] >= l+1
        layer_active = (n_choices >= l + 1).float().to(device)  # (B,)
        # 传输帧是否收到：Bernoulli(1 - plr)
        recv_mask = (torch.rand(B, T_enc) >= plr_vec.unsqueeze(1)).float().to(device)  # (B, T_enc)
        # 只有 active 且未丢包的帧才累加
        recv = (layer_active.unsqueeze(1) * recv_mask).unsqueeze(1)  # (B, 1, T_enc)
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
    logger.info(f"设备: {device}")

    # ── 加载 SpeechTokenizer（冻结）──────────────────────────────────
    logger.info("加载 SpeechTokenizer...")
    st_model = SpeechTokenizer.load_from_checkpoint(
        args.config_path, args.ckpt_path
    )
    st_model.eval().to(device)
    for p in st_model.parameters():
        p.requires_grad_(False)

    # latent 维度（从模型配置读取）
    latent_dim = st_model.quantizer.dimension
    logger.info(f"  latent_dim = {latent_dim}")

    # ── 说话人编码器（预训练 ECAPA-TDNN，冻结骨干）───────────────────
    logger.info("加载预训练说话人编码器（resemblyzer GE2E）...")
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
    logger.info(f"  FlowModel  参数量: {n_params:.1f}M")
    logger.info(f"  SpeakerEnc 参数量: {n_spk:.1f}M")

    # ── Mel 谱损失用的变换（固定，不参与训练）────────────────────────
    mel_transform = TAT.MelSpectrogram(
        sample_rate = st_model.sample_rate,
        n_fft       = 1024,
        hop_length  = 256,
        n_mels      = 80,
        f_min       = 0.0,
        f_max       = 8000.0,
    ).to(device)

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
        drop_last   = True,
    )

    # ── 优化器（Flow模型全参数 + 说话人编码器投影层）─────────────────
    # spk_encoder.backbone 已冻结，只有 spk_encoder.proj 参与训练
    params = list(flow_model.parameters()) + list(spk_encoder.proj.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr           = args.lr,
        betas        = (0.9, 0.99),
        weight_decay = 1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode    = "min",
        factor  = 0.5,
        patience= 5,
        min_lr  = args.lr * 0.01,
    )

    # ── AMP（混合精度）────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 恢复检查点
    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            logger.warning(f"[Resume] 检查点文件不存在: {args.resume}，从头开始训练")
        else:
            logger.info(f"[Resume] 正在加载检查点: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            missing, unexpected = flow_model.load_state_dict(ckpt["flow_model"], strict=False)
            if missing:
                logger.info(f"[Resume] 新增参数（随机初始化）: {missing}")
            logger.info(f"[Resume] ✓ Flow 模型权重加载成功")
            spk_encoder.load_state_dict(ckpt["spk_encoder"], strict=False)
            logger.info(f"[Resume] ✓ 说话人编码器权重加载成功")
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                logger.info(f"[Resume] ✓ 优化器状态加载成功")
            else:
                logger.info(f"[Resume] ⚠ 检查点无优化器状态（best.pt），使用新优化器，LR={args.lr:.2e}")
            saved_loss = ckpt.get("avg_loss", float("nan"))
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"[Resume] ✓ 从 epoch {start_epoch} 继续训练（上次保存时 avg_loss={saved_loss:.4f}）")

    # ── 训练循环 ──────────────────────────────────────────────────────
    loss_fn      = nn.MSELoss()
    best_loss    = float("inf")
    no_improve   = 0   # early stopping 计数器

    # 恢复 best_loss，防止 resume 时覆盖更优的 best.pt
    if args.resume and os.path.isfile(args.resume):
        _ckpt = torch.load(args.resume, map_location="cpu")
        best_loss = _ckpt.get("avg_loss", float("inf"))
        del _ckpt
        logger.info(f"[Resume] best_loss 恢复为 {best_loss:.4f}")

    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        flow_model.train()
        spk_encoder.train()

        epoch_loss = 0.0
        t0 = time.time()

        for step, (wavs, ref_wavs) in enumerate(loader):
            # 1. 获取 (latent_received=x_0, latent_8=x_1, n_layers_batch)
            x0, x1, n_layers_batch = prepare_training_batch(
                st_model      = st_model,
                wavs          = wavs,
                device        = device,
                p_loss_values = args.p_loss_values,
            )
            n_layers_batch = n_layers_batch.to(device)

            # 2. 说话人嵌入（从参考音频提取）
            ref = ref_wavs.to(device)                 # (B, 1, T)
            spk_emb = spk_encoder(ref.squeeze(1))     # (B, spk_dim)

            # 3. Rectified Flow 插值
            B = x0.shape[0]
            t = torch.rand(B, device=device)          # t ~ U[0,1]
            t_bc = t[:, None, None]                   # (B,1,1) for broadcasting
            xt = (1 - t_bc) * x0 + t_bc * x1         # (B, D, T)

            # 4. 目标速度场
            target = x1 - x0                          # (B, D, T)

            # 5. 模型预测（AMP autocast）
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                v_pred = flow_model(xt, t, spk_emb, n_layers_batch)   # (B, D, T)
                L_flow = loss_fn(v_pred, target)

            # ── Mel 谱损失（辅，每 mel_every 步计算一次）────────────
            # 梯度穿过冻结 Decoder 反传到 Flow 模型，需在 autocast 外算
            L_mel = torch.tensor(0.0, device=device)
            if args.lambda_mel > 0 and step % args.mel_every == 0:
                x1_pred = (x0 + v_pred).float()               # (B, D, T)
                with torch.backends.cudnn.flags(enabled=False):
                    wav_pred = st_model.decoder(x1_pred)       # (B, 1, T_wav)
                wav_ref  = wavs.to(device).float()

                T_wav = min(wav_pred.shape[-1], wav_ref.shape[-1])
                wav_pred_c = wav_pred[..., :T_wav].squeeze(1).clamp(-1.0, 1.0)
                mel_pred = mel_transform(wav_pred_c)
                mel_ref  = mel_transform(wav_ref[..., :T_wav].squeeze(1))
                T_mel    = min(mel_pred.shape[-1], mel_ref.shape[-1])
                L_mel    = F.l1_loss(
                    mel_pred[..., :T_mel].clamp(min=1e-5, max=1e4).log(),
                    mel_ref[..., :T_mel].clamp(min=1e-5, max=1e4).log(),
                )

            # guard mel loss separately — prevents NaN from propagating to flow loss
            if not torch.isfinite(L_mel):
                L_mel = torch.tensor(0.0, device=device)

            loss = L_flow + args.lambda_mel * L_mel

            # Always do backward + scaler.update() so GradScaler can track
            # inf/nan and correctly reduce its scale factor.
            # (Skipping scaler.update() causes scale to grow unboundedly → collapse)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm=1.0)

            if not torch.isfinite(loss):
                logger.warning(f"[step {step}] loss=nan/inf，跳过参数更新")
                optimizer.zero_grad()
                scaler.update()   # ← critical: lets scaler reduce scale factor
                continue

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % args.log_every == 0:
                lr_cur = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[epoch {epoch:3d} | step {step:5d}/{len(loader)}] "
                    f"loss={loss.item():.4f}  "
                    f"flow={L_flow.item():.4f}  mel={L_mel.item():.4f}  "
                    f"lr={lr_cur:.2e}"
                )

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        scheduler.step(avg_loss)
        lr_cur = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d} 完成 | avg_loss={avg_loss:.4f} | lr={lr_cur:.2e} | 耗时={elapsed:.0f}s"
        )

        # 判断是否有改善（必须在更新 best_loss 之前）
        improved = avg_loss < best_loss

        # 保存检查点
        if (epoch + 1) % args.save_every == 0 or improved:
            ckpt_path = os.path.join(
                args.save_dir,
                f"flow_epoch{epoch:03d}_loss{avg_loss:.4f}.pt",
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
            # 同时保存最新 best
            if improved:
                best_loss = avg_loss
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
            logger.info(f"已保存: {ckpt_path}")

        # Early stopping
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            logger.info(f"[EarlyStopping] 无改善 {no_improve}/{args.patience} 轮")
            if no_improve >= args.patience:
                logger.info(f"[EarlyStopping] 已连续 {args.patience} 轮无改善，停止训练")
                break

    logger.info(f"训练完成。最优 loss={best_loss:.4f}")
    logger.info(f"最优检查点: {os.path.join(args.save_dir, 'best.pt')}")


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # 路径
    p.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path",   default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--split",       default="train-clean-100")
    p.add_argument("--save_dir",    default="output/flow_checkpoints")
    p.add_argument("--resume",      default="",    help="恢复训练的检查点路径")

    # 模型结构
    p.add_argument("--base_ch",   type=int,   default=512)
    p.add_argument("--ch_mults",  type=int,   nargs="+", default=[1, 1, 2])
    p.add_argument("--cond_dim",  type=int,   default=512)
    p.add_argument("--spk_dim",   type=int,   default=256)
    p.add_argument("--time_dim",  type=int,   default=128)
    p.add_argument("--n_res",     type=int,   default=2)
    p.add_argument("--n_mid_res", type=int,   default=2)

    # 训练超参
    p.add_argument("--max_epochs",  type=int,   default=500,
                   help="最大训练轮数（配合 --patience 实现早停）")
    p.add_argument("--patience",    type=int,   default=20,
                   help="early stopping：连续多少轮 val loss 无改善则停止")
    p.add_argument("--batch_size",  type=int,   default=12,
                   help="A40 48GB 推荐 12，显存不足时降到 8")
    p.add_argument("--segment_sec", type=float, default=4.0,
                   help="训练时裁剪片段长度（秒）")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log_every",   type=int,   default=20)
    p.add_argument("--save_every",  type=int,   default=5)

    # 损失权重
    p.add_argument("--lambda_mel",   type=float, default=0.1,
                   help="Mel 谱 L1 损失权重（0 表示关闭）")
    p.add_argument("--mel_every",    type=int,   default=4,
                   help="每隔多少步计算一次 Mel 损失（减少 Decoder 调用开销）")

    # 信道参数（训练时离散随机化）
    p.add_argument("--p_loss_values", type=float, nargs="+",
                   default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                   help="训练用的离散 PLR 列表（含 0.0），每个 batch 样本随机选一个")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
