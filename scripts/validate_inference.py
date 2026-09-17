"""
验证 SpeechTokenizer 推理流程
对比 4层 RVQ vs 8层 RVQ 的重建质量

评估指标:
  PESQ  - 感知语音质量 (ITU-T P.862)，范围 [-0.5, 4.5]，越高越好
  STOI  - 短时客观可懂度，范围 [0, 1]，越高越好

依赖安装:
  pip install pesq pystoi

用法:
  python scripts/validate_inference.py \
      --config_path  model_hub/speechtokenizer_hubert_avg/config.json \
      --ckpt_path    model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt \
      --data_dir     data \
      --split        test-clean \
      --num_samples  20 \
      --output_dir   output/validate
"""

import os
import argparse
import random
import json

import torch
import torchaudio
import soundfile as sf
import numpy as np
from scipy.io.wavfile import write as wav_write
from pesq import pesq
from pystoi import stoi

from speechtokenizer import SpeechTokenizer


# ─────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────

def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """
    PESQ (Perceptual Evaluation of Speech Quality)
    ITU-T P.862.2 宽带模式，要求 sr=16000
    返回 MOS-LQO 分数，范围 [-0.5, 4.5]
    """
    try:
        return pesq(sr, ref, deg, "wb")
    except Exception as e:
        print(f"  [PESQ 计算失败] {e}")
        return float("nan")


def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """
    STOI (Short-Time Objective Intelligibility)
    范围 [0, 1]，衡量语音可懂度
    """
    try:
        return stoi(ref, deg, sr, extended=False)
    except Exception as e:
        print(f"  [STOI 计算失败] {e}")
        return float("nan")


def latent_relative_error(latent_4: torch.Tensor, latent_8: torch.Tensor) -> float:
    """Q5-Q8 残差在 latent 空间的相对能量: ||latent_8 - latent_4|| / ||latent_8||"""
    diff = latent_8 - latent_4
    return (diff.norm() / (latent_8.norm() + 1e-8)).item()


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_wav_list(data_dir: str, split: str) -> list:
    list_path = os.path.join(data_dir, f"{split}_files.txt")
    if os.path.exists(list_path):
        with open(list_path) as f:
            return [l.strip() for l in f if l.strip()]
    split_dir = os.path.join(data_dir, "LibriSpeech", split)
    files = []
    for root, _, fnames in os.walk(split_dir):
        for fn in sorted(fnames):
            if fn.endswith(".flac") or fn.endswith(".wav"):
                files.append(os.path.join(root, fn))
    return sorted(files)


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """返回 (1, T)，单声道，已重采样"""
    audio, sr = sf.read(path, dtype='float32')
    wav = torch.from_numpy(audio).T  # (channels, T)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav[:1, :]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


# ─────────────────────────────────────────────
# 核心推理
# ─────────────────────────────────────────────

def run_encode_decode(model: SpeechTokenizer, wav: torch.Tensor, device: str):
    """
    返回:
      wav_4    : 4层 RVQ 解码音频  (1, T)
      wav_8    : 8层 RVQ 解码音频  (1, T)
      latent_4 : 前4层反量化之和  (1, D, T_enc)
      latent_8 : 全8层反量化之和  (1, D, T_enc)
    """
    x = wav.unsqueeze(0).to(device)  # (1, 1, T)
    with torch.no_grad():
        codes    = model.encode(x)              # (n_q, 1, T_enc)
        latent_8 = model.quantizer.decode(codes)       # (1, D, T_enc)
        latent_4 = model.quantizer.decode(codes[:4])   # (1, D, T_enc)
        wav_8    = model.decode(codes)                 # (1, 1, T)
        wav_4    = model.decode(codes[:4])             # (1, 1, T)
    return wav_4.cpu(), wav_8.cpu(), latent_4.cpu(), latent_8.cpu()


def align_and_to_numpy(ref, deg4, deg8):
    """对齐长度，转 numpy float64（PESQ/STOI 要求）"""
    min_len = min(ref.shape[-1], deg4.shape[-1], deg8.shape[-1])
    ref  = ref[..., :min_len].squeeze().numpy().astype(np.float64)
    deg4 = deg4[..., :min_len].squeeze().numpy().astype(np.float64)
    deg8 = deg8[..., :min_len].squeeze().numpy().astype(np.float64)
    return ref, deg4, deg8


# ─────────────────────────────────────────────
# 保存音频
# ─────────────────────────────────────────────

def save_wav(path: str, audio: np.ndarray, sr: int):
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    wav_write(path, sr, audio)


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("=" * 70)
    print("加载 SpeechTokenizer 模型...")
    model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    sr = model.sample_rate
    bitrate_4 = (sr // model.downsample_rate) * 10 * 4 / 1000
    bitrate_8 = (sr // model.downsample_rate) * 10 * 8 / 1000
    print(f"  设备: {device}  |  采样率: {sr} Hz  |  RVQ层数: {model.n_q}")
    print(f"  码率:  4层 = {bitrate_4:.1f} kbps    8层 = {bitrate_8:.1f} kbps")

    # 文件列表
    all_files = load_wav_list(args.data_dir, args.split)
    if not all_files:
        raise FileNotFoundError(
            f"未找到 {args.split} 的音频文件，请先运行:\n"
            "  python scripts/download_data.py"
        )
    random.seed(42)
    selected = random.sample(all_files, min(args.num_samples, len(all_files)))
    print(f"\n共 {len(all_files)} 个文件，随机选取 {len(selected)} 个\n")

    # 表头
    print("=" * 70)
    print(f"{'文件':<32} {'PESQ-4':>8} {'PESQ-8':>8} {'STOI-4':>8} {'STOI-8':>8} {'latent误差':>10}")
    print("-" * 70)

    results = []
    for i, fpath in enumerate(selected):
        fname = os.path.basename(fpath)
        wav = load_audio(fpath, sr)

        # 截断最大处理时长
        max_samples = int(args.max_sec * sr)
        wav = wav[:, :max_samples]

        wav_4, wav_8, latent_4, latent_8 = run_encode_decode(model, wav, device)

        ref, deg4, deg8 = align_and_to_numpy(wav, wav_4, wav_8)

        pesq_4 = compute_pesq(ref, deg4, sr)
        pesq_8 = compute_pesq(ref, deg8, sr)
        stoi_4 = compute_stoi(ref, deg4, sr)
        stoi_8 = compute_stoi(ref, deg8, sr)
        lat_err = latent_relative_error(latent_4, latent_8)

        results.append({
            "file":   fname,
            "pesq_4": pesq_4,
            "pesq_8": pesq_8,
            "stoi_4": stoi_4,
            "stoi_8": stoi_8,
            "latent_relative_error": lat_err,
        })

        print(f"{fname:<32} {pesq_4:>8.3f} {pesq_8:>8.3f} {stoi_4:>8.4f} {stoi_8:>8.4f} {lat_err:>10.4f}")

        # 保存前 N 条音频
        if i < args.save_samples:
            stem = os.path.splitext(fname)[0]
            save_wav(os.path.join(args.output_dir, f"{stem}_original.wav"), ref.astype(np.float32), sr)
            save_wav(os.path.join(args.output_dir, f"{stem}_4layer.wav"),  deg4.astype(np.float32), sr)
            save_wav(os.path.join(args.output_dir, f"{stem}_8layer.wav"),  deg8.astype(np.float32), sr)

    # 汇总统计
    print("\n" + "=" * 70)
    print("汇总统计（均值 ± 标准差）")
    print("=" * 70)

    def stats(key):
        vals = [r[key] for r in results if not np.isnan(r[key])]
        return np.mean(vals), np.std(vals)

    pesq4_mean, pesq4_std = stats("pesq_4")
    pesq8_mean, pesq8_std = stats("pesq_8")
    stoi4_mean, stoi4_std = stats("stoi_4")
    stoi8_mean, stoi8_std = stats("stoi_8")
    late_mean,  late_std  = stats("latent_relative_error")

    print(f"  {'指标':<20} {'4层 (2kbps)':>18} {'8层 (4kbps)':>18} {'差距':>10}")
    print(f"  {'-'*68}")
    print(f"  {'PESQ':<20} {pesq4_mean:>8.3f} ±{pesq4_std:.3f}   {pesq8_mean:>8.3f} ±{pesq8_std:.3f}   {pesq8_mean-pesq4_mean:>+8.3f}")
    print(f"  {'STOI':<20} {stoi4_mean:>8.4f} ±{stoi4_std:.4f}   {stoi8_mean:>8.4f} ±{stoi8_std:.4f}   {stoi8_mean-stoi4_mean:>+8.4f}")
    print(f"  {'Latent 相对误差':<20} {late_mean:>8.4f} ±{late_std:.4f}")

    print(f"""
说明:
  PESQ: 感知语音质量 [-0.5, 4.5]，>3.5 为优秀，>2.5 为良好
  STOI: 语音可懂度  [0, 1]，  >0.9 为高可懂度
  Latent 相对误差: Flow 模型需要补全的 latent 能量占比
  →  Flow 模型目标: 将 4层重建的 PESQ/STOI 提升到接近 8层水平
""")

    # 保存结果
    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"详细结果: {out_json}")
    print(f"重建音频: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",  default="model_hub/speechtokenizer_hubert_avg/config.json")
    parser.add_argument("--ckpt_path",    default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    parser.add_argument("--data_dir",     default="data")
    parser.add_argument("--split",        default="test-clean")
    parser.add_argument("--num_samples",  type=int,   default=20)
    parser.add_argument("--save_samples", type=int,   default=5)
    parser.add_argument("--max_sec",      type=float, default=10.0)
    parser.add_argument("--output_dir",   default="output/validate")
    args = parser.parse_args()
    main(args)
