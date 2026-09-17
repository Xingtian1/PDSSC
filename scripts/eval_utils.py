"""
eval_utils.py  ─  评估脚本公共工具

被 eval_flow.py / eval_rate.py / eval_plr.py 共用。
"""

import os
import random
import subprocess
import tempfile
import io
import warnings
import contextlib
from typing import Optional

import numpy as np
import torch
import soundfile as sf

# ── ViSQOL Python API ────────────────────────────────────────────────────────
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _try_load_utmos_local():
    """
    Prefer a local SpeechMOS checkout to avoid torch.hub / GitHub / SSL issues.

    Expected local layout examples:
      <repo>/external/SpeechMOS
      <repo>/SpeechMOS
      <repo>/third_party/SpeechMOS
    """
    import sys as _sys

    candidate_dirs = [
        os.path.join(_ROOT_DIR, "external", "SpeechMOS"),
        os.path.join(_ROOT_DIR, "SpeechMOS"),
        os.path.join(_ROOT_DIR, "third_party", "SpeechMOS"),
    ]
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        if d not in _sys.path:
            _sys.path.insert(0, d)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
                    category=FutureWarning,
                )
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    from utmos import create_model as _create_utmos_model
                    model = _create_utmos_model("utmos22_strong")
            model.eval()
            print(f"[eval_utils] UTMOS loaded from local SpeechMOS repo: {d}")
            return model
        except Exception as e:
            print(f"[eval_utils] local SpeechMOS load failed at {d}: {e}")
    return None


try:
    from visqol import VisqolApi as _VisqolApiClass
    HAS_VISQOL = True
except Exception as _ve:
    _VisqolApiClass = None
    HAS_VISQOL = False
    print(f"[eval_utils] visqol-python 未安装，VISQoL 将输出 nan。({_ve})")

_utmos_model = None
HAS_UTMOS = False

try:
    from encodec import EncodecModel
    HAS_ENCODEC = True
except ImportError:
    HAS_ENCODEC = False

try:
    import sys as _sys
    _plcmos_dir = os.path.join(_ROOT_DIR, "plcmos")
    if _plcmos_dir not in _sys.path:
        _sys.path.insert(0, _plcmos_dir)
    from plc_mos import PLCMOSEstimator as _PLCMOSEstimator
    _plcmos_instance = None
    HAS_PLCMOS = True
except ImportError:
    _plcmos_instance = None
    HAS_PLCMOS = False
    print("[eval_utils] plcmos not found; PLCMOS 将输出 nan。请确认 plcmos/plc_mos.py 存在。")

_encodec_model_cache: dict = {}   # (source_bw, device_str) -> model


# =============================================================================
# 基础工具
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nanmean(vals):
    v = [x for x in vals if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """返回 (1, T) float32 tensor。"""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    wav = torch.from_numpy(audio).unsqueeze(0)
    if sr != target_sr:
        import torchaudio.functional as TAF
        wav = TAF.resample(wav, sr, target_sr)
    return wav


def load_filelist(data_dir: str, split: str) -> list:
    list_path = os.path.join(data_dir, f"{split}_files.txt")
    if os.path.exists(list_path):
        with open(list_path) as f:
            return [l.strip() for l in f if l.strip()]
    split_dir = os.path.join(data_dir, "LibriSpeech", split)
    if not os.path.isdir(split_dir):
        split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        split_dir = data_dir
    files = []
    for root, _, fnames in os.walk(split_dir):
        for fn in sorted(fnames):
            if fn.endswith(".flac") or fn.endswith(".wav"):
                files.append(os.path.join(root, fn))
    return sorted(files)


def build_spk2files(files: list) -> dict:
    from collections import defaultdict
    spk2files = defaultdict(list)
    for f in files:
        parts = f.replace("\\", "/").split("/")
        for i, p in enumerate(parts):
            if p in ("test-clean", "train-clean-100", "dev-clean"):
                try:
                    spk2files[parts[i + 1]].append(f)
                except IndexError:
                    pass
                break
    return dict(spk2files)


def pick_ref_wav(fpath: str, spk2files: dict, sr: int) -> torch.Tensor:
    """选同说话人另一条音频作为参考（找不到则用自身）。返回 (1, T)。"""
    parts  = fpath.replace("\\", "/").split("/")
    spk_id = None
    for i, p in enumerate(parts):
        if p in ("test-clean", "train-clean-100", "dev-clean"):
            try:
                spk_id = parts[i + 1]
            except IndexError:
                pass
            break
    if spk_id and spk_id in spk2files and len(spk2files[spk_id]) > 1:
        ref_path = fpath
        while ref_path == fpath:
            ref_path = random.choice(spk2files[spk_id])
    else:
        ref_path = fpath
    return load_audio(ref_path, sr)


# =============================================================================
# 指标计算
# =============================================================================

def _visqol_one(args_tuple):
    """Worker function for parallel VISQoL computation (picklable top-level)."""
    ref_np, deg_np, sr, mode = args_tuple
    ref_path = deg_path = None
    try:
        import soundfile as _sf
        from visqol import VisqolApi
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _sf.write(f.name, ref_np, sr); ref_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _sf.write(f.name, deg_np, sr); deg_path = f.name
        api = VisqolApi()
        api.create(mode=mode)
        result = api.measure(ref_path, deg_path)
        return float(result.moslqo)
    except Exception:
        return float("nan")
    finally:
        for p in [ref_path, deg_path]:
            if p and os.path.exists(p):
                try: os.unlink(p)
                except OSError: pass


def _visqol_one_debug(args_tuple):
    """Single VISQoL call that also returns error text for diagnostics."""
    ref_np, deg_np, sr, mode = args_tuple
    ref_path = deg_path = None
    try:
        import soundfile as _sf
        from visqol import VisqolApi
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _sf.write(f.name, ref_np, sr); ref_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _sf.write(f.name, deg_np, sr); deg_path = f.name
        api = VisqolApi()
        api.create(mode=mode)
        result = api.measure(ref_path, deg_path)
        return float(result.moslqo), ""
    except Exception as e:
        return float("nan"), f"{type(e).__name__}: {e}"
    finally:
        for p in [ref_path, deg_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def calc_visqol(ref_np: np.ndarray, deg_np: np.ndarray, sr: int, mode: str = "speech") -> float:
    """ViSQOL MOS-LQO, single call."""
    if not HAS_VISQOL:
        return float("nan")
    n = min(len(ref_np), len(deg_np))
    return _visqol_one((ref_np[:n], deg_np[:n], sr, mode))


def calc_visqol_batch(ref_list, deg_list, sr: int, n_workers: int = 8, mode: str = "speech") -> list:
    """Parallel VISQoL for a list of (ref, deg) pairs."""
    if not HAS_VISQOL:
        return [float("nan")] * len(ref_list)

    pairs = []
    for ref_np, deg_np in zip(ref_list, deg_list):
        n = min(len(ref_np), len(deg_np))
        pairs.append((ref_np[:n], deg_np[:n], sr, mode))

    if not pairs:
        return []

    n_workers = max(1, min(int(n_workers), len(pairs)))
    # Avoid potential fork hangs when CUDA context is already initialized.
    if n_workers > 1 and torch.cuda.is_available() and torch.cuda.is_initialized():
        print("[eval_utils] CUDA context detected; forcing VISQOL workers=1 to avoid multiprocessing fork hang.")
        n_workers = 1
    if n_workers == 1:
        results = []
        total = len(pairs)
        for i, p in enumerate(pairs, 1):
            results.append(_visqol_one(p))
            if total >= 20 and (i == 1 or i == total or i % 10 == 0):
                print(f"[eval_utils] VISQOL single-process progress: {i}/{total}", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_visqol_one, pairs))
        except Exception as e:
            print(f"[eval_utils] VISQoL ProcessPool failed, fallback to single-process: {e}")
            results = [_visqol_one(p) for p in pairs]

    if results and all(np.isnan(v) for v in results):
        _, err = _visqol_one_debug(pairs[0])
        if err:
            print(f"[eval_utils] VISQoL all-NaN diagnostic: {err}")
    return results


def calc_utmos(wav_np: np.ndarray, sr: int) -> float:
    """UTMOS22 strong learner predicted MOS, range ~1-5.
    Model forward: (B, T), sr -> (B,)
    """
    if not HAS_UTMOS:
        return float("nan")
    try:
        wav_t = torch.from_numpy(wav_np.astype(np.float32)).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            score = _utmos_model(wav_t, sr)   # handles resampling internally
        return float(score.mean().item())
    except Exception:
        return float("nan")


# =============================================================================
# 信道仿真：N 层传输 + Bernoulli zero-PLC
# =============================================================================

@torch.no_grad()
def channel_simulate(st_model, wav: torch.Tensor, n_layers: int,
                     p_loss: float, device) -> dict:
    """
    语义交织信道仿真：与训练 prepare_training_batch 保持一致。

    设计：
      Q1（RVQ 第 1 层）：逐帧 Bernoulli 丢包，丢失帧用相邻帧 latent 线性插值恢复。
      Q2..QN           ：各层独立逐帧 Bernoulli 丢包，丢失帧置零。
      Q(N+1)..Q8       ：主动丢弃（降码率），由 Flow 补全。

    返回：
      latent_ch : Q1（已插值恢复）+ Q2..QN（部分丢包后）之和，作为 Flow 输入
      latent_8  : 8 层完整 latent（质量上界）
      wav_8     : 8 层解码音频 (1,T) cpu tensor
    """
    x         = wav.unsqueeze(0).to(device)   # (1, 1, T)
    codes_all = st_model.encode(x)             # (8, 1, T_enc)
    T_enc     = codes_all.shape[2]
    D         = st_model.quantizer.dimension

    def safe_decode(idx) -> torch.Tensor:
        """返回 (1, D, T_enc)"""
        vq_l = st_model.quantizer.vq.layers[idx]
        d    = vq_l.decode(codes_all[idx])
        if d.shape[-1] == D:
            d = d.permute(0, 2, 1)
        return d.contiguous()

    # ── Q1：逐帧 Bernoulli 丢包，丢失帧用相邻帧 latent 线性插值恢复 ──
    latent_q1 = safe_decode(0)                              # (1, D, T_enc)
    q1_recv   = (torch.rand(T_enc) >= p_loss)               # (T_enc,) bool
    if not q1_recv.all():
        latent_q1 = _latent_linear_interp(latent_q1, q1_recv)

    latent_ch = latent_q1.clone()

    # ── Q2..QN：各层独立逐帧 Bernoulli 丢包，丢失帧置零 ────────────
    for l in range(1, n_layers):
        decoded   = safe_decode(l)
        recv_mask = (torch.rand(T_enc) >= p_loss).float().to(device)
        latent_ch = latent_ch + decoded * recv_mask.unsqueeze(0).unsqueeze(0)

    latent_8 = st_model.quantizer.decode(codes_all)

    return {
        "latent_ch" : latent_ch,
        "latent_8"  : latent_8,
        "wav_8"     : st_model.decoder(latent_8).squeeze(0).cpu(),
    }


# =============================================================================
# 竞品仿真：编解码 + 信道模型
# =============================================================================

def _fec_residual_plr(plr: float, fec_overhead: float) -> float:
    """
    理想删除纠错码（Ideal Erasure Code）模型。

    fec_overhead = fec_bw / total_bw  （冗余比例，0~1）
    - 若 PLR ≤ fec_overhead：FEC 完全恢复，有效 PLR = 0
    - 若 PLR > fec_overhead：线性残余 PLR = PLR - fec_overhead

    该模型对竞品偏乐观（实际 FEC 性能通常更差），保证对比公平。
    """
    if plr <= fec_overhead:
        return 0.0
    return plr - fec_overhead


@torch.no_grad()
def encodec_with_plr(wav: torch.Tensor, sr: int, source_bw: float,
                     p_loss: float, device,
                     fec_bw: float = 0.0) -> Optional[np.ndarray]:
    """
    EnCodec 编解码 + Bernoulli zero-PLC（可选加 FEC 保护）。

    source_bw : EnCodec 源编码码率（kbps），支持 1.5 / 3.0 / 6.0 / 12.0 / 24.0
    fec_bw    : 信道编码开销（kbps），0 表示无 FEC
                total_bw = source_bw + fec_bw
    p_loss    : 信道丢包率
    返回 : 解码波形 np.ndarray (T,) float64，失败返回 None
    """
    if not HAS_ENCODEC:
        return None
    import torchaudio.functional as TAF

    # FEC 后的有效丢包率
    if fec_bw > 0:
        total_bw     = source_bw + fec_bw
        fec_overhead = fec_bw / total_bw
        effective_plr = _fec_residual_plr(p_loss, fec_overhead)
    else:
        effective_plr = p_loss

    cache_key = (source_bw, str(device))
    if cache_key not in _encodec_model_cache:
        try:
            _m = EncodecModel.encodec_model_24khz()
        except Exception as e:
            print(f"  [EnCodec] 模型加载失败（无外网？）: {e}")
            return None
        _m.set_target_bandwidth(source_bw)
        _encodec_model_cache[cache_key] = _m.to(device).eval()
    enc_model = _encodec_model_cache[cache_key]
    enc_model.set_target_bandwidth(source_bw)   # 每次重新设置，防止带宽串扰

    enc_sr  = enc_model.sample_rate
    wav_enc = TAF.resample(wav, sr, enc_sr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)
    frames  = enc_model.encode(wav_enc)

    new_frames = []
    for codes, scale in frames:
        T_f      = codes.shape[-1]
        mask     = (torch.rand(T_f) < effective_plr).to(device)
        codes_lc = codes.clone()
        codes_lc[:, :, mask] = 0          # 丢失 token 帧置零（zero-PLC）
        new_frames.append((codes_lc, scale))

    wav_dec = enc_model.decode(new_frames).squeeze()
    wav_out = TAF.resample(wav_dec.unsqueeze(0), enc_sr, sr).squeeze()
    return wav_out.cpu().numpy().astype(np.float64)


def ffmpeg_codec_with_plr(wav: torch.Tensor, sr: int,
                          codec: str, bitrate_kbps: float,
                          p_loss: float,
                          fec_bw: float = 0.0,
                          frame_ms: float = 20.0) -> Optional[np.ndarray]:
    """
    通用 ffmpeg 编解码器 + 音频帧级 zero-PLC（可选 FEC）。

    codec       : ffmpeg 编解码器名，如 "libopus"、"libopencore_amrnb"
    bitrate_kbps: 源编码码率
    fec_bw      : 信道编码开销（kbps）
    frame_ms    : 丢包仿真帧长（ms），Opus=20ms，AMR=20ms
    返回 : 解码波形 np.ndarray (T,) float64，失败返回 None

    依赖: ffmpeg 已安装且支持对应编解码器
      sudo apt install ffmpeg
    """
    import torchaudio.functional as TAF

    # FEC 后有效丢包率
    if fec_bw > 0:
        total_bw     = bitrate_kbps + fec_bw
        fec_overhead = fec_bw / total_bw
        effective_plr = _fec_residual_plr(p_loss, fec_overhead)
    else:
        effective_plr = p_loss

    # Opus 要求 48kHz；AMR-NB 要求 8kHz
    codec_sr_map = {
        "libopus"           : 48000,
        "libopencore_amrnb" : 8000,
        "libcodec2"         : 8000,
    }
    codec_sr = codec_sr_map.get(codec, 16000)

    wav_resampled = TAF.resample(wav, sr, codec_sr).squeeze().numpy()  # (T,)

    # AMR-NB 固定码率模式（ffmpeg 不支持自由码率，选最近档位）
    bitrate_arg = f"{int(bitrate_kbps * 1000)}"

    ref_path = enc_path = dec_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav_resampled.astype(np.float32), codec_sr)
            ref_path = f.name

        # 选输出格式
        if codec == "libopus":
            ext = ".ogg"
        elif "amrnb" in codec:
            ext = ".amr"
        elif "amrwb" in codec or codec == "libvo_amrwbenc":
            ext = ".3gp"
        elif "aac" in codec:
            ext = ".m4a"
        else:
            ext = ".wav"
        enc_path = ref_path.replace(".wav", ext)
        dec_path = ref_path.replace(".wav", "_dec.wav")

        # 编码（libopus 加 -application voip；原生 aac 编码器需要 -strict -2）
        opus_flags  = ["-application", "voip"] if codec == "libopus" else []
        strict_flags = ["-strict", "-2"] if codec == "aac" else []
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", ref_path,
             "-c:a", codec, "-b:a", bitrate_arg] + opus_flags + strict_flags + [enc_path],
            capture_output=True, timeout=30
        )
        if ret.returncode != 0:
            print(f"  [ffmpeg encode] codec={codec} br={bitrate_arg} failed: "
                  f"{ret.stderr.decode(errors='ignore')[-300:]}")
            return None

        # 解码
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", enc_path, dec_path],
            capture_output=True, timeout=30
        )
        if ret.returncode != 0:
            return None

        dec_audio, _ = sf.read(dec_path, dtype="float32")
        if dec_audio.ndim > 1:
            dec_audio = dec_audio.mean(axis=-1)

        # 音频帧级 zero-PLC
        frame_len = int(codec_sr * frame_ms / 1000)
        n_frames  = len(dec_audio) // frame_len
        for fi in range(n_frames):
            if random.random() < effective_plr:
                dec_audio[fi * frame_len : (fi + 1) * frame_len] = 0.0

        # 重采样回原始 sr
        out = TAF.resample(
            torch.from_numpy(dec_audio).unsqueeze(0), codec_sr, sr
        ).squeeze().numpy()
        return out.astype(np.float64)

    except Exception:
        return None
    finally:
        for p in [ref_path, enc_path, dec_path]:
            if p and os.path.exists(p):
                try: os.unlink(p)
                except OSError: pass


# =============================================================================
# Flow 推理辅助（兼容单码率 & 多码率模型）
# =============================================================================

# =============================================================================
# 波形域线性插值 PLC + EnCodec / ffmpeg LFR
# =============================================================================

def _waveform_linear_interp(wav_np, lost_mask):
    """对 wav_np 中 lost_mask==True 的连续区间做线性插值。"""
    out = wav_np.copy().astype(np.float64)
    N   = len(out)
    i   = 0
    while i < N:
        if i < len(lost_mask) and lost_mask[i]:
            start = i
            while i < N and i < len(lost_mask) and lost_mask[i]:
                i += 1
            end        = i
            val_before = float(out[start - 1]) if start > 0 else 0.0
            val_after  = float(out[end])        if end   < N else 0.0
            length     = end - start
            if length > 0:
                out[start:end] = np.linspace(val_before, val_after, length + 2)[1:-1]
        else:
            i += 1
    return out


def _latent_linear_interp(lat: torch.Tensor, recv_mask: torch.Tensor) -> torch.Tensor:
    """
    对丢失帧在 latent 域做线性插值（替换 Flow 对零输入帧的不可靠输出）。
    lat       : (1, D, T_enc)  Flow 模型输出的完整 latent
    recv_mask : (T_enc,) bool，True=该帧已收到
    返回      : (1, D, T_enc)，丢失帧已被相邻收到帧的 latent 线性插值替换
    """
    lat_out = lat.clone()
    T       = lat_out.shape[-1]
    recv    = recv_mask.tolist()
    i = 0
    while i < T:
        if not recv[i]:
            start = i
            while i < T and not recv[i]:
                i += 1
            end = i   # [start, end) 为丢失区间
            left  = lat_out[:, :, start - 1] if start > 0 else torch.zeros_like(lat_out[:, :, 0])
            right = lat_out[:, :, end]        if end   < T else torch.zeros_like(lat_out[:, :, 0])
            length = end - start
            for k in range(length):
                alpha = (k + 1) / (length + 1)
                lat_out[:, :, start + k] = (1.0 - alpha) * left + alpha * right
        else:
            i += 1
    return lat_out


def encodec_with_interp(wav, sr, source_bw, p_loss, device):
    """
    EnCodec 编解码 + 波形域线性插值 PLC。
    wav      : (T,) or (1,T) float32 CPU tensor
    source_bw: 目标码率 kbps（1.5 / 3.0 / 6.0 / 12.0 / 24.0）
    返回     : (T',) float64 ndarray；失败返回 None
    """
    if not HAS_ENCODEC:
        return None
    import torchaudio.functional as TAF
    cache_key = (source_bw, str(device))
    if cache_key not in _encodec_model_cache:
        try:
            _m = EncodecModel.encodec_model_24khz()
        except Exception as e:
            print("  [EnCodec] 模型加载失败: {}".format(e))
            return None
        _m.set_target_bandwidth(source_bw)
        _encodec_model_cache[cache_key] = _m.to(device).eval()
    enc_model = _encodec_model_cache[cache_key]
    enc_model.set_target_bandwidth(source_bw)   # 每次重新设置，防止多码率共用缓存时带宽串扰

    enc_sr = enc_model.sample_rate          # 24000
    hop    = enc_sr // 75                   # 320 samples/frame

    wav_enc = TAF.resample(wav.squeeze(), sr, enc_sr).unsqueeze(0).unsqueeze(0).to(device)  # 16k→24k
    with torch.no_grad():
        frames = enc_model.encode(wav_enc)

    new_frames = []
    lost_masks = []
    for codes, scale in frames:
        T_f  = codes.shape[-1]
        mask = (torch.rand(T_f) < p_loss).numpy()
        codes_lc = codes.clone()
        codes_lc[:, :, mask] = 0
        new_frames.append((codes_lc, scale))
        lost_masks.append(mask)

    with torch.no_grad():
        wav_dec = enc_model.decode(new_frames).squeeze().cpu().numpy()

    # 构建 sample 级别丢失 mask
    total        = len(wav_dec)
    lost_samples = np.zeros(total, dtype=bool)
    for mask in lost_masks:
        for fi, is_lost in enumerate(mask):
            if is_lost:
                s = fi * hop
                e = min(s + hop, total)
                lost_samples[s:e] = True

    wav_interp = _waveform_linear_interp(wav_dec, lost_samples)
    out = TAF.resample(
        torch.from_numpy(wav_interp.astype(np.float32)).unsqueeze(0), enc_sr, sr
    ).squeeze()
    return out.numpy().astype(np.float64)


def encodec_with_lfrplc(wav, sr, source_bw, p_loss, device):
    """
    EnCodec 编解码 + token 级 LFR（Last Frame Repeat）PLC。

    每个 token 帧（≈13.3ms @ 75fps）独立伯努利丢包。
    丢失的 token 帧直接用前一帧的 codes 重复（不置零、不插值），
    解码器接收到的始终是合法 codebook 条目，不会产生 artifact。
    """
    if not HAS_ENCODEC:
        return None
    import torchaudio.functional as TAF

    cache_key = (source_bw, str(device))
    if cache_key not in _encodec_model_cache:
        try:
            _m = EncodecModel.encodec_model_24khz()
        except Exception as e:
            print("  [EnCodec] 模型加载失败: {}".format(e))
            return None
        _m.set_target_bandwidth(source_bw)
        _encodec_model_cache[cache_key] = _m.to(device).eval()
    enc_model = _encodec_model_cache[cache_key]
    enc_model.set_target_bandwidth(source_bw)

    enc_sr  = enc_model.sample_rate   # 24000
    wav_enc = TAF.resample(wav.squeeze(), sr, enc_sr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        frames = enc_model.encode(wav_enc)

    new_frames = []
    for codes, scale in frames:
        T_f      = codes.shape[-1]
        codes_out = codes.clone()
        for t in range(T_f):
            if random.random() < p_loss:
                # 丢包：重复上一帧 codes（第 0 帧丢时保持原始）
                if t > 0:
                    codes_out[:, :, t] = codes_out[:, :, t - 1]
        new_frames.append((codes_out, scale))

    with torch.no_grad():
        wav_dec = enc_model.decode(new_frames).squeeze()

    out = TAF.resample(wav_dec.unsqueeze(0), enc_sr, sr).squeeze()
    return out.cpu().numpy().astype(np.float64)


def ffmpeg_codec_lfr(wav, sr, codec, bitrate_kbps, p_loss, frame_ms=20.0):
    """
    通用 ffmpeg 编解码 + 帧级 LFR（最后帧重复）PLC。
    codec       : "libopus" / "libvo_amrwbenc" 等
    bitrate_kbps: 编码码率
    返回        : (T',) float64 ndarray；失败返回 None
    """
    import torchaudio.functional as TAF

    codec_sr_map = {
        "libopus"           : 48000,
        "libvo_amrwbenc"    : 16000,
        "libopencore_amrwb" : 16000,
        "libopencore_amrnb" : 8000,
    }
    codec_sr = codec_sr_map.get(codec, sr)
    wav_rs   = TAF.resample(wav.squeeze(), sr, codec_sr).numpy().astype(np.float32)

    if   codec == "libopus":         ext = ".ogg"
    elif "amrwb" in codec:           ext = ".3gp"
    elif "amrnb" in codec:           ext = ".amr"
    else:                            ext = ".wav"

    ref_path = enc_path = dec_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav_rs, codec_sr)
            ref_path = f.name
        enc_path = ref_path.replace(".wav", ext)
        dec_path = ref_path.replace(".wav", "_dec.wav")

        opus_flags = ["-application", "voip"] if codec == "libopus" else []
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", ref_path,
             "-c:a", codec, "-b:a", str(int(bitrate_kbps * 1000))] + opus_flags + [enc_path],
            capture_output=True, timeout=30
        )
        if ret.returncode != 0:
            return None

        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", enc_path, dec_path],
            capture_output=True, timeout=30
        )
        if ret.returncode != 0:
            return None

        dec_audio, _ = sf.read(dec_path, dtype="float32")
        if dec_audio.ndim > 1:
            dec_audio = dec_audio.mean(axis=-1)

        # 帧级 LFR（ffmpeg_codec_lfr）
        frame_len = int(codec_sr * frame_ms / 1000)
        n_frames  = len(dec_audio) // frame_len
        if n_frames > 0:
            last_good = dec_audio[:frame_len].copy()
            for fi in range(n_frames):
                if random.random() < p_loss:
                    dec_audio[fi * frame_len:(fi + 1) * frame_len] = last_good
                else:
                    last_good = dec_audio[fi * frame_len:(fi + 1) * frame_len].copy()

        out = TAF.resample(
            torch.from_numpy(dec_audio).unsqueeze(0), codec_sr, sr
        ).squeeze().numpy()
        return out.astype(np.float64)

    except Exception:
        return None
    finally:
        for p in [ref_path, enc_path, dec_path]:
            if p and os.path.exists(p):
                try:    os.unlink(p)
                except OSError: pass


def ffmpeg_codec_with_builtin_plc(wav, sr, codec, bitrate_kbps, p_loss, frame_ms=20.0):
    """
    Approximate codec-native PLC for low-bitrate speech codecs.

    For AMR-NB in this repo we approximate built-in decoder concealment with
    frame hold / last-frame repeat after encoded-domain transmission, which is
    closer to codec-side PLC behavior than waveform linear interpolation.
    """
    return ffmpeg_codec_lfr(wav, sr, codec, bitrate_kbps, p_loss, frame_ms=frame_ms)


def ffmpeg_codec_with_interp(wav, sr, codec, bitrate_kbps, p_loss, frame_ms=20.0):
    """
    ffmpeg 编解码 + 帧级丢包 + 波形域线性插值 PLC。
    策略与 encodec_with_interp 一致，适用于 Opus / AMR-WB 等 ffmpeg 编解码器。
    """
    import torchaudio.functional as TAF

    codec_sr_map = {
        "libopus"           : 48000,
        "libvo_amrwbenc"    : 16000,
        "libopencore_amrwb" : 16000,
        "libopencore_amrnb" : 8000,
    }
    codec_sr = codec_sr_map.get(codec, sr)
    wav_rs   = TAF.resample(wav.squeeze(), sr, codec_sr).numpy().astype(np.float32)

    if   codec == "libopus":      ext = ".ogg"
    elif "amrwb" in codec:        ext = ".3gp"
    elif "amrnb" in codec:        ext = ".amr"
    elif codec == "aac":          ext = ".m4a"
    else:                         ext = ".wav"

    ref_path = enc_path = dec_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav_rs, codec_sr)
            ref_path = f.name
        enc_path = ref_path.replace(".wav", ext)
        dec_path = ref_path.replace(".wav", "_dec.wav")

        opus_flags   = ["-application", "voip"] if codec == "libopus" else []
        strict_flags = ["-strict", "-2"] if codec == "aac" else []
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", ref_path,
             "-c:a", codec, "-b:a", str(int(bitrate_kbps * 1000))] + opus_flags + strict_flags + [enc_path],
            capture_output=True, timeout=30
        )
        if ret.returncode != 0:
            print(f"  [ffmpeg encode] codec={codec} br={int(bitrate_kbps*1000)} failed: "
                  f"{ret.stderr.decode(errors='ignore')[-300:]}")
            return None

        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", enc_path, dec_path],
            capture_output=True, timeout=30
        )
        if ret.returncode != 0:
            return None

        dec_audio, _ = sf.read(dec_path, dtype="float32")
        if dec_audio.ndim > 1:
            dec_audio = dec_audio.mean(axis=-1)

        # 帧级丢包 + 波形线性插值 PLC
        frame_len    = int(codec_sr * frame_ms / 1000)
        n_frames     = len(dec_audio) // frame_len
        total        = len(dec_audio)
        lost_samples = np.zeros(total, dtype=bool)
        for fi in range(n_frames):
            if random.random() < p_loss:
                s = fi * frame_len
                e = min(s + frame_len, total)
                lost_samples[s:e] = True
        dec_audio = _waveform_linear_interp(dec_audio, lost_samples).astype(np.float32)

        out = TAF.resample(
            torch.from_numpy(dec_audio).unsqueeze(0), codec_sr, sr
        ).squeeze().numpy()
        return out.astype(np.float64)

    except Exception:
        return None
    finally:
        for p in [ref_path, enc_path, dec_path]:
            if p and os.path.exists(p):
                try:    os.unlink(p)
                except OSError: pass


def flow_sample(flow_model, latent: torch.Tensor, spk_emb: torch.Tensor,
                n_steps: int = 10, n_layers: int = None) -> torch.Tensor:
    """
    调用 flow_model.sample()，自动兼容是否有 n_layers 参数。
    多码率模型训练完后 sample() 接受 n_layers；单码率版本不需要。
    """
    import inspect
    sig = inspect.signature(flow_model.sample)
    if "n_layers" in sig.parameters and n_layers is not None:
        n_t = torch.tensor([n_layers], device=latent.device)
        return flow_model.sample(latent, spk_emb, n_steps=n_steps, n_layers=n_t)
    return flow_model.sample(latent, spk_emb, n_steps=n_steps)


# =============================================================================
# PLCMOS（Microsoft，有参考版）
# =============================================================================

def calc_plcmos(deg_np: np.ndarray, ref_np: np.ndarray, sr: int) -> float:
    """
    PLCMOS v2（Microsoft，非侵入式），范围约 1~5。
    v2 模型只需要 degraded 音频，ref_np 保留参数兼容性但不使用。
    音频需要 16kHz，函数内部自动重采样。
    """
    global _plcmos_instance
    if not HAS_PLCMOS:
        return float("nan")
    import torchaudio.functional as TAF
    try:
        if _plcmos_instance is None:
            _plcmos_instance = _PLCMOSEstimator(model_version=2)
        if sr != 16000:
            deg_16k = TAF.resample(
                torch.from_numpy(deg_np.astype(np.float32)).unsqueeze(0), sr, 16000
            ).squeeze().numpy().astype(np.float32)
        else:
            deg_16k = deg_np.astype(np.float32)
        score = _plcmos_instance.run(deg_16k, 16000)
        return float(score)
    except Exception:
        return float("nan")


# =============================================================================
# Opus + inband FEC encode + residual PLC approximation
# =============================================================================

def opus_lbrr_with_plr(wav: torch.Tensor, sr: int, bitrate_kbps: float,
                        p_loss: float, frame_ms: float = 20.0,
                        lbrr_bw: float = 2.0) -> Optional[np.ndarray]:
    """
    Approximate Opus packet-loss simulation with real libopus FEC encode flags.

    Encoder side uses libopus voip/FEC settings. Decoder side remains an
    adapted approximation in this repo: residual loss is modeled with p^2 and
    concealed by waveform linear interpolation.
    """
    import torchaudio.functional as TAF
    codec_sr = 48000
    wav_rs   = TAF.resample(wav.squeeze(), sr, codec_sr).numpy().astype(np.float32)

    # LBRR Bernoulli 模型：每个包携带上一帧冗余，只有连续两包都丢才不可恢复
    # P(不可恢复) = P(包N丢) × P(包N+1也丢) = p²
    # 注：lbrr_bw 决定总码率标签（如 12+2=14kbps，开销比 2/14≈14.3%），
    #     但 p² 模型本身不依赖开销比，适用于任意码率。
    effective_plr = p_loss ** 2

    ref_path = enc_path = dec_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav_rs, codec_sr)
            ref_path = f.name
        enc_path = ref_path.replace(".wav", ".ogg")
        dec_path = ref_path.replace(".wav", "_dec.wav")

        # 编码：-application voip 使 encoder 嵌入 LBRR 数据
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", ref_path,
             "-c:a", "libopus",
             "-b:a", str(int(bitrate_kbps * 1000)),
             "-vbr", "constrained",
             "-application", "voip",
             "-frame_duration", str(frame_ms),
             "-packet_loss", str(max(0, min(100, int(round(p_loss * 100))))),
             "-fec", "1",
             enc_path],
            capture_output=True, timeout=30
        )
        if ret.returncode != 0:
            # 若服务器 ffmpeg 不支持 -application voip，回退到标准编码
            print(f"[opus_lbrr] -application voip 失败，回退标准编码: "
                  f"{ret.stderr.decode(errors='ignore')[-200:]}")
            ret = subprocess.run(
                ["ffmpeg", "-y", "-i", ref_path,
                 "-c:a", "libopus",
                 "-b:a", str(int(bitrate_kbps * 1000)),
                 "-vbr", "constrained",
                 "-frame_duration", str(frame_ms),
                 "-application", "voip",
                 enc_path],
                capture_output=True, timeout=30
            )
            if ret.returncode != 0:
                print(f"[opus_lbrr] 编码失败: {ret.stderr.decode(errors='ignore')[-300:]}")
                return None

        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", enc_path, dec_path],
            capture_output=True, timeout=30
        )
        if ret.returncode != 0:
            print(f"[opus_lbrr] 解码失败: {ret.stderr.decode(errors='ignore')[-300:]}")
            return None

        dec_audio, _ = sf.read(dec_path, dtype="float32")
        if dec_audio.ndim > 1:
            dec_audio = dec_audio.mean(axis=-1)

        # 用有效 PLR 做帧级线性插值 PLC（模拟 LBRR 恢复后的残余丢包）
        frame_len    = int(codec_sr * frame_ms / 1000)
        n_frames     = len(dec_audio) // frame_len
        total        = len(dec_audio)
        lost_samples = np.zeros(total, dtype=bool)
        for fi in range(n_frames):
            if random.random() < effective_plr:
                s = fi * frame_len
                e = min(s + frame_len, total)
                lost_samples[s:e] = True
        dec_audio = _waveform_linear_interp(dec_audio, lost_samples).astype(np.float32)

        out = TAF.resample(
            torch.from_numpy(dec_audio).unsqueeze(0), codec_sr, sr
        ).squeeze().numpy()
        return out.astype(np.float64)

    except Exception:
        return None
    finally:
        for p in [ref_path, enc_path, dec_path]:
            if p and os.path.exists(p):
                try:    os.unlink(p)
                except OSError: pass


# =============================================================================
# 新增指标：PESQ / STOI / WER / SIM
# =============================================================================

_pesq_fn = None
HAS_PESQ = False
_stoi_fn = None
HAS_STOI = False
_whisper_mod = None
_whisper_instance = None
HAS_WHISPER = False


def load_transcript(fpath: str) -> str:
    """读取 LibriSpeech .trans.txt 文字标注。"""
    base  = os.path.splitext(os.path.basename(fpath))[0]
    parts = base.split("-")
    if len(parts) < 2:
        return ""
    trans_path = os.path.join(os.path.dirname(fpath),
                              f"{parts[0]}-{parts[1]}.trans.txt")
    if not os.path.exists(trans_path):
        return ""
    with open(trans_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(base + " "):
                return line[len(base):].strip()
    return ""


def _word_error_rate(ref: str, hyp: str) -> float:
    import re
    ref = re.sub(r"[^\w\s]", "", ref.lower()).split()
    hyp = re.sub(r"[^\w\s]", "", hyp.lower()).split()
    if not ref:
        return float("nan")
    n, m = len(ref), len(hyp)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev = d[:]
        d[0] = i
        for j in range(1, m + 1):
            d[j] = prev[j-1] if ref[i-1] == hyp[j-1] else \
                   1 + min(prev[j], d[j-1], prev[j-1])
    return d[m] / n


def calc_pesq(deg_np: np.ndarray, ref_np: np.ndarray, sr: int) -> float:
    """PESQ-WB（16kHz 宽带）。"""
    if not HAS_PESQ:
        return float("nan")
    try:
        import torchaudio.functional as TAF
        target = 16000
        if sr != target:
            to16 = lambda x: TAF.resample(
                torch.from_numpy(x.astype(np.float32)).unsqueeze(0), sr, target
            ).squeeze().numpy()
            r, d = to16(ref_np), to16(deg_np)
        else:
            r, d = ref_np.astype(np.float32), deg_np.astype(np.float32)
        n = min(len(r), len(d))
        return float(_pesq_fn(target, r[:n], d[:n], "wb"))
    except Exception:
        return float("nan")


def calc_stoi(deg_np: np.ndarray, ref_np: np.ndarray, sr: int) -> float:
    """ESTOI（Extended STOI）。"""
    if not HAS_STOI:
        return float("nan")
    try:
        n = min(len(ref_np), len(deg_np))
        return float(_stoi_fn(ref_np[:n], deg_np[:n], sr, extended=True))
    except Exception:
        return float("nan")


def calc_wer(deg_np: np.ndarray, ref_text: str, sr: int,
             model_size: str = "base") -> float:
    """WER：Whisper 转写 deg_np 后与 ref_text 比较。"""
    global _whisper_instance
    if not HAS_WHISPER or not ref_text:
        return float("nan")
    tmp = None
    try:
        if _whisper_instance is None:
            _whisper_instance = _whisper_mod.load_model(model_size)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, deg_np.astype(np.float32), sr)
            tmp = f.name
        result = _whisper_instance.transcribe(tmp, language="en")
        return _word_error_rate(ref_text, result["text"])
    except Exception:
        return float("nan")
    finally:
        if tmp and os.path.exists(tmp):
            try: os.unlink(tmp)
            except OSError: pass


def calc_sim(deg_np: np.ndarray, ref_np: np.ndarray, sr: int,
             spk_encoder=None, device=None) -> float:
    """说话人余弦相似度（需传入 spk_encoder）。"""
    if spk_encoder is None:
        return float("nan")
    try:
        import torch.nn.functional as _F
        if device is None:
            device = next(spk_encoder.parameters()).device
        deg_t = torch.from_numpy(deg_np.astype(np.float32)).unsqueeze(0).to(device)
        ref_t = torch.from_numpy(ref_np.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb_d = spk_encoder(deg_t)
            emb_r = spk_encoder(ref_t)
        return float(_F.cosine_similarity(emb_d, emb_r).item())
    except Exception:
        return float("nan")


def calc_mcd(deg_np: np.ndarray, ref_np: np.ndarray, sr: int,
             n_mfcc: int = 13) -> float:
    """
    Mel Cepstral Distortion (MCD, dB)，越低越好。
    直接量化两条音频 mel 倒谱系数的 L2 距离，
    与 mel 频谱的视觉相似度高度一致。
    依赖: librosa（pip install librosa）
    """
    try:
        import librosa
        n = min(len(deg_np), len(ref_np))
        if n < 512:
            return float("nan")
        r = ref_np[:n].astype(np.float32)
        d = deg_np[:n].astype(np.float32)
        mfcc_r = librosa.feature.mfcc(y=r, sr=sr, n_mfcc=n_mfcc)   # (n_mfcc, T)
        mfcc_d = librosa.feature.mfcc(y=d, sr=sr, n_mfcc=n_mfcc)
        # 对齐帧数（librosa 不同长度可能差 1 帧）
        t = min(mfcc_r.shape[1], mfcc_d.shape[1])
        diff = mfcc_r[:, :t] - mfcc_d[:, :t]                        # (n_mfcc, T)
        mcd  = (10.0 / np.log(10)) * np.sqrt(2.0 * np.mean(np.sum(diff ** 2, axis=0)))
        return float(mcd)
    except Exception:
        return float("nan")
