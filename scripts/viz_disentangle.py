# -*- coding: utf-8 -*-
"""
viz_disentangle.py  ─  特征分离可视化

验证 SpeechTokenizer 确实完成了语义/音色分离：
  - RVQ-1 latent    : 应捕获语义（内容），不同说话人说相同内容时应接近
  - RVQ2-8 latent   : 应捕获音色（说话人），同一说话人的表示应聚集
  - Encoder 输出    : 未量化，包含全部信息

方法：
  1. t-SNE 可视化（3 个子图，按说话人着色）
  2. 定量分析：说话人内 vs 说话人间 余弦相似度

用法:
  python scripts/viz_disentangle.py \\
      --data_dir /home/chenghao/SpeechTokenizer-main/LibriSpeech \\
      --split test-clean \\
      --n_speakers 10 \\
      --n_per_speaker 10
"""

import os, argparse, random, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speechtokenizer import SpeechTokenizer
from scripts.eval_utils import set_seed, load_audio, load_filelist, build_spk2files


# =============================================================================
# 特征提取
# =============================================================================

_N_LAYERS = 8   # SpeechTokenizer RVQ 层数

@torch.no_grad()
def extract_features(st_model, fpath, sr, max_sec, device):
    """
    返回 dict: {
      "rvq1" .. "rvq8": 第 i 层量化 latent 的时间平均向量 (D,),
      "rvq28"         : 第 2-8 层 latent 之和的时间平均向量 (D,)
    }
    """
    wav   = load_audio(fpath, sr)[:, :int(max_sec * sr)].unsqueeze(0).to(device)
    codes = st_model.encode(wav)    # (8, 1, T_enc)

    def decode_layer(idx):
        vq_l = st_model.quantizer.vq.layers[idx]
        d    = vq_l.decode(codes[idx])
        if d.shape[-1] == st_model.quantizer.dimension:
            d = d.permute(0, 2, 1)
        return d.contiguous()       # (1, D, T_enc)

    to_vec = lambda x: x.squeeze(0).mean(dim=-1).cpu().numpy()   # (D,)

    result = {}
    layers = [decode_layer(i) for i in range(_N_LAYERS)]
    for i, lyr in enumerate(layers):
        result[f"rvq{i+1}"] = to_vec(lyr)
    result["rvq28"] = to_vec(sum(layers[1:]))   # layers 2-8 sum
    return result


# =============================================================================
# t-SNE 可视化
# =============================================================================

def plot_tsne(feat_dict, spk_labels, out_dir):
    """
    feat_dict: {"rvq1".."rvq8": (N,D), "rvq28": (N,D)}
    Layout: 3x3 — rvq1..rvq8 in first 8 cells, rvq28 in last cell
    """
    unique_spks = sorted(set(spk_labels))
    cmap        = plt.get_cmap("tab10")
    colors      = {spk: cmap(i % 10) for i, spk in enumerate(unique_spks)}

    # ordered keys: rvq1..rvq8, then rvq28
    keys = [f"rvq{i}" for i in range(1, _N_LAYERS + 1)] + ["rvq28"]
    panel_titles = {f"rvq{i}": f"RVQ Layer {i}" for i in range(1, _N_LAYERS + 1)}
    panel_titles["rvq1"]  += "\n(semantic)"
    panel_titles["rvq28"]  = "RVQ 2-8 Sum\n(timbre)"

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    axes_flat  = axes.flatten()

    for pi, key in enumerate(keys):
        ax    = axes_flat[pi]
        feats = feat_dict[key]
        print(f"  t-SNE: {key} ...", flush=True)
        tsne = TSNE(n_components=2, perplexity=min(30, len(feats) - 1),
                    random_state=42, n_iter=1000)
        emb  = tsne.fit_transform(feats)

        for spk in unique_spks:
            idx = [i for i, s in enumerate(spk_labels) if s == spk]
            ax.scatter(emb[idx, 0], emb[idx, 1],
                       color=colors[spk], label=f"Spk {spk}",
                       s=30, alpha=0.8, edgecolors="none")

        ax.set_title(panel_titles[key], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.2)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(unique_spks), 10),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("RVQ Layer Feature t-SNE (colored by speaker identity)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_disentangle_tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")


# =============================================================================
# 定量分析：说话人内 vs 说话人间 余弦相似度
# =============================================================================

def quantitative_analysis(feat_dict, spk_labels, out_dir):
    """
    对每种特征空间计算：
      within_sim  : 同说话人对的平均余弦相似度
      between_sim : 不同说话人对的平均余弦相似度
    差距越大 → 该特征越依赖说话人身份。
    """
    results = {}
    for key, feats in feat_dict.items():
        feats_t = torch.from_numpy(feats.astype(np.float32))
        feats_n = F.normalize(feats_t, dim=-1)  # 归一化
        sim_mat  = (feats_n @ feats_n.T).numpy()  # (N, N)

        within, between = [], []
        N = len(spk_labels)
        for i in range(N):
            for j in range(i + 1, N):
                if spk_labels[i] == spk_labels[j]:
                    within.append(sim_mat[i, j])
                else:
                    between.append(sim_mat[i, j])

        results[key] = {
            "within_sim" : float(np.mean(within))  if within  else float("nan"),
            "between_sim": float(np.mean(between)) if between else float("nan"),
            "gap"        : float(np.mean(within) - np.mean(between))
                           if (within and between) else float("nan"),
        }

    # 打印结果
    print("\n── 定量分析：说话人相似度 ──────────────────────")
    print(f"{'Feature':12s}  {'Within-Spk':>12s}  {'Between-Spk':>12s}  {'Gap':>8s}")
    for key, r in results.items():
        print(f"{key:12s}  {r['within_sim']:12.4f}  {r['between_sim']:12.4f}  "
              f"{r['gap']:8.4f}")

    # 条形图
    fig, ax = plt.subplots(figsize=(7, 4))
    keys    = list(results.keys())
    x       = np.arange(len(keys))
    w       = 0.35
    ax.bar(x - w/2, [results[k]["within_sim"]  for k in keys],
           w, label="Within Speaker",  color="#d62728", alpha=0.8)
    ax.bar(x + w/2, [results[k]["between_sim"] for k in keys],
           w, label="Between Speaker", color="#1f77b4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Encoder\nOutput", "RVQ-1\nLatent", "RVQ2-8\nLatent"],
                       fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Speaker Identity in Feature Spaces\n"
                 "(larger gap → more speaker-dependent)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(out_dir, "fig_disentangle_sim.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")

    return results


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path",   default="model_hub/speechtokenizer_hubert_avg/config.json")
    p.add_argument("--ckpt_path",     default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    p.add_argument("--data_dir",      default="/home/chenghao/SpeechTokenizer-main/LibriSpeech")
    p.add_argument("--split",         default="test-clean")
    p.add_argument("--n_speakers",    type=int, default=10,
                   help="参与可视化的说话人数量")
    p.add_argument("--n_per_speaker", type=int, default=10,
                   help="每位说话人随机选取的句子数")
    p.add_argument("--max_sec",       type=float, default=6.0)
    p.add_argument("--out_dir",       default="output/eval_ablation")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    st_model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    st_model.eval().to(device)
    for p in st_model.parameters():
        p.requires_grad_(False)

    sr    = st_model.sample_rate
    files = load_filelist(args.data_dir, args.split)
    spk2f = build_spk2files(files)

    # 选取说话人
    spk_ids = sorted(spk2f.keys())
    random.shuffle(spk_ids)
    selected_spks = spk_ids[:args.n_speakers]

    feat_keys    = [f"rvq{i}" for i in range(1, _N_LAYERS + 1)] + ["rvq28"]
    accum        = {k: [] for k in feat_keys}
    spk_labels   = []

    print(f"Extracting features: {args.n_speakers} speakers x {args.n_per_speaker} utts ...",
          flush=True)
    for spk_idx, spk in enumerate(selected_spks):
        files_spk = spk2f[spk]
        chosen    = random.sample(files_spk, min(args.n_per_speaker, len(files_spk)))
        for fpath in chosen:
            try:
                feats = extract_features(st_model, fpath, sr, args.max_sec, device)
                for k in feat_keys:
                    accum[k].append(feats[k])
                spk_labels.append(spk_idx)
            except Exception as e:
                print(f"  [skip] {os.path.basename(fpath)}: {e}")

    print(f"Total samples: {len(spk_labels)}")

    feat_dict = {k: np.stack(accum[k]) for k in feat_keys}

    os.makedirs(args.out_dir, exist_ok=True)

    print("\nt-SNE visualization ...")
    plot_tsne(feat_dict, spk_labels, args.out_dir)

    print(f"\nResults saved to: {args.out_dir}")


if __name__ == "__main__":
    main(parse_args())
