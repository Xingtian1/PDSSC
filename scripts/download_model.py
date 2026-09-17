"""
下载 SpeechTokenizer 官方预训练权重到 model_hub/
包含两个 checkpoint:
  - speechtokenizer_hubert_avg  (LibriSpeech, ELU 激活)
  - speechtokenizer_snake       (LibriSpeech + CommonVoice, Snake 激活)
"""

import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download

def download_speechtokenizer_hubert_avg(local_dir: str):
    print("=" * 60)
    print("下载 speechtokenizer_hubert_avg (推荐，论文主模型)")
    print("=" * 60)
    snapshot_download(
        repo_id="fnlp/SpeechTokenizer",
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print(f"\n完成，保存到: {local_dir}")

def download_speechtokenizer_snake(local_dir: str):
    print("=" * 60)
    print("下载 speechtokenizer_snake (Snake 激活, LibriSpeech+CommonVoice)")
    print("=" * 60)
    snapshot_download(
        repo_id="fnlp/AnyGPT-speech-modules",
        allow_patterns=["speechtokenizer/*"],
        local_dir=local_dir,
    )
    print(f"\n完成，保存到: {local_dir}")

def verify_files(model_dir: str):
    required = {
        "speechtokenizer_hubert_avg": [
            "speechtokenizer_hubert_avg/config.json",
            "speechtokenizer_hubert_avg/SpeechTokenizer.pt",
        ]
    }
    print("\n验证文件完整性...")
    all_ok = True
    for name, files in required.items():
        for f in files:
            path = os.path.join(model_dir, f)
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                print(f"  OK  {f}  ({size_mb:.1f} MB)")
            else:
                print(f"  MISS {f}")
                all_ok = False
    if all_ok:
        print("\n所有文件下载完成，可以开始验证推理。")
    else:
        print("\n部分文件缺失，请重新运行脚本。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model_hub",
        help="模型保存目录 (默认: model_hub/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["hubert_avg", "snake", "both"],
        default="hubert_avg",
        help="下载哪个模型 (默认: hubert_avg)",
    )
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    if args.model in ("hubert_avg", "both"):
        download_speechtokenizer_hubert_avg(args.model_dir)

    if args.model in ("snake", "both"):
        download_speechtokenizer_snake(args.model_dir)

    verify_files(args.model_dir)
