"""
下载 LibriSpeech 数据集

第一阶段验证推理:  test-clean  (约 346 MB, 40 说话人, 5.4h)
第二阶段训练 Flow: train-clean-100 (约 6.3 GB, 251 说话人, 100h)

LibriSpeech 目录结构（下载后）:
  LibriSpeech/
  └── {split}/
      └── {speaker_id}/
          └── {chapter_id}/
              ├── {spk}-{chap}-{utt}.flac
              └── {spk}-{chap}.trans.txt

说话人数量保证了 "同人取不同句" 的需求:
  test-clean       40 说话人, 平均 ~8 min/人
  train-clean-100  251 说话人, 平均 ~24 min/人
"""

import os
import argparse
import tarfile
import urllib.request

URLS = {
    "test-clean":      ("https://www.openslr.org/resources/12/test-clean.tar.gz",      "346M"),
    "dev-clean":       ("https://www.openslr.org/resources/12/dev-clean.tar.gz",       "337M"),
    "train-clean-100": ("https://www.openslr.org/resources/12/train-clean-100.tar.gz", "6.3G"),
    "train-clean-360": ("https://www.openslr.org/resources/12/train-clean-360.tar.gz", "23G"),
}


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        downloaded_mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {pct:5.1f}%  {downloaded_mb:.1f}/{total_mb:.1f} MB", end="", flush=True)


def download_and_extract(split: str, data_dir: str):
    if split not in URLS:
        raise ValueError(f"未知 split: {split}，可选: {list(URLS.keys())}")

    url, size = URLS[split]
    tar_name = url.split("/")[-1]
    tar_path = os.path.join(data_dir, tar_name)
    extract_path = os.path.join(data_dir, "LibriSpeech", split)

    if os.path.exists(extract_path):
        print(f"[跳过] {split} 已存在: {extract_path}")
        return

    print(f"\n下载 {split} (预计大小 {size})")
    print(f"  URL: {url}")

    if not os.path.exists(tar_path):
        urllib.request.urlretrieve(url, tar_path, reporthook=show_progress)
        print()  # 换行
    else:
        print(f"  tar 包已存在，跳过下载: {tar_path}")

    print(f"  解压到 {data_dir} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print(f"  解压完成")

    # 删除 tar 包节省空间
    os.remove(tar_path)
    print(f"  已删除 tar 包: {tar_path}")


def build_file_list(data_dir: str, split: str, output_path: str):
    """
    生成文件列表，格式: 每行一个 .flac 路径
    后续 validate_inference.py 直接读取
    """
    split_dir = os.path.join(data_dir, "LibriSpeech", split)
    if not os.path.exists(split_dir):
        print(f"[警告] 目录不存在: {split_dir}，跳过生成文件列表")
        return

    files = []
    for spk in sorted(os.listdir(split_dir)):
        spk_dir = os.path.join(split_dir, spk)
        if not os.path.isdir(spk_dir):
            continue
        for chap in sorted(os.listdir(spk_dir)):
            chap_dir = os.path.join(spk_dir, chap)
            if not os.path.isdir(chap_dir):
                continue
            for fname in sorted(os.listdir(chap_dir)):
                if fname.endswith(".flac"):
                    files.append(os.path.join(chap_dir, fname))

    with open(output_path, "w") as f:
        f.write("\n".join(files))

    # 统计说话人数量
    speaker_ids = set()
    for p in files:
        spk_id = os.path.basename(p).split("-")[0]
        speaker_ids.add(spk_id)

    print(f"\n文件列表已写入: {output_path}")
    print(f"  总文件数: {len(files)}")
    print(f"  说话人数: {len(speaker_ids)}")


def print_dir_structure(data_dir: str, split: str):
    split_dir = os.path.join(data_dir, "LibriSpeech", split)
    if not os.path.exists(split_dir):
        return
    speakers = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    # 打印前3个说话人示例
    print(f"\n目录结构示例 ({split}):")
    for spk in sorted(speakers)[:3]:
        spk_dir = os.path.join(split_dir, spk)
        chapters = [d for d in os.listdir(spk_dir) if os.path.isdir(os.path.join(spk_dir, d))]
        for chap in sorted(chapters)[:1]:
            chap_dir = os.path.join(spk_dir, chap)
            wavs = [f for f in os.listdir(chap_dir) if f.endswith(".flac")]
            print(f"  {split}/{spk}/{chap}/")
            for w in sorted(wavs)[:3]:
                print(f"    {w}")
            if len(wavs) > 3:
                print(f"    ... ({len(wavs)} 个文件)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据集保存目录 (默认: data/)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test-clean", "train-clean-100"],
        choices=list(URLS.keys()),
        help=(
            "要下载的 split (默认: test-clean train-clean-100)\n"
            "  test-clean:      346MB, 40说话人  → 推理验证用\n"
            "  train-clean-100: 6.3GB, 251说话人 → Flow 模型训练用"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    for split in args.splits:
        download_and_extract(split, args.data_dir)
        list_path = os.path.join(args.data_dir, f"{split}_files.txt")
        build_file_list(args.data_dir, split, list_path)
        print_dir_structure(args.data_dir, split)

    print("\n全部完成！")
    print("下一步: python scripts/validate_inference.py --data_dir", args.data_dir)
