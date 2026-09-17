# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
patch_workspace.py  —  离线编译 ViSQOL 前的 WORKSPACE 补丁脚本

功能：
  1. 读取 ~/bazel_distdir/checksums.txt 获取各压缩包的 SHA256
  2. 将 WORKSPACE 中所有 git_repository / new_git_repository
     转换为 http_archive，url 指向本地 file:// 路径
  3. 将现有 http_archive 的 urls / url 字段也追加本地 file:// URL
     （Bazel 会优先使用能访问到的 URL，本地 file:// 优先级最高）
  4. 覆盖写回 WORKSPACE（原文件自动备份为 WORKSPACE.orig）

用法（在服务器上的 visqol-master 目录下运行）：
  python patch_workspace.py
  python patch_workspace.py --distdir /path/to/bazel_distdir
"""

import argparse
import os
import re
import shutil

# ── pffft 的 build_file_content（来自原始 WORKSPACE new_git_repository）────────
_PFFFT_BUILD = """\
cc_library(
    name = "pffft_lib",
    srcs = glob(["pffft.c"]),
    hdrs = glob(["pffft.h"]),
    copts = select({
    "@bazel_tools//src/conditions:windows": [
        "/D_USE_MATH_DEFINES",
        "/W0",
    ],
    "//conditions:default": [
    ]}),
    visibility = ["//visibility:public"],
)
"""

# ── git_repository name → { file, strip_prefix, build_file_content(可选) } ────
GIT_REPO_MAP = {
    "pybind11_abseil": {
        "file": "pybind11_abseil.tar.gz",
        "strip_prefix": "pybind11_abseil-a0c36ca08d894b5a138dff31a9057a7dcacfb8fc",
    },
    "pybind11_protobuf": {
        "file": "pybind11_protobuf.tar.gz",
        "strip_prefix": "pybind11_protobuf-83f055cc82d983b7d5c3ce3f59ec034ba546d094",
    },
    "org_tensorflow": {
        "file": "tensorflow.tar.gz",
        "strip_prefix": "tensorflow-d5b57ca93e506df258271ea00fc29cf98383a374",
    },
    "rules_cc": {
        "file": "rules_cc.tar.gz",
        "strip_prefix": "rules_cc-40548a2974f1aea06215272d9c2b47a14a24e556",
    },
    "com_google_googletest": {
        "file": "googletest.tar.gz",
        "strip_prefix": "googletest-release-1.10.0",
    },
    "com_google_absl": {
        "file": "abseil.tar.gz",
        "strip_prefix": "abseil-cpp-20211102.0",
    },
    "pffft_lib": {
        "file": "pffft.tar.gz",
        "strip_prefix": "jpommier-pffft-7c3b5a7dc510",
        "build_file_content": _PFFFT_BUILD,
    },
}

# ── http_archive name → 对应本地文件名 ────────────────────────────────────────
HTTP_ARCHIVE_FILES = {
    "pybind11_bazel":        "pybind11_bazel.tar.gz",
    "pybind11":              "pybind11.tar.gz",
    "six":                   "six-1.12.0.tar.gz",
    "com_google_protobuf":   "protobuf-3.19.1.tar.gz",
    "rules_pkg":             "rules_pkg-0.2.5.tar.gz",
    "svm_lib":               "libsvm-v324.zip",
    "armadillo_headers":     "armadillo-14.2.3.tar.xz",
}


def read_checksums(distdir: str) -> dict:
    """从 checksums.txt 读取 filename → sha256 映射。"""
    mapping = {}
    ck_path = os.path.join(distdir, "checksums.txt")
    if not os.path.exists(ck_path):
        print(f"[警告] 未找到 {ck_path}，将跳过 SHA256 填充")
        return mapping
    with open(ck_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                sha256, fname = parts
                mapping[os.path.basename(fname.strip())] = sha256
    print(f"[SHA256] 已读取 {len(mapping)} 条记录")
    return mapping


def build_http_archive(name: str, info: dict, distdir: str, checksums: dict) -> str:
    """为 git_repository 生成等价的 http_archive starlark 块。"""
    fpath = os.path.join(distdir, info["file"])
    file_url = f"file://{fpath}"
    sha256 = checksums.get(info["file"], "")
    if not sha256:
        print(f"[警告] 未能获取 {info['file']} 的 SHA256，留空（可手动填入）")

    parts = ["http_archive(", f'    name = "{name}",']
    parts.append(f'    strip_prefix = "{info["strip_prefix"]}",')
    if sha256:
        parts.append(f'    sha256 = "{sha256}",')
    if "build_file_content" in info:
        # 用三引号字符串嵌入 build_file_content
        bfc = info["build_file_content"]
        parts.append(f'    build_file_content = """\n{bfc}""",')
    parts.append(f'    urls = ["{file_url}"],')
    parts.append(")")
    return "\n".join(parts)


def patch_git_repositories(content: str, distdir: str, checksums: dict) -> str:
    """把所有 git_repository / new_git_repository 块替换为 http_archive。"""

    def replace_block(m: re.Match) -> str:
        func_name = m.group(1)
        block = m.group(0)
        name_m = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_m:
            return block
        name = name_m.group(1)
        if name not in GIT_REPO_MAP:
            print(f"[跳过] {func_name}('{name}') 不在映射表中，保持不变")
            return block
        replacement = build_http_archive(name, GIT_REPO_MAP[name], distdir, checksums)
        print(f"[转换] {func_name}('{name}') -> http_archive")
        return replacement

    pattern = re.compile(
        r'(new_git_repository|git_repository)\s*\([^)]*?\)',
        re.DOTALL
    )
    return pattern.sub(replace_block, content)


def patch_http_archives(content: str, distdir: str, checksums: dict) -> str:
    """在现有 http_archive 的 urls / url 字段里追加本地 file:// URL。"""

    def replace_block(m: re.Match) -> str:
        block = m.group(0)
        name_m = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_m:
            return block
        name = name_m.group(1)
        if name not in HTTP_ARCHIVE_FILES:
            return block
        fname = HTTP_ARCHIVE_FILES[name]
        fpath = os.path.join(distdir, fname)
        file_url = f"file://{fpath}"

        if file_url in block:
            return block  # 已经打过补丁

        # 处理 urls = [...] 多 URL 形式
        def insert_into_urls(um: re.Match) -> str:
            inner = um.group(1).rstrip()
            sep = "," if inner and inner[-1] != "," else ""
            return f'urls = [{inner}{sep}\n        "{file_url}",\n    ]'

        new_block, n = re.subn(
            r'urls\s*=\s*\[([^\]]*)\]',
            insert_into_urls,
            block,
            flags=re.DOTALL
        )
        if n == 0:
            # 处理 url = "..." 单 URL 形式（如 rules_pkg）
            def single_to_urls(sm: re.Match) -> str:
                orig = sm.group(1)
                return (f'urls = [\n'
                        f'        "{orig}",\n'
                        f'        "{file_url}",\n'
                        f'    ]')
            new_block, n = re.subn(
                r'\burl\s*=\s*"([^"]+)"',
                single_to_urls,
                block,
                count=1
            )

        if n > 0:
            print(f"[补丁] http_archive '{name}' 追加本地 file:// URL")
        return new_block

    pattern = re.compile(r'http_archive\s*\([^)]*?\)', re.DOTALL)
    return pattern.sub(replace_block, content)


def main():
    parser = argparse.ArgumentParser(description="离线补丁 ViSQOL WORKSPACE")
    default_distdir = os.path.expanduser("~/bazel_distdir")
    parser.add_argument(
        "--distdir", default=default_distdir,
        help=f"bazel_distdir 目录路径（默认: {default_distdir}）"
    )
    parser.add_argument(
        "--workspace", default="WORKSPACE",
        help="WORKSPACE 文件路径（默认: ./WORKSPACE）"
    )
    args = parser.parse_args()

    distdir = os.path.abspath(os.path.expanduser(args.distdir))
    workspace_path = args.workspace

    print(f"distdir   : {distdir}")
    print(f"WORKSPACE : {os.path.abspath(workspace_path)}")
    print()

    if not os.path.exists(workspace_path):
        print(f"[错误] 未找到 WORKSPACE 文件: {workspace_path}")
        return 1

    # 备份
    backup = workspace_path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(workspace_path, backup)
        print(f"[备份] 已备份至 {backup}")
    else:
        print(f"[备份] {backup} 已存在，跳过")

    with open(workspace_path, encoding="utf-8") as f:
        content = f.read()

    checksums = read_checksums(distdir)
    print()

    content = patch_git_repositories(content, distdir, checksums)
    print()
    content = patch_http_archives(content, distdir, checksums)

    with open(workspace_path, "w", encoding="utf-8") as f:
        f.write(content)

    print()
    print("=" * 60)
    print("[完成] WORKSPACE 补丁写入成功")
    print("=" * 60)
    print()
    print("下一步：在 visqol-master 目录执行：")
    print("  pip install . --no-build-isolation 2>&1 | tee build.log")
    print()
    print("若 Bazel 仍尝试访问网络，则再加：")
    print(f"  EXTRA_BAZEL_ARGS='--distdir={distdir}' \\")
    print("  pip install . --no-build-isolation 2>&1 | tee build.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
