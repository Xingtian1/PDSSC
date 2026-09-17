#!/bin/bash
# 在 mgmt 节点上运行（有 GitHub 访问权限）
# 下载 ViSQOL 所有编译依赖到 ~/bazel_distdir

set -e
DISTDIR="$HOME/bazel_distdir"
mkdir -p "$DISTDIR"
cd "$DISTDIR"

echo "=== 下载 ViSQOL 编译依赖 ==="
echo "目标目录: $DISTDIR"

# ── http_archive 依赖（WORKSPACE 里有 SHA256，Bazel 可通过 --distdir 匹配）──
echo "[1/14] pybind11_bazel..."
curl -L --progress-bar -o pybind11_bazel.tar.gz \
  "https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.tar.gz"

echo "[2/14] pybind11..."
curl -L --progress-bar -o pybind11.tar.gz \
  "https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz"

echo "[3/14] six..."
curl -L --progress-bar -o six-1.12.0.tar.gz \
  "https://pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz"

echo "[4/14] protobuf..."
curl -L --progress-bar -o protobuf-3.19.1.tar.gz \
  "https://github.com/protocolbuffers/protobuf/archive/v3.19.1.tar.gz"

echo "[5/14] rules_pkg..."
curl -L --progress-bar -o rules_pkg-0.2.5.tar.gz \
  "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz"

echo "[6/14] libsvm..."
curl -L --progress-bar -o libsvm-v324.zip \
  "https://github.com/cjlin1/libsvm/archive/v324.zip"

echo "[7/14] armadillo..."
curl -L --progress-bar -o armadillo-14.2.3.tar.xz \
  "http://sourceforge.net/projects/arma/files/armadillo-14.2.3.tar.xz"

# ── git_repository 依赖（需转成本地 http_archive）────────────────────────────
echo "[8/14] pybind11_abseil..."
curl -L --progress-bar -o pybind11_abseil.tar.gz \
  "https://github.com/mchinen/pybind11_abseil/archive/a0c36ca08d894b5a138dff31a9057a7dcacfb8fc.tar.gz"

echo "[9/14] pybind11_protobuf..."
curl -L --progress-bar -o pybind11_protobuf.tar.gz \
  "https://github.com/pybind/pybind11_protobuf/archive/83f055cc82d983b7d5c3ce3f59ec034ba546d094.tar.gz"

echo "[10/14] TensorFlow (~300MB，耐心等待)..."
curl -L --progress-bar -o tensorflow.tar.gz \
  "https://github.com/tensorflow/tensorflow/archive/d5b57ca93e506df258271ea00fc29cf98383a374.tar.gz"

echo "[11/14] rules_cc..."
curl -L --progress-bar -o rules_cc.tar.gz \
  "https://github.com/bazelbuild/rules_cc/archive/40548a2974f1aea06215272d9c2b47a14a24e556.tar.gz"

echo "[12/14] googletest..."
curl -L --progress-bar -o googletest.tar.gz \
  "https://github.com/google/googletest/archive/refs/tags/release-1.10.0.tar.gz"

echo "[13/14] abseil-cpp..."
curl -L --progress-bar -o abseil.tar.gz \
  "https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.tar.gz"

echo "[14/14] pffft..."
curl -L --progress-bar -o pffft.tar.gz \
  "https://bitbucket.org/jpommier/pffft/get/7c3b5a7dc510a0f513b9c5b6dc5b56f7aeeda422.tar.gz"

# ── 计算所有 SHA256 ────────────────────────────────────────────────────────────
echo ""
echo "=== 计算 SHA256 ==="
sha256sum *.tar.gz *.zip *.tar.xz | tee checksums.txt

echo ""
echo "=== 下载完成 ==="
ls -lh "$DISTDIR"
echo "请在 visqol-master 目录下运行: python patch_workspace.py"
