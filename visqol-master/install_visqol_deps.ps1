# =============================================================
#  install_visqol_deps.ps1
#  在本机下载全部 ViSQOL 依赖，上传到服务器，生成 checksums.txt
#  用法：右键 -> "用 PowerShell 运行"，或在 PowerShell 里执行：
#    .\install_visqol_deps.ps1
# =============================================================

# ── 修改这两行为你自己的服务器信息 ──────────────────────────
$SERVER = "chenghao@10.61.197.247"
$REMOTE_DIR = "~/bazel_distdir"
# ─────────────────────────────────────────────────────────────

$LOCAL_DIR = "$env:USERPROFILE\Downloads\bazel_distdir"
New-Item -ItemType Directory -Force -Path $LOCAL_DIR | Out-Null
Set-Location $LOCAL_DIR
Write-Host "`n[目录] 本地保存路径: $LOCAL_DIR`n" -ForegroundColor Cyan

$files = @(
    @{ name="pybind11_bazel.tar.gz";      url="https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.tar.gz" },
    @{ name="pybind11.tar.gz";            url="https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz" },
    @{ name="six-1.12.0.tar.gz";          url="https://pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz" },
    @{ name="protobuf-3.19.1.tar.gz";     url="https://github.com/protocolbuffers/protobuf/archive/v3.19.1.tar.gz" },
    @{ name="rules_pkg-0.2.5.tar.gz";     url="https://github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz" },
    @{ name="libsvm-v324.zip";            url="https://github.com/cjlin1/libsvm/archive/v324.zip" },
    @{ name="armadillo-14.2.3.tar.xz";   url="http://sourceforge.net/projects/arma/files/armadillo-14.2.3.tar.xz" },
    @{ name="pybind11_abseil.tar.gz";     url="https://github.com/mchinen/pybind11_abseil/archive/a0c36ca08d894b5a138dff31a9057a7dcacfb8fc.tar.gz" },
    @{ name="pybind11_protobuf.tar.gz";   url="https://github.com/pybind/pybind11_protobuf/archive/83f055cc82d983b7d5c3ce3f59ec034ba546d094.tar.gz" },
    @{ name="tensorflow.tar.gz";          url="https://github.com/tensorflow/tensorflow/archive/d5b57ca93e506df258271ea00fc29cf98383a374.tar.gz" },
    @{ name="rules_cc.tar.gz";            url="https://github.com/bazelbuild/rules_cc/archive/40548a2974f1aea06215272d9c2b47a14a24e556.tar.gz" },
    @{ name="googletest.tar.gz";          url="https://github.com/google/googletest/archive/refs/tags/release-1.10.0.tar.gz" },
    @{ name="abseil.tar.gz";              url="https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.tar.gz" },
    @{ name="pffft.tar.gz";              url="https://bitbucket.org/jpommier/pffft/get/7c3b5a7dc510a0f513b9c5b6dc5b56f7aeeda422.tar.gz" }
)

# ── 下载 ──────────────────────────────────────────────────────
$i = 1
foreach ($f in $files) {
    $dest = Join-Path $LOCAL_DIR $f.name
    if (Test-Path $dest) {
        Write-Host "[$i/14] 已存在，跳过: $($f.name)" -ForegroundColor Yellow
    } else {
        Write-Host "[$i/14] 下载: $($f.name)" -ForegroundColor Green
        try {
            Invoke-WebRequest -Uri $f.url -OutFile $dest -UseBasicParsing
            Write-Host "       完成" -ForegroundColor Green
        } catch {
            Write-Host "       失败: $_" -ForegroundColor Red
        }
    }
    $i++
}

Write-Host "`n[完成] 全部文件下载完毕`n" -ForegroundColor Cyan

# ── 上传到服务器 ──────────────────────────────────────────────
Write-Host "[上传] 正在上传到 ${SERVER}:${REMOTE_DIR} ..." -ForegroundColor Cyan
ssh $SERVER "mkdir -p $REMOTE_DIR"
scp "$LOCAL_DIR\*" "${SERVER}:${REMOTE_DIR}/"

# ── 在服务器生成 checksums.txt ────────────────────────────────
Write-Host "`n[校验] 在服务器生成 checksums.txt ..." -ForegroundColor Cyan
ssh $SERVER @"
cd $REMOTE_DIR
sha256sum *.tar.gz *.zip *.tar.xz | tee checksums.txt
echo ''
echo '=== 文件列表 ==='
ls -lh $REMOTE_DIR
"@

Write-Host "`n[全部完成] 现在可以在 gpu05 上运行:" -ForegroundColor Green
Write-Host "  cd ~/SpeechTokenizer-main/visqol-master" -ForegroundColor White
Write-Host "  conda activate chenghao" -ForegroundColor White
Write-Host "  python patch_workspace.py" -ForegroundColor White
Write-Host "  pip install . --no-build-isolation 2>&1 | tee build.log" -ForegroundColor White
