#!/bin/bash
# =============================================================
# SpeechTokenizer 完整验证流程
# 功能: 安装依赖 → 下载模型 → 下载数据 → 推理验证
# 用法: bash scripts/run_validate.sh
# =============================================================

set -e  # 任意步骤失败立即退出

# ── 可配置参数 ─────────────────────────────────────────────
MODEL_DIR="model_hub"
DATA_DIR="data"
OUTPUT_DIR="output/validate"
SPLIT="test-clean"
NUM_SAMPLES=20       # 验证的音频条数
SAVE_SAMPLES=5       # 保存重建音频的条数
MAX_SEC=10.0         # 每条音频最大处理秒数
# ───────────────────────────────────────────────────────────

CONFIG_PATH="${MODEL_DIR}/speechtokenizer_hubert_avg/config.json"
CKPT_PATH="${MODEL_DIR}/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
LOG_FILE="${OUTPUT_DIR}/run.log"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] 错误: $1${NC}" | tee -a "$LOG_FILE"; exit 1; }

mkdir -p "$OUTPUT_DIR"
echo "验证流程开始: $(date)" > "$LOG_FILE"

# ── Step 0: 环境检查 ────────────────────────────────────────
log "Step 0: 检查运行环境"

python -c "import torch; print(f'  PyTorch: {torch.__version__}')" || err "未找到 PyTorch，请先安装"
python -c "import torch; print(f'  CUDA 可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || warn "  未检测到 GPU，将使用 CPU（速度较慢）"
python -c "import sys; print(f'  Python: {sys.version.split()[0]}')"

# ── Step 1: 安装依赖 ────────────────────────────────────────
log "Step 1: 安装依赖"

pip install -q beartype 2>&1 | tail -3
pip install -q -e . 2>&1 | tail -3
pip install -q pesq pystoi 2>&1 | tail -3

# 验证关键包
python -c "from pesq import pesq; print('  pesq: OK')"   || err "pesq 安装失败，尝试: pip install pesq"
python -c "from pystoi import stoi; print('  pystoi: OK')" || err "pystoi 安装失败，尝试: pip install pystoi"
python -c "from speechtokenizer import SpeechTokenizer; print('  speechtokenizer: OK')" || err "speechtokenizer 安装失败"

log "依赖安装完成"

# ── Step 2: 下载模型 ────────────────────────────────────────
log "Step 2: 下载 SpeechTokenizer 预训练权重"

if [ -f "$CKPT_PATH" ] && [ -f "$CONFIG_PATH" ]; then
    warn "  模型已存在，跳过下载: $CKPT_PATH"
else
    python scripts/download_model.py --model_dir "$MODEL_DIR" --model hubert_avg \
        2>&1 | tee -a "$LOG_FILE"
fi

[ -f "$CKPT_PATH" ]   || err "模型权重未找到: $CKPT_PATH"
[ -f "$CONFIG_PATH" ] || err "模型配置未找到: $CONFIG_PATH"
log "模型就绪"

# ── Step 3: 下载数据集 ──────────────────────────────────────
log "Step 3: 下载 LibriSpeech ${SPLIT}"

LIST_FILE="${DATA_DIR}/${SPLIT}_files.txt"
if [ -f "$LIST_FILE" ] && [ -s "$LIST_FILE" ]; then
    COUNT=$(wc -l < "$LIST_FILE")
    warn "  数据集已存在，跳过下载 ($COUNT 个文件)"
else
    python scripts/download_data.py \
        --data_dir "$DATA_DIR" \
        --splits "$SPLIT" \
        2>&1 | tee -a "$LOG_FILE"
fi

[ -f "$LIST_FILE" ] || err "文件列表未生成: $LIST_FILE"
log "数据集就绪"

# ── Step 4: 推理验证 ────────────────────────────────────────
log "Step 4: 推理验证（4层 vs 8层 RVQ）"

python scripts/validate_inference.py \
    --config_path  "$CONFIG_PATH" \
    --ckpt_path    "$CKPT_PATH"   \
    --data_dir     "$DATA_DIR"    \
    --split        "$SPLIT"       \
    --num_samples  "$NUM_SAMPLES" \
    --save_samples "$SAVE_SAMPLES" \
    --max_sec      "$MAX_SEC"     \
    --output_dir   "$OUTPUT_DIR"  \
    2>&1 | tee -a "$LOG_FILE"

# ── Step 5: 结果汇总 ────────────────────────────────────────
log "Step 5: 结果汇总"

RESULT_JSON="${OUTPUT_DIR}/results.json"
if [ -f "$RESULT_JSON" ]; then
    python - <<'EOF'
import json, numpy as np, sys, os

path = os.environ.get("RESULT_JSON", "output/validate/results.json")
with open(path) as f:
    results = json.load(f)

pesq4 = [r["pesq_4"] for r in results if r["pesq_4"] == r["pesq_4"]]
pesq8 = [r["pesq_8"] for r in results if r["pesq_8"] == r["pesq_8"]]
stoi4 = [r["stoi_4"] for r in results if r["stoi_4"] == r["stoi_4"]]
stoi8 = [r["stoi_8"] for r in results if r["stoi_8"] == r["stoi_8"]]
late  = [r["latent_relative_error"] for r in results]

print("\n" + "="*50)
print("最终结果对比")
print("="*50)
print(f"  样本数: {len(results)}")
print(f"  {'指标':<12} {'4层(2kbps)':>14} {'8层(4kbps)':>14} {'差距':>10}")
print(f"  {'-'*52}")
print(f"  {'PESQ':<12} {np.mean(pesq4):>7.3f}±{np.std(pesq4):.3f}   {np.mean(pesq8):>7.3f}±{np.std(pesq8):.3f}   {np.mean(pesq8)-np.mean(pesq4):>+8.3f}")
print(f"  {'STOI':<12} {np.mean(stoi4):>7.4f}±{np.std(stoi4):.4f}   {np.mean(stoi8):>7.4f}±{np.std(stoi8):.4f}   {np.mean(stoi8)-np.mean(stoi4):>+8.4f}")
print(f"  {'Latent误差':<12} {np.mean(late):>7.4f}±{np.std(late):.4f}")
print("="*50)
EOF
    RESULT_JSON="$RESULT_JSON" python - <<'PYEOF'
import json, numpy as np, os
path = os.environ.get("RESULT_JSON", "output/validate/results.json")
with open(path) as f:
    results = json.load(f)
late = [r["latent_relative_error"] for r in results]
print(f"\n  Flow 模型需要补全的 latent 相对能量: {np.mean(late):.4f}")
print("  → 该值越大，Flow 模型的难度越高\n")
PYEOF
fi

log "全部完成！"
echo ""
echo "输出文件:"
echo "  重建音频: ${OUTPUT_DIR}/*.wav"
echo "  详细结果: ${OUTPUT_DIR}/results.json"
echo "  运行日志: ${LOG_FILE}"
