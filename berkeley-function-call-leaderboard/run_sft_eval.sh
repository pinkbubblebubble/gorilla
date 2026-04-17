#!/bin/bash
set -e

# 工作目录（result/score/lock 文件全写这里，避免权限冲突）
export BFCL_PROJECT_ROOT=/mnt/shared-storage-user/ai4good1-share/yimin/bfcl_workspace

# 清理 proxy 和远端 endpoint 环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset REMOTE_OPENAI_BASE_URL REMOTE_OPENAI_API_KEY REMOTE_OPENAI_TOKENIZER_PATH OPENAI_BASE_URL

MODEL_PATH_BASE=/mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full

# 修复 SFT 模型 tokenizer_config.json 中 extra_special_tokens 为 list 的问题
for model_path in "$MODEL_PATH_BASE/full-training-20260416" "$MODEL_PATH_BASE/helpfulness-only-20260416"; do
    python3 -c "
import json
path = '$model_path/tokenizer_config.json'
with open(path) as f:
    cfg = json.load(f)
if isinstance(cfg.get('extra_special_tokens'), list):
    cfg['extra_special_tokens'] = {}
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print('Patched tokenizer_config.json:', path)
"
done

run_eval() {
    local model_key=$1
    local model_path=$2

    echo "========================================"
    echo "Evaluating: $model_key"
    echo "Path: $model_path"
    echo "========================================"

    bfcl generate \
        --model "$model_key" \
        --test-category all \
        --backend vllm \
        --num-gpus 1 \
        --gpu-memory-utilization 0.9 \
        --local-model-path "$model_path"

    bfcl evaluate \
        --model "$model_key" \
        --test-category all
}

run_eval "qwen3-4b-sft-full-training-FC"    "$MODEL_PATH_BASE/full-training-20260416"
run_eval "qwen3-4b-sft-helpfulness-only-FC" "$MODEL_PATH_BASE/helpfulness-only-20260416"

echo "All done! Results in $BFCL_PROJECT_ROOT/score/"
