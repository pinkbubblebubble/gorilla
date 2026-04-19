#!/bin/bash
set -e

# 工作目录（result/score/lock 文件全写这里，避免权限冲突）
export BFCL_PROJECT_ROOT=/mnt/shared-storage-user/ai4good1-share/yimin/bfcl_workspace

# 清理 proxy 和残留的远端 endpoint 环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset REMOTE_OPENAI_BASE_URL REMOTE_OPENAI_API_KEY REMOTE_OPENAI_TOKENIZER_PATH OPENAI_BASE_URL

VLLM_PORT=8000

# 要跑的类别：显式列出 all 的所有分类，但去掉 memory_vector
# （memory_vector 在 worker 节点上无法访问 HF 下载 all-MiniLM-L6-v2）
# 若没配 SerpAPI，可额外去掉 web_search_base,web_search_no_snippet
TEST_CATEGORIES="simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_rec_sum,web_search_base,web_search_no_snippet,format_sensitivity"


run_eval() {
    local model_key=$1
    local model_path=$2

    echo "========================================"
    echo "Evaluating: $model_key"
    echo "Path: $model_path"
    echo "========================================"

    # 启动 vLLM server
    echo "Starting vLLM server on port $VLLM_PORT..."
    vllm serve "$model_path" \
        --port $VLLM_PORT \
        --tensor-parallel-size 1 \
        --trust-remote-code &
    VLLM_PID=$!

    # 等待 server 就绪
    echo "Waiting for vLLM server to be ready..."
    until curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
        sleep 5
    done
    echo "vLLM server is ready."

    # 指向外部 server
    export REMOTE_OPENAI_BASE_URL="http://localhost:$VLLM_PORT/v1"
    export REMOTE_OPENAI_TOKENIZER_PATH="$model_path"

    bfcl generate \
        --model "$model_key" \
        --test-category "$TEST_CATEGORIES" \
        --backend vllm \
        --skip-server-setup \
        --local-model-path "$model_path"

    bfcl evaluate \
        --model "$model_key" \
        --test-category "$TEST_CATEGORIES" \
        --partial-eval

    # 关闭 vLLM server
    echo "Shutting down vLLM server..."
    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null || true
    unset REMOTE_OPENAI_BASE_URL REMOTE_OPENAI_TOKENIZER_PATH
}

run_eval \
    "qwen3-4b-instruct-2507-local-FC" \
    "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/eb25fbe4f35f7147763bc24445679d1c00588d89"

echo "All done! Results in $BFCL_PROJECT_ROOT/score/"
