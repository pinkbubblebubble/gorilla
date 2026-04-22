#!/bin/bash
set -e

# 工作目录（result/score/lock 文件全写这里，避免权限冲突）
export BFCL_PROJECT_ROOT=/mnt/shared-storage-user/ai4good1-share/yimin/bfcl_workspace

# 清理 proxy 和远端 endpoint 环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset REMOTE_OPENAI_BASE_URL REMOTE_OPENAI_API_KEY REMOTE_OPENAI_TOKENIZER_PATH OPENAI_BASE_URL

MODEL_PATH_BASE=/mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full
VLLM_PORT=8000

# 显式列出 all 的所有分类，去掉 memory_vector
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

    # 修复 SFT 导出脚本写出的 extra_special_tokens=list（transformers >=4.51 要求 dict）
    # 保留式转换：list -> {tok: tok}，避免抹掉 <|im_start|> 等真实 token
    python3 -c "
import json
path = '$model_path/tokenizer_config.json'
with open(path) as f:
    cfg = json.load(f)
v = cfg.get('extra_special_tokens')
if isinstance(v, list):
    cfg['extra_special_tokens'] = {t: t for t in v}
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print('Patched tokenizer_config.json (list->dict, kept', len(v), 'tokens):', path)
"

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
        --local-model-path "$model_path" \
        --num-threads 16

    bfcl evaluate \
        --model "$model_key" \
        --test-category "$TEST_CATEGORIES" \
        --partial-eval

    # 关闭 vLLM server：先 TERM 子进程树，必要时 KILL，最后确认端口真正空
    # （只 kill $VLLM_PID 会留下 EngineCore 子进程占显存+端口，下一个 run 会起不来）
    echo "Shutting down vLLM server..."
    pkill -TERM -P $VLLM_PID 2>/dev/null || true
    kill -TERM $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true

    # 等最多 30 秒让端口释放；还没释放就 KILL
    for _ in $(seq 1 15); do
        curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1 || break
        sleep 2
    done
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM did not exit cleanly, force killing..."
        pkill -KILL -P $VLLM_PID 2>/dev/null || true
        kill -KILL $VLLM_PID 2>/dev/null || true
        pkill -KILL -f "vllm serve .*--port $VLLM_PORT" 2>/dev/null || true
        sleep 3
    fi
    echo "vLLM server shut down."

    unset REMOTE_OPENAI_BASE_URL REMOTE_OPENAI_TOKENIZER_PATH
}

# run_eval "qwen3-4b-sft-full-training-FC"              "$MODEL_PATH_BASE/full-training-20260416"
# run_eval "qwen3-4b-sft-merged-FC"                     "$MODEL_PATH_BASE/merged-20260419"
# run_eval "qwen3-4b-sft-safety-only-FC"                "$MODEL_PATH_BASE/safety-only-20260419"
# run_eval "qwen3-4b-sft-merged-20260420-FC"            "$MODEL_PATH_BASE/merged-20260420"
# run_eval "qwen3-4b-sft-4dataset-merged-FC"            "$MODEL_PATH_BASE/4dataset-merged-20260420"
# run_eval "qwen3-4b-sft-helpfulness-stepfun-0421-FC"   "$MODEL_PATH_BASE/helpfulness-stepfun-0421"
# run_eval "qwen3-4b-sft-helpfulness-only-FC"           "$MODEL_PATH_BASE/helpfulness-only-20260416"
run_eval "qwen3-4b-sft-helpfulness-4dataset-0421-FC"  "$MODEL_PATH_BASE/helpfulness-4dataset-0421"

echo "All done! Results in $BFCL_PROJECT_ROOT/score/"
