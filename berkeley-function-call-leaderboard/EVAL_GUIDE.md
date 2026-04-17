# 本地模型 BFCL 评测指南

## 环境信息

- 集群：H集群 worker 节点
- PyPI 镜像源：`http://mirrors.i.h.pjlab.org.cn/pypi/simple/`
- 模型路径：`/mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/asb-safety-fixed`

## 1. 安装（已完成）

```bash
# 设置 git safe directory
git config --global --add safe.directory /mnt/shared-storage-user/ai4good1-share/yimin/models/gorilla

# 安装 bfcl_eval + vllm 后端
cd /mnt/shared-storage-user/ai4good1-share/yimin/models/gorilla/berkeley-function-call-leaderboard
pip install -e .[oss_eval_vllm] -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
```

## 2. 生成模型响应

### FC 模式（Function Calling）

```bash
bfcl generate \
  --model Qwen/Qwen3-4B-Instruct-2507-FC \
  --test-category all \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.9 \
  --local-model-path /mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/asb-safety-fixed
```

### Prompt 模式

```bash
bfcl generate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --test-category all \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.9 \
  --local-model-path /mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/asb-safety-fixed
```

## 3. 评估结果

```bash
# FC 模式评估
bfcl evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507-FC \
  --test-category all

# Prompt 模式评估
bfcl evaluate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --test-category all
```

## 4. 查看结果

评分文件保存在 `score/` 目录下：

- `score/data_overall.csv` — 总体得分
- `score/data_live.csv` — Live 单轮测试详情
- `score/data_non_live.csv` — Non-Live 单轮测试详情
- `score/data_multi_turn.csv` — 多轮测试详情

## 可选参数

| 参数 | 说明 |
|------|------|
| `--test-category` | 指定测试类别，如 `simple`, `parallel`, `multi_turn`，用逗号分隔多个 |
| `--backend` | `vllm` 或 `sglang`（sglang 更快但需要 SM 80+ GPU） |
| `--num-gpus` | GPU 数量，4B 模型用 1 张够了 |
| `--gpu-memory-utilization` | GPU 显存使用比例，默认 0.9，OOM 时调低 |
| `--partial-eval` | 只评估部分生成结果时使用 |

---

## SFT 模型评测（2026-04-16）

4B 模型 1 张 GPU 即可（建议 A100/H800 40G+，`--gpu-memory-utilization 0.9`）。

### full-training-20260416

```bash
# 生成
bfcl generate \
  --model qwen3-4b-sft-full-training-FC \
  --test-category all \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.9 \
  --local-model-path /mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/full-training-20260416

# 评估
bfcl evaluate \
  --model qwen3-4b-sft-full-training-FC \
  --test-category all
```

### helpfulness-only-20260416

```bash
# 生成
bfcl generate \
  --model qwen3-4b-sft-helpfulness-only-FC \
  --test-category all \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.9 \
  --local-model-path /mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/helpfulness-only-20260416

# 评估
bfcl evaluate \
  --model qwen3-4b-sft-helpfulness-only-FC \
  --test-category all
```
