# 本地 SFT 模型 BFCL 评测指南（外部 vLLM server 方式）

本指南对应 `run_sft_eval.sh` 的工作方式：**vLLM 作为独立 server 启动，bfcl 通过 `--skip-server-setup` 连接现成的 OpenAI-compatible endpoint**。

## 环境

- 集群：H 集群 worker 节点（Python 3.12，root 用户）
- PyPI 镜像：`http://mirrors.i.h.pjlab.org.cn/pypi/simple/`
- 模型目录：`/mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/`
- 工作目录（result/score/lock 全部写这里，避免 root 权限冲突）:
  `/mnt/shared-storage-user/ai4good1-share/yimin/bfcl_workspace`
- 仓库位置：
  `/mnt/shared-storage-user/ai4good1-share/yimin/models/gorilla/berkeley-function-call-leaderboard`

> ⚠️ **不要使用** `registry.h.pjlab.org.cn/library/ml-base:20.04` 镜像的 rlaunch 容器跑评测，历史上踩过多个坑。

---

## 1. 安装 BFCL + vLLM 后端

```bash
git config --global --add safe.directory /mnt/shared-storage-user/ai4good1-share/yimin/models/gorilla

cd /mnt/shared-storage-user/ai4good1-share/yimin/models/gorilla/berkeley-function-call-leaderboard

pip install -e ".[oss_eval_vllm]" \
  -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ \
  --trusted-host mirrors.i.h.pjlab.org.cn
```

安装时会出现 `filelock` 版本警告（`virtualenv 20.39.0 requires filelock>=3.24.2`），**只影响 virtualenv 工具本身，可忽略**。

---

## 2. 修复环境依赖问题

### 2.1 vLLM 0.9.1 `aimv2` 冲突（必修）

报错：

```
ValueError: 'aimv2' is already used by a Transformers config, pick another name.
```

`aimv2` 在 transformers 4.52 被原生加入,vLLM 0.9.1 启动时又注册一次导致冲突。Qwen3 需要 `transformers>=4.51`，**不能降级到 4.51 以下**，但 `4.51.x` 既支持 Qwen3 又不会触发 `aimv2` 冲突（`aimv2` 是 4.52 才进 core 的）。把 transformers 固定到 4.51.x 即可：

```bash
# 同时锁 tokenizers，避免 transformers 4.51 与 tokenizers 0.22 不匹配
pip install "transformers>=4.51,<4.52" "tokenizers>=0.21,<0.22" --force-reinstall \
  -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
```

> ⚠️ **不要加 `--no-deps`**：transformers 4.51 严格要求 `tokenizers>=0.21,<0.22`，而新容器常预装 `tokenizers==0.22.x`。加了 `--no-deps` 就不会同步降 tokenizers，vLLM 启动时会报：
> ```
> ImportError: tokenizers>=0.21,<0.22 is required ... but found tokenizers==0.22.2
> ```
> 已经踩过的话，单独补一句 `pip install "tokenizers>=0.21,<0.22"` 即可。

> 备用方案：若因为别的依赖必须保留 transformers ≥ 4.52，可以给 vLLM 的 `ovis.py` 打幂等补丁（`AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)`）。本次环境没走这条路。

### 2.1b `numba` 要求 numpy ≤ 2.2（必修）

vLLM 的 spec-decode 代码链路用了 `numba`，新容器常预装 numpy 2.4，启动 vLLM 时报：

```
ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.4.
```

修法：

```bash
pip install "numpy<2.3" \
  -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
```

### 2.2 `qwen_agent` 缺 `soundfile`（必修）

报错：`ModuleNotFoundError: No module named 'soundfile'`

```bash
pip install soundfile \
  -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ \
  --trusted-host mirrors.i.h.pjlab.org.cn

# 如果 import 时仍报缺 libsndfile.so：
apt-get update && apt-get install -y libsndfile1
```

### 2.3 系统预装的包污染 transformers sandbox（必修）

BFCL 的**多轮测试工具执行沙盒**会 `import transformers`，transformers 在导入时会探测并尝试加载 `flash_attn`、`apex` 等可选加速库。如果系统（`/usr/local/lib/python3.12/dist-packages/`）里有版本不兼容或损坏的 `flash-attn` / `apex`，整条 import 链都会崩，从而让所有多轮用例失败。vLLM server 本身不受影响（它用自己的 attention kernel，不走这条路径）。

遇到过的两个报错：

```
# (1) flash-attn 用旧 torch 编译，和当前 torch 2.7.0 的 C++ ABI 对不上
ImportError: /usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so:
undefined symbol: _ZN3c104cuda9SetDeviceEab
```

```
# (2) 系统里装的是 PyPI 上无关的 `apex` 包，没有 NVIDIA Apex 的 amp 子模块
ImportError: cannot import name 'amp' from 'apex' (/usr/local/lib/python3.12/dist-packages/apex/__init__.py)
```

**修法：两个都卸掉，BFCL sandbox 都用不到，transformers 会自动回退到 SDPA / 不走 apex 分支：**

```bash
pip uninstall -y flash-attn apex
```

> ⚠️ 症状识别：`bfcl generate` 的 tqdm 明显"太快"（例如 5217 个用例 17 秒跑完，~300 it/s），几乎肯定是多轮用例在沙盒 import 阶段全部失败。遇到这种情况当前模型的结果**作废**，卸完污染包后整个模型要重跑。

### 2.4 BFCL 发给 vLLM 的 model name 不匹配（404）

报错：

```
openai.NotFoundError: Error code: 404 -
  {'message': 'The model `Qwen/Qwen3-4B` does not exist.', ...}
```

BFCL handler 默认把 `model_config.py` 里注册的 `model_name`（这里是 `"Qwen/Qwen3-4B"`）当作 API 的 `model` 字段发给 vLLM；而 vLLM server 是用**完整模型路径**启动的，默认 `served_model_name = $model_path`，两者对不上就 404。

**不用重启 vLLM 的修法**：给 `bfcl generate` 加 `--local-model-path` 覆盖 handler 里的 model id：

```bash
bfcl generate \
  --model qwen3-4b-sft-full-training-FC \
  --test-category all \
  --backend vllm \
  --skip-server-setup \
  --local-model-path /mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/full-training-20260416
```

原理：`base_oss_handler.py` 里 `self.model_path_or_id = local_model_path`（优先命令行），就是 API 请求里的 `model` 字段，和 vLLM 的 `served_model_name`（即 `$model_path`）自然对齐。

`run_sft_eval.sh` 已经把 `--local-model-path "$model_path"` 加进去了，跑脚本无需额外处理。

### 2.5 SFT checkpoint `extra_special_tokens=[]` 与 transformers ≥4.51 不兼容

报错（vLLM 启动 tokenizer 时崩在 transformers 内部）：

```
File ".../transformers/tokenization_utils_base.py", line 1190, in _set_model_specific_special_tokens
    self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())
AttributeError: 'list' object has no attribute 'keys'
```

**根因**：

- `extra_special_tokens` 在 transformers 里被声明为 `Optional[dict[str, str]]`，4.51 新加的 `_set_model_specific_special_tokens` 直接 `.keys()`，没做类型兜底
- LLaMA-Factory 之类的 SFT 框架在 `save_pretrained` 时会把这个字段写成 `"extra_special_tokens": []`（空 list），不是 `{}` —— **写时宽松，读时严格**
- 原版 Qwen3 的 `tokenizer_config.json` 根本没这个字段（缺省 = `None`），所以预训练模型不踩，**只有 SFT 后才踩**

**验证某个 checkpoint 是否有这个问题**（替换路径，单行避免 shell 拆行）：

```bash
python3 -c "import json,sys; cfg=json.load(open(sys.argv[1])); print(type(cfg.get('extra_special_tokens')).__name__, '=', cfg.get('extra_special_tokens'))" /path/to/model/tokenizer_config.json
```

输出 `list = ...` → 需要 patch；输出 `dict = ...` 或 `NoneType = None` → 没事。

> ⚠️ 实测 `merged-20260419` 的 list 不是空的，里面有 13 个真实 token（`<|im_start|>`、`<|im_end|>`、`<|vision_start|>` 等 Qwen2-VL 系列）。**不能直接 patch 成 `{}`**，那样会丢 token。

**修复**：`run_sft_eval.sh` 的 `run_eval` 里已经内置一段幂等的 in-place patch，每次启动 vLLM 前自动把 `extra_special_tokens` 从 `list` **保留式转换**成 `dict`（`[t1, t2, ...]` → `{t1: t1, t2: t2, ...}`），写回磁盘。这样既满足 transformers ≥4.51 的类型契约，又**完整保留所有 token 注册**，不影响模型回答质量。

#### 2.5.1 手动 patch 单个 checkpoint（脚本未 pull / 想立即修）

如果服务器没 `git pull` 到新版脚本，或者想在跑 eval 之前先手工修好两个 20260419 checkpoint，**每条单独粘**（路径写死，无变量、无循环、无换行，避免 shell 误拆）：

```bash
python3 -c "import json; path='/mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/merged-20260419/tokenizer_config.json'; cfg=json.load(open(path)); v=cfg.get('extra_special_tokens'); cfg['extra_special_tokens']={t:t for t in v} if isinstance(v,list) else v; json.dump(cfg, open(path,'w'), indent=2, ensure_ascii=False); print('Done:', path, '->', type(cfg['extra_special_tokens']).__name__, '/', len(cfg['extra_special_tokens']) if isinstance(cfg['extra_special_tokens'],dict) else '')"
```

```bash
python3 -c "import json; path='/mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/safety-only-20260419/tokenizer_config.json'; cfg=json.load(open(path)); v=cfg.get('extra_special_tokens'); cfg['extra_special_tokens']={t:t for t in v} if isinstance(v,list) else v; json.dump(cfg, open(path,'w'), indent=2, ensure_ascii=False); print('Done:', path, '->', type(cfg['extra_special_tokens']).__name__, '/', len(cfg['extra_special_tokens']) if isinstance(cfg['extra_special_tokens'],dict) else '')"
```

每条预期输出：`Done: /mnt/.../tokenizer_config.json -> dict / 13`

验证（同样路径写死）：

```bash
python3 -c "import json; cfg=json.load(open('/mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/merged-20260419/tokenizer_config.json')); v=cfg['extra_special_tokens']; print(type(v).__name__, len(v))"
```

```bash
python3 -c "import json; cfg=json.load(open('/mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/safety-only-20260419/tokenizer_config.json')); v=cfg['extra_special_tokens']; print(type(v).__name__, len(v))"
```

两条都看到 `dict 13` 就说明 patch 成功，可以直接 `bash run_sft_eval.sh`。

> ⚠️ 之前用 `for p in ...; do ... $p ...; done` 多行循环踩过坑：终端把 here-string 拆行后 `$p` 变量替换失败，命令会落到 `.../full//tokenizer_config.json` 那种空路径上，patch 看似执行实际没改任何文件。**要么用上面"路径写死、单行"版本，要么把 for 循环写到一个 `.sh` 文件里 `bash` 执行，不要直接粘到 prompt。**

**治本**：去 LLaMA-Factory 的 tokenizer 保存逻辑里把 `extra_special_tokens=[]` 改成 `{}`（或者 `del` 掉走 `None` 默认值），以后训出来的 checkpoint 就不用再 patch。

### 2.x 通用恢复流程

上面任何一类错误导致 result 被写入错误条目后，**必须先清 result 再重跑**。BFCL `generate` 是续跑模式：已经在 result 文件里的 test ID 会被跳过，不论记录是真输出还是错误：

```bash
# 清掉失败结果（按模型 key 替换）
rm -rf "$BFCL_PROJECT_ROOT/result/qwen3-4b-sft-full-training-FC"

# 可选：顺手清 score，避免旧分数混淆
rm -rf "$BFCL_PROJECT_ROOT/score/qwen3-4b-sft-full-training-FC"
```

然后按"6. 断点续跑"的步骤重跑 `bfcl generate` + `bfcl evaluate` 即可。

---

## 3. 配置工作目录

```bash
export BFCL_PROJECT_ROOT=/mnt/shared-storage-user/ai4good1-share/yimin/bfcl_workspace
```

- `result/`、`score/`、`.file_locks/` 都会写到 `$BFCL_PROJECT_ROOT` 下
- `.env` 文件也从这里读

---

## 4. 注册 SFT 模型

`run_sft_eval.sh` 里用的 model key（例如 `qwen3-4b-sft-full-training-FC`）必须先在 `bfcl_eval/constants/model_config.py` 注册，指向对应的 handler。本仓库已经注册了 full-training 和 helpfulness-only 两个变体，新增模型需要相应扩展。

---

## 5. 跑评测（通过 `run_sft_eval.sh`）

```bash
bash run_sft_eval.sh
```

脚本对每个模型依次做：

1. `vllm serve <model_path> --port 8000 --tensor-parallel-size 1 --trust-remote-code &` 启动 vLLM server
2. 轮询 `http://localhost:8000/health`，直到返回 200
3. `export REMOTE_OPENAI_BASE_URL=http://localhost:8000/v1` + `REMOTE_OPENAI_TOKENIZER_PATH=<model_path>`
4. `bfcl generate --model <key> --test-category all --backend vllm --skip-server-setup --local-model-path <model_path>`
   - `--skip-server-setup`：bfcl 不再自己起 vLLM，复用外部 server
   - `--local-model-path <model_path>`：让 bfcl 用模型路径作 API `model` 字段，匹配 vLLM 默认的 `served_model_name`（见 2.4）
5. `bfcl evaluate --model <key> --test-category all`
6. `kill $VLLM_PID`，换下一个模型

### 预期耗时

- vLLM 冷启动（权重加载 + `torch.compile` + CUDA graph 捕获）：约 **2–3 分钟**
- `bfcl generate all` 对 qwen3-4b：视数据量而定，通常 20–40 分钟
- 单卡 A100/H800 40G+ 足够跑 4B 模型

### 显存不够的话

脚本默认上下文长度是模型自带的（qwen3 默认 `max_seq_len=262144`），若 OOM，在 `run_sft_eval.sh` 的 `vllm serve` 命令里加：

```
--max-model-len 32768 --gpu-memory-utilization 0.85
```

---

## 6. 断点续跑（bfcl 失败但 vLLM 还活着）

vLLM 冷启动很贵，如果 `bfcl generate` 报错但 vLLM server 仍在运行，**不要重跑整个脚本**，手动续跑：

```bash
# 1. 确认 vLLM 还活着
curl -s http://localhost:8000/health && echo " still up" || echo " already down"

# 2. 导入环境变量
export REMOTE_OPENAI_BASE_URL="http://localhost:8000/v1"
export REMOTE_OPENAI_TOKENIZER_PATH="/mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/full-training-20260416"

# 3. 重跑 bfcl（注意加 --local-model-path 避免 model name 不匹配 404）
bfcl generate \
  --model qwen3-4b-sft-full-training-FC \
  --test-category all \
  --backend vllm \
  --skip-server-setup \
  --local-model-path /mnt/shared-storage-user/ai4good1-share/yimin/ATbench_Engine_luohaoyu/saves/qwen3-4b/full/full-training-20260416

bfcl evaluate \
  --model qwen3-4b-sft-full-training-FC \
  --test-category all \
  --partial-eval
```

跑完当前模型后再 kill vLLM，换下一个：

```bash
pgrep -af "vllm serve"
kill <PID>
```

---

## 7. 查看结果

```
$BFCL_PROJECT_ROOT/
├── result/<model_key>/BFCL_v4_*_result.json   # 模型原始输出
└── score/
    ├── <model_key>/BFCL_v4_*_score.json       # 单类别详细得分
    ├── data_overall.csv                        # 总体分数（上榜用）
    ├── data_live.csv                           # Live 单轮
    ├── data_non_live.csv                       # Non-Live 单轮
    └── data_multi_turn.csv                     # 多轮
```

快速看总体分数：

```bash
column -s, -t < "$BFCL_PROJECT_ROOT/score/data_overall.csv" | head
```

---

## 8. 跑 baseline Qwen3-4B-Instruct-2507（对照组）

对应脚本 `run_baseline_eval.sh`，和 `run_sft_eval.sh` 同样用外部 vLLM + `--skip-server-setup`，区别是：

1. **新注册的 model key**（见 `bfcl_eval/constants/model_config.py`）：
   - `qwen3-4b-instruct-2507-local-FC`（FC 模式，`QwenFCHandler`）
   - `qwen3-4b-instruct-2507-local`（Prompt 模式，`QwenHandler`）

   > 注：没用 `qwen3-4b-instruct-2507-FC` 这种不带 `-local-` 的名字，是为了和未来可能加的 `QwenAPIHandler`（调云端 API）区分。

2. **跳过 `memory_vector` 分类**。该分类需要从 `huggingface.co` 下载 `sentence-transformers/all-MiniLM-L6-v2`，worker 节点访问不了 HF，会全部超时失败。脚本里已经显式列出 all 的其它所有类别：

   ```
   simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,
   live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,
   multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,
   memory_kv,memory_rec_sum,
   web_search_base,web_search_no_snippet,
   format_sensitivity
   ```

   - 没配 **SerpAPI** 的话，顺手把 `web_search_base,web_search_no_snippet` 也去掉。
   - `evaluate` 步加了 `--partial-eval`，只算已生成的条目，分数会和官方 leaderboard 不直接可比。

3. **模型路径**用共享存储里预下载好的 snapshot：

   ```
   /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/eb25fbe4f35f7147763bc24445679d1c00588d89
   ```

运行：

```bash
bash run_baseline_eval.sh
```

---

## 附录：历史遗留问题（仅参考）

### vLLM 0.8.5 + transformers 版本冲突

旧版本 vLLM 0.8.5 搭配 transformers 4.50+ 会报：

```
TypeError: non-default argument 'vision_config' follows default argument
```

解法（仅当回滚到 0.8.5 时用）：

```bash
pip install transformers==4.49.0 \
  -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn
```

当前 BFCL 已经 pin 到 `vllm==0.9.1`，不会再遇到这个问题。

### 用 uv 加速安装（可选）

```bash
pip install uv \
  -i http://mirrors.i.h.pjlab.org.cn/pypi/simple/ --trusted-host mirrors.i.h.pjlab.org.cn

uv pip install -e ".[oss_eval_vllm]" \
  --index-url http://mirrors.i.h.pjlab.org.cn/pypi/simple/ \
  --trusted-host mirrors.i.h.pjlab.org.cn
```
