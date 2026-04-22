# SFT vs Baseline — Qwen3-4B BFCL 对比

对比对象：

- **Baseline**：`qwen3-4b-instruct-2507-local-FC`（`Qwen/Qwen3-4B-Instruct-2507` 原模型，本地 vLLM）
- **SFT**：`qwen3-4b-sft-full-training-FC`（full-training checkpoint，本地 vLLM）

评测框架：BFCL v4，`bfcl generate` + `bfcl evaluate --partial-eval`。

## Overall 汇总（BFCL leaderboard 口径）

| 指标 | Baseline | SFT | Δ |
|---|---:|---:|---:|
| **Overall Acc** | **34.08%** | **11.43%** | **−22.65 pp**（≈ 66% 相对下降） |
| Non-Live AST Acc | 87.58% | 12.75% | −74.83 |
| &nbsp;&nbsp;Simple AST | 74.83% | 10.00% | −64.83 |
| &nbsp;&nbsp;Multiple AST | 93.50% | 40.00% | −53.50 |
| &nbsp;&nbsp;Parallel AST | 92.50% | 0.00% | −92.50 |
| &nbsp;&nbsp;Parallel Multiple AST | 89.50% | 1.00% | −88.50 |
| Live Acc | 76.39% | 9.03% | −67.36 |
| Multi Turn Acc | 21.50% | 0.00% | −21.50 |
| Web Search Acc | 0.00% | 0.00% | —（未配 SerpAPI） |
| Memory Acc | N/A | N/A | —（未跑 memory_vector） |
| Relevance Detection | 87.50% | 18.75% | −68.75 |
| Irrelevance Detection | 85.19% | 90.79% | **+5.60** |

> Overall 是按 BFCL 的组加权得分（不是每类算术平均），以 `score/leaderboard.csv` 为准。
> Memory Acc 显示 N/A 是因为没跑 `memory_vector`（worker 访问不到 HF），整个 Memory 组不参与汇总；`memory_kv` / `memory_rec_sum` 的分类明细见下表。

## 分类准确率

| 类别 | Baseline | SFT | Δ | 判断 |
|---|---:|---:|---:|---|
| **Simple / AST** | | | | |
| simple_python | 95.50 | 24.00 | **−71.50** | 崩盘 |
| simple_java | 63.00 | 2.00 | **−61.00** | 崩盘 |
| simple_javascript | 66.00 | 4.00 | **−62.00** | 崩盘 |
| multiple | 93.50 | 40.00 | −53.50 | 严重退化 |
| parallel | 92.50 | 0.00 | **−92.50** | 完全失效 |
| parallel_multiple | 89.50 | 1.00 | **−88.50** | 完全失效 |
| irrelevance | 89.17 | 85.42 | −3.75 | 基本持平 |
| **Live** | | | | |
| live_simple | 78.29 | 13.18 | −65.11 | 崩盘 |
| live_multiple | 76.35 | 8.36 | −67.99 | 崩盘 |
| live_parallel | 62.50 | 0.00 | −62.50 | 完全失效 |
| live_parallel_multiple | 66.67 | 0.00 | −66.67 | 完全失效 |
| live_irrelevance | 81.22 | **96.15** | **+14.93** | ⚠ 唯一上升 |
| live_relevance | 87.50 | 18.75 | −68.75 | 崩盘 |
| **Multi-turn** | | | | |
| multi_turn_base | 25.50 | 0.00 | −25.50 | 归零 |
| multi_turn_miss_func | 20.50 | 0.00 | −20.50 | 归零 |
| multi_turn_miss_param | 15.50 | 0.00 | −15.50 | 归零 |
| multi_turn_long_context | 24.50 | 0.00 | −24.50 | 归零 |
| **Memory** | | | | |
| memory_kv | 10.32 | 1.29 | −9.03 | 归零 |
| memory_rec_sum | 30.32 | 1.29 | −29.03 | 崩盘 |
| **Web Search** | | | | |
| web_search_base | 0.00 | 0.00 | — | 未配 SerpAPI |
| web_search_no_snippet | 0.00 | 0.00 | — | 未配 SerpAPI |

> 注：`web_search_*` 需要 `SERPAPI_API_KEY`，两个模型都没配所以都是 0，不参与对比。

## 明显的 pattern

1. **Parallel / multi-turn 完全归零**
   这两类最像是训练数据里**完全没有覆盖到**的分布——多工具并行调用、多轮对话——SFT 之后模型连格式都输不对了。

2. **Live_irrelevance 反而 +15 pp**
   结合其他类几乎全 0，这很可能是模型**学会了"什么都不调 / 拒绝调用"**的倾向，导致 irrelevance（正确答案就是"不要调工具"）分数虚高，而 relevance、simple、parallel 这些**该调工具的场景**全崩。这是典型的"偏置坍塌"信号，而不是真本事。

3. **AST 格式类 simple/parallel 几乎全崩**
   说明 SFT 后连工具调用 JSON schema 都对不上了——可能是训练数据的 tool-call 格式与 BFCL `QwenFCHandler` 期望的格式不一致。

4. **Memory 和 multi-turn 归零**
   多半是训练里没有这类多轮 / 状态维护样本，SFT 把这些能力洗掉了。

## 一句话结论

这次 SFT 是**负向的**——模型在几乎所有需要调用工具的任务上都塌了，唯一升高的 `live_irrelevance` 很可能是"模型学会什么都不调"的副作用而非真实收益。

## 建议排查方向

- **先看几条 SFT 的 raw output**
  `$BFCL_PROJECT_ROOT/result/qwen3-4b-sft-full-training-FC/BFCL_v4_simple_python_result.json`，看是**格式错**（JSON 没对上）还是**语义错**（模型选错函数 / 根本不调）。这一步最能定位问题。

- **确认 chat template 是否一致**
  SFT 保存的 `tokenizer_config.json` 的 `chat_template` 是否和 baseline 的 Qwen3-Instruct 一样；如果训练时用了不同模板，推理端 vLLM 渲染的 prompt 就是错的。

- **检查训练数据分布**
  live / multi-turn / parallel 有没有覆盖？如果只训了 single-turn simple 场景，这个结果就合理但说明数据配比严重偏了。

- **把 `helpfulness-only` checkpoint 也跑一下**
  可以区分"是 full-training 配方的问题"还是"基础 SFT pipeline 的问题"。
