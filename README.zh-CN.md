# 96G 显存 + 64G 内存，RTX 6000 Pro 单卡驯服 Qwen3.8-Flash-Next：KV 池 374K → 1,111,168 全记录（1M 上下文 · 4 路并发）

> *Qwen3.8-Flash-Next (180B MoE) on one RTX 6000 Pro (96GB) + 64GB system RAM — 1.11M-token KV pool, 1M context, native NVFP4 MTP*
>
> RTX 6000 Pro 单卡 96G 显存 + 64G 内存（实测可用 ~62G）· 09-07 → 09-09 00:08 全程战役 · KV 池 **374,208 → 1,111,168 tokens（×2.97）** · YaRN **1M 上下文窗口** · 原生 NVFP4 MTP · 20 槽 / 4 路配置
>
> 一行 `gc.collect()`，池子 +528K tokens——全场最大单跳；**挤出的每一 G 都是预算不是横财：一半买池子，一半买命。**
>
> 从九级点火，到 mixed FP8 顶屋、NVFP4 PLE pinned，再到原生 MTP 一本账。最终 cap 恰好等于 **64×17,362**，页对齐零损耗。吉利数收官，验收账不抹零：**最终扩容版只做 mini 热身；768K 与 soak 双 PASS 属于 10 槽 / cap1M 前身。**
>
> 这是一份工程实录。草稿的规矩是：“所有数字均出自逐字留档日志，无一处推算。”终稿继续守住其中最要紧的一条：实测不靠推算补位。日志摘录逐字保留；比例、页对齐与显存账的算术明确标为复算；张量审计、报告转述和未验方案各自注明出处。**不把容量估算写成测试成绩，不把旧实例 PASS 挪给新配置。**

头部两端池值来自 `sglang-silver-b2-retry3.log` 与 `sglang-silver-native-cap1111168.log`；×2.97 为两值之比四舍五入。本文说的 1M 窗口是 1,048,576，最终 1,111,168 是全局 token 池，不能写成单请求 1.11M 窗口。时间均为 Asia/Shanghai；前史保留 09-06 晚的 Route A 失败。

## 目录

- [1. 开场：硬件、模型、挑战](#section-1)
- [2. 实验矩阵：早期九级点火全记录](#section-2)
- [3. 发现](#section-3)
- [4. 容量调优公式拆解](#section-4)
- [5. 三级异构架构：把 PLE 驻留与 KV 缓存分开记](#section-5)
- [6. PLE 钉内存分支线：三连败与重生](#section-6)
- [7. 晚间战役：18:14 → 00:08，九幕收官](#section-7)
- [8. 坑清单：现象 → 根因 → 修法](#section-8)
- [9. 方法论：让下一轮从账本继续](#section-9)
- [10. 复现指南：最终原生 NVFP4 路线](#section-10)
- [11. 池的边界模型：fraction 退化为护栏（09-10/09-11 续测）](#section-boundary)
- [12. 现役配置快照与未完成项](#section-11)
- [13. 发布前检查清单](#section-12)
- [附录 A：补丁 diff 与适用范围](#appendix-a)
- [附录 B：日志与报告索引](#appendix-b)
- [附录 C：关键数字抽查记录](#appendix-c)
- [鸣谢](#鸣谢)

---

<a id="section-1"></a>

## 1. 开场：硬件、模型、挑战

### 1.1 硬件

| 项 | 配置 | 备注 |
|---|---|---|
| GPU | RTX 6000 PRO 96G（Blackwell，SM120，device `10de:2bb1`） | 单卡，TP=1 |
| 驱动 | `nvidia-driver-610-open` 610.43.02（CUDA UMD 13.3） | **Blackwell 强制 open 内核模块**，闭源 610 变体初始化即 `RmInitAdapter failed` |
| Host RAM | 62 GB | 本系列所有"物理不可行"判定的约束来源 |
| 磁盘 | NVMe 2.1 TB | 承载 PLE 流式外置 + 全部缓存 |
| 工具链 | CUDA 13.3 toolkit + GCC 15.3（venv 内，不碰驱动） | FlashInfer / sglang JIT 必需 |

62 GB host RAM 是全篇的关键约束：官方满配（pinned PLE 47.68G + HiCache 32G ≈ 79.7G）在这台机器上**物理装不下**，这直接决定了架构选型（见 §3.1 发现③）。

### 1.2 模型与量化形态

- 模型：**Qwen3.8-Flash-Next**（混合 MoE，内部架构名 `Qwen4Exp`，自带 MTP draft 块；`Qwen3.8-27B` 是服务别名，本篇不据此判定总参数量）
- 起步 checkpoint：NVFP4 **SSD-Stream** 格式——主权重 78G 上 GPU，PLE 大表（51.2G）抽到外置 bin 按 NVMe 页缓存流式供给，staging 仅 2×320 MiB registered buffers
- KV cache：`fp8_e4m3`；mamba SSM 状态：`bfloat16`
- 投机解码：NEXTN（EAGLE 家族）+ 自带 MTP draft，`speculative_num_steps=3, topk=1, draft_tokens=4`
- 推理栈：SGLang（上游 jpezzulli/sglang-rtxpro6000 `pennyroyal-main-sm120-final`，tag `pennyroyal-v2.3.0`，HEAD `836206a0ad`；本机树 `b21f03306`）

### 1.3 挑战：374,208 vs 824,384

同一张 RTX 6000 Pro 96G（torch 可见 95.01 GiB）、同一份代码树、同一个 `mem-fraction-static`：

- 上游 jpezzulli 文档记录：`max_total_num_tokens = 824,384`（qualified）
- 本机 Route B（SSD-Stream 插件）首次点火：`max_total_num_tokens = 374,208`——**约为上游45%（两池值复算）**

按草稿单价复算，差额 450,176 tokens × 13,312 B/token = **5.59 GiB**。这 5.59 GiB 去哪了、还能不能要回来、要回来之后天花板在哪——是这场战役的起点；晚间如何把收益换成运行余量，见 §7。

```text
[2026-09-07 01:36:33] max_total_num_tokens=374208, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=3, context_len=524288, available_gpu_mem=9.99 GB
```

---

本节硬件、版本和上游参考继承`README.md` §1；本次未重测环境。两日日记补时间线，最终现役只以00:08的`CURRENT-PRODUCTION.md`为准。

<a id="section-2"></a>

## 2. 实验矩阵：早期九级点火全记录

### 2.1 矩阵总表

| # | 轮次 | 变更 | 池子 `max_total_num_tokens` | 增量 | 定价后 avail | 上下文 |
|---|---|---|---|---|---|---|
| L1 | ssd-stream 点火 | 基线（mem-fraction 0.981，mamba 24） | **374,208** | — | 9.99 GB | 512K |
| — | 单变量对照 | draft 量化 fp8→unquant | 374,208 | ±0 | — | 512K |
| L2 | 0.992 + mamba12 | `mem-fraction 0.992`，mamba 24→12 | **519,552** | +145,344 | 8.85 GB | 512K |
| L3 | **poolfix** | 删 `expandable_segments` + empty_cache 补丁 | **558,400** | +38,848 | 9.41 GB | 512K |
| L4 | **estfix** | estimator 漏账修复 | **608,320** | +49,920 | 8.76 GB | 512K |
| D | [诊断轮] 拆投机全套 | 无 draft | **968,832** | +360,512 vs L4 | 7.08 GB | 512K |
| L5 | **gcfix** | pool sizing 前 `gc.collect()` | **1,136,960** | +528,640 vs L4 | 1.86 GB | 512K |
| L6 | 原生 NEXTN + FP8 MTP + 提前共享 | 外置 draft→原生，cap 1,100,000 | **1,099,968** | −3.2% vs L5 | 1.37 GB | 512K |
| L7 | 1M 上下文 | context 1,048,576，rope ×4.0，cap 870,000 | **869,952** | — | 3.95 GB | **1M** |
| L8 | **mamba10 精调（当时现役）** | mamba 16→10，cap 925,000 | **924,992** | +55,040 | 3.62 GB | **1M** |

> D 行是诊断轮，不是现役配置：它的价值是**实锤**——拆掉投机全套后池子凭空多出 360,512 tokens，证明整套投机路径占用预算；不能把360,512全部当作可由“原生一本索引”消除的实例税。
>
> 另有三轮分支点火未入矩阵：09-07 22:23–22:48 的 NVFP4 PLE 钉内存两轮——钉载机制成功、启动未走完闭环（§6.3）；09-08 01:30 的 cap 1.2M 试探轮——boot-only 探针，graph 捕获差 3 MB 未闭环（§6.4）。矩阵九级只收录池子数字闭环的点火。

### 2.2 逐级日志原文

以下带完整日期的启动行逐字引自当轮日志；代码块中的轮次注释为编者所加。§6的短时间戳和省略路径片段沿用草稿节录，不冒称完整原行：

```text
# L1 ssd-stream 点火（sglang-silver-b2-retry3.log）
[2026-09-07 01:36:33] max_total_num_tokens=374208, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=3, context_len=524288, available_gpu_mem=9.99 GB

# L2 mem-fraction 0.992 + mamba 12（sglang-silver-m12.log）
[2026-09-07 08:36:14] max_total_num_tokens=519552, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=2, context_len=524288, available_gpu_mem=8.85 GB

# L3 poolfix（sglang-silver-poolfix.log）
[2026-09-07 09:12:58] max_total_num_tokens=558400, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=2, context_len=524288, available_gpu_mem=9.41 GB

# L4 estfix（sglang-silver-estfix.log）
[2026-09-07 10:16:18] max_total_num_tokens=608320, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=2, context_len=524288, available_gpu_mem=8.76 GB

# D 无 draft 诊断（sglang-silver-nodraft.log）
[2026-09-07 14:14:32] max_total_num_tokens=968832, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=2, context_len=524288, available_gpu_mem=7.08 GB

# L5 gcfix（sglang-silver-gcfix.log）
[2026-09-07 17:27:41] max_total_num_tokens=1136960, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=2, context_len=524288, available_gpu_mem=1.86 GB
```

L6（原生 NEXTN）是三轮调参落地的：greedy 定价 1,168,704 → 1,149,952 两次 OOM 后定额 cap=1,100,000 稳过：

```text
# L6 原生 NEXTN + FP8 MTP 三轮调参（sglang-silver-fp8mtp*.log）
[2026-09-07 23:00:52] max_total_num_tokens=1168704, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=2, context_len=524288, available_gpu_mem=0.45 GB
[2026-09-07 23:07:13] max_total_num_tokens=1149952, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=2, context_len=524288, available_gpu_mem=0.72 GB
[2026-09-07 23:11:06] max_total_num_tokens=1099968, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=2, context_len=524288, available_gpu_mem=1.37 GB
```

L7/L8 是 1M 上下文窗口时代：

```text
# L7 1M 上下文（sglang-silver-1mctx-r6.log，6 轮 OOM 调参后的收官轮）
[2026-09-07 23:52:41] max_total_num_tokens=869952, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=3, context_len=1048576, available_gpu_mem=3.95 GB

# L8 mamba10 精调（sglang-silver-mamba10.log，当时现役）
[2026-09-08 00:17:31] max_total_num_tokens=924992, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=2, context_len=1048576, available_gpu_mem=3.62 GB
```

---

L1–L8加D为早期九级；白天C方案与晚间九幕见§6.5、§7。

<a id="section-3"></a>

## 3. 发现

### 3.1 首阶段四大发现（①–④）

**发现①：池子差距是"定价时刻可用内存"之争，不是预留变大。**
同卡同代码同 fraction 下 450K tokens 的差距，首轮归因候选有四：权重上卡膨胀、draft 量化瞬态、QSA 索引结构、加载器缓冲滞留。第一枪就把"draft 量化瞬态"打死了——**单变量对照实验**：draft 量化 fp8→unquant，池子 374,208 分毫不差。后续 gcfix 轮（发现⑦之前的 L5）给出真答案：定价时刻的"已占用"里混着大量**不该在场的临时占用**（见 L3–L5），逐个清掉即可。该组对照只否定了当轮 draft 开关假说；上游与本机仍有树、版本及加载路径差异，不能用一次开关实验把跨环境差额全部归因。

**发现②：KV 池单价恒定，容量账可以完全代数化。**
草稿按主/draft KV 分配记下 **13,312 B/token** 的早期单价，以下乘除为账本复算：824,384 × 13,312 = 10.22 GiB（上游文档 9.44 + 0.78 ✓）；374,208 × 13,312 = 4.64 GiB ✓。单价含主模型 KV（2.14 + 2.14 GiB 部分比例）+ draft KV（0.18 + 0.18）。这意味着整个容量问题可以化归为一道减法题：**可用于池预算的字节 ÷ 13,312**。公式拆解见 §4。

**发现③：Route A（pinned PLE）在 62G host 上物理不可行，SSD-Stream 流式是当时唯一活路。**
官方 qualified 路线要求 PLE 大表 47.68 GiB 常驻 pinned host memory；加载瞬态约13G再叠加系统等占用，撞上62G host限制，kernel OOM，**三连败证死**——钉内存连 swap 都救不了（`cudaHostAlloc` 不走 swap）。SSD-Stream 页缓存流式外置（staging 2×320 MiB registered buffers）把 host 侧需求从 47.7G 压到不足 1G，是 62G 机器跑满血 fp8 PLE 的唯一路径。该判定的完整法证，以及 NVFP4 量化把表砍到 28.01 GiB 后钉内存起死回生的分支后续，见 §6。

**发现④：`expandable_segments` 偷走 ~7% 池子。**
金线（补丁版前身）实测：加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后池子 538,880 → 500,032（−7.2%）。在 mem-fraction 0.992 的极限压榨环境下，分配器段管理策略的每一字节都要计较——启动配置里**必删**（poolfix 轮在银线复验：删除配置与 poolfix 合并后贡献 L3 的 +38,848；不是把两项分别测出同一收益）。

### 3.2 续作五大发现（⑤–⑨）

以下五条来自 09-07 深夜至 09-08 凌晨的 1M 上下文冲刺与词表压缩实验。

**发现⑤：QSA extend 全前缀 gather 是 O(D) 显存的真瓶颈——名义 1M ≠ 实际 1M。**
1M 上下文轮 6 次 OOM 的根因不在 KV 池容量，而在 **QSA（稀疏注意力）extend 路径的全前缀 gather**：预填阶段按整个序列深度 D  gather 中间量，显存占用 O(D)。后果是：`context_len=1048576` 只是名义窗口，**当时配置的单请求边界估计约 ~850K tokens**——再往上加，OOM 不发生在"池子不够"，而发生在"extend 峰值爆表"。这是当时 cap 降到 870,000 的原因。该值不是新 mixed 栈的已测上限；晚间只把严格零缓存 786,432 输入列作实测通过。

**发现⑥：当轮 chunk 8192 与 mem-fraction 0.992 组合失败。**
直觉上 prefill chunk 越大吞吐越高，但 chunked prefill 的激活峰值 ~2.5 GB 恰好踩进 0.992 环境下 graph/warmup 的生存空间——两轮 OOM 实证后回退 `--chunked-prefill-size 4096`。在极限压榨配置里，**吞吐参数与容量参数不是独立变量**。

**发现⑦：mamba 槽位语义——MTP 状态税是 5 槽/路。**
混合线性注意力（gdn/mamba）状态下每一路并发请求占用 **5 个槽位 = 4 个 draft 位 + 1 个主状态位**，因此 `并发数 = floor(max_mamba_cache_size / 5)`，且槽随请求结束即还（日志中 `mamba num: 3→4` 随请求起落可见）。实操推论：
- mamba 16 → 3 路（`max_running_requests=4` 实际被钳到 3）
- mamba 10 → **2 路**，对应当时两路负载预算
- 16→10 释放的显存折算 +55,040 tokens 池子（L7→L8），这就是"按负载精调 10 槽 + cap 925,000"的出处

**发现⑧：FR-Spec 词表压缩——机制无罪，语料定生死。**
FR-Spec 将 draft 候选词表从 248,320 砍到 65,536（频率排序高频 token，target 验证仍全量），窄头仅 +320 MiB。实测：

| 负载 | 结果 |
|---|---|
| 英文代码 | **177.2 tok/s，与基线持平**——机制正常（上游单流实测 171.93 tok/s、+9.7~16.8% 可佐证机制本身有效） |
| 中文负载 | **106.5 tok/s（−31%）**（公平热身 A/B 口径：基线 155–158 tok/s → FR 106.5），accept rate 自基线 0.57–0.66 崩至 **0.06–0.28** |

根因：官方 64K map 基于 **SGLang Python 源码语料**构建，65,536 个 token 里只有 **13 个 CJK token（0.02%）**——中文 draft 步步落空。教训一句话：**这次词表压缩败在 map 语料，不能据此判压缩机制无效**。上游 sglang issue #8581 印证连默认 map 在英文场景都可能翻车。

重建路径（已明确，未执行）：以 [thunlp/FR-Spec](https://github.com/thunlp/FR-Spec)（ACL 2025）方法 + Pennyroyal `build_token_map.py`，用中文语料自建 token map，重建后必须 A/B 实测再定去留。FR-Spec 轮为独立验证轮，现役配置未开启。

**发现⑨：续轮缓存实证——99.99% 前缀命中，二轮 prefill 快 172×。**
768K 大海捞针同 prompt 续问（radix 前缀复用；仅凭命中不能证明发生过 HiCache 驱逐后回载）：

| 轮次 | prefill 耗时 | 前缀命中 |
|---|---|---|
| 首轮 | 138.3 s | — |
| 二轮 | **0.8 s** | **99.99%**（#cached-token 768,320 / 768,345） |

```text
# 二轮续问时的 decode 日志（sglang-silver-mamba10.log）——高前缀命中、mamba 使用计数随请求变化：
[2026-09-08 00:21:09] Decode batch, #running-req: 1, #full token: 768320, full token usage: 0.83, mamba num: 4, mamba usage: 0.40, accept len: 2.85, accept rate: 0.62, cuda graph: True, gen throughput (token/s): 0.73, #queue-req: 0
```

**按日志展示精度复算约 173×；草稿归档称“172×”，保留为原报告口径**。对"多轮追问同一个长文档"的真实负载，长上下文的可用性直接翻一个数量级——这证明前缀复用有效；L2 回载性能需要独立驱逐/回载实验。

---

FR数字见`2026-09-07.md`与`sglang-silver-frspec.log`；公平热身降幅用约−31%，旧−40%作废。上游速度仅属草稿背景，不计本机战果。

<a id="section-4"></a>

## 4. 容量调优公式拆解

### 4.1 定价公式

以下是草稿对早期定价过程的简化账本，不是源码逐行等价式。实际 estimator、余量预留与 draft 路径须以当轮日志校准；晚间实测 +100K 启动成本另记 1.30GB（`2026-09-08.md`，pil2048 复盘）。SGLang 在权重加载完毕、CUDA graph 捕获前为 KV 池定价：

```text
pool_tokens = page_align( min(cap, floor(可用字节 / 13,312)) )
其中：
  可用字节 = mem_fraction_static × GPU 总显存 − 定价时刻已占用
  定价时刻已占用 = 主权重 + draft 权重 + mamba 状态 + 装载期残留（垃圾/瞬态） + 图/warmup 预留
  page_align(x) = floor(x / page_size) × page_size   # page_size = 64
```

三个可动的旋钮，对应矩阵的每一级：

| 旋钮 | 动作 | 对应轮次 |
|---|---|---|
| `mem_fraction_static` | 0.981 → 0.992（把"预留"压到极限） | L2 |
| 定价时刻已占用 | poolfix（清瞬态）+ estfix（修漏账）+ gcfix（清垃圾） | L3/L4/L5 |
| draft 架构 | 外置 draft → 原生 NEXTN + FP8 + 提前共享 | L6 |

### 4.2 页对齐与 cap 定额

`page_size=64` 下所有 cap 定额都要过一遍 `floor(cap/64)×64`，早期三次定额全部吻合；最终 1,111,168 的零损耗对齐另见 §7.9：

| 设定 cap | 实际池子 | 页对齐验证 |
|---|---|---|
| 1,100,000 | 1,099,968 | 1,100,000 ÷ 64 = 17,187.5 → 17,187 × 64 ✓ |
| 870,000 | 869,952 | 870,000 ÷ 64 = 13,593.75 → 13,593 × 64 ✓ |
| 925,000 | 924,992 | 925,000 ÷ 64 = 14,453.125 → 14,453 × 64 ✓ |

**为什么必须 cap 定额、不能 greedy？** L6 的教训：0.992 环境下 greedy 多轮取大池后，CUDA graph 捕获 + tokenizer 进程首次 CUDA 初始化 + vision 预处理 `.to(device)` 的余量被吃光，连续 OOM（sglang-silver-fp8mtp 系列日志三轮实证：余 0.45G OOM → 余 0.72G 仍 OOM → 定额 1,100,000 留 1.37G 稳态余量 ✓）。**极限压榨配置里必须给 graph/warmup 留出确定性的余量**。

### 4.3 mamba 槽位税

mamba 状态按槽计价（实测 24 槽 = 1.37 GiB，conv 0.05 + ssm 1.32，约 57 MiB/槽）。结合发现⑦的 5 槽/路语义：

| mamba 槽位 | 并发路数 | mamba 显存 | 适用负载 |
|---|---|---|---|
| 24 | 4 | 1.37 GiB | L1 基线（对容量最贵） |
| 16 | 3 | ~0.92 GiB | 1M 上下文轮 |
| 10 | **2** | ~0.58 GiB | 早期两路配置；终态已扩到 20 槽 / 4 路 |

### 4.4 HiCache L2 尺寸

`--hicache-size 16`（GB 预算）的逐字账本：

```text
[2026-09-07 18:39:10] Allocating kv hierarchical KV host pool: 1078016 tokens, 14.35 GB host memory, packed MTP KV layers: target_layers=12, draft_layers=1, total_layers=13.
[2026-09-07 18:39:15] Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=False hybrid_ssm=True hicache_attached=True streaming_wrapped=False
```

host 池 14.35 GB + mamba host 1.36 GB + QSA 索引 0.86 GB ≈ 16G 预算，正好卡满。host MemAvailable 稳定 ~39.8G、swap 零活动、NVMe `r_await` 1–4 ms——62G host 跑 L2 完全可行（前提是不走 pinned PLE，见发现③）。

**运维注意**：gcfix 后设备池 1,136,960 已超过 host 池 1,078,016，L2 反向倒挂——要么加大 `hicache-size`，要么接受超出部分不落 L2（老板决策点，早期 mamba10 配置下设备池 924,992 < 1,078,016，暂无倒挂）。

---

终态槽位实测另记：20 槽、mamba cache 1.16G、4 路（`CURRENT-PRODUCTION.md`；最终日志）。早期每槽估算不能替代终态实称，扩槽也没有完成调度断言的源码加护。

<a id="section-5"></a>

## 5. 三级异构架构：把 PLE 驻留与 KV 缓存分开记

早期是 GPU KV、host HiCache、NVMe PLE 三层；到 00:08，PLE 已从持续页缓存流式换成 NVFP4 pinned 驻留。SSD-Stream 仍是装载插件名，不能因此写成“现役每步都从 SSD 读 PLE”。

```text
RTX 6000 PRO 96G
  mixed target：NVFP4 experts + FP8 QSA 12 层 / GDN 36 层投影
  native MTP：在线 NVFP4 routed experts；其余 BF16
  embed/head：draft 别名共享 target；在 KV 定价前执行
  KV：FP8 E4M3，池 1,111,168，单请求窗口 1,048,576
  mamba：20 槽，BF16 状态；调度上限 4 路

Host RAM 62G
  PLE：NVFP4 packed，28.01 GiB pinned-resident
  HiCache：13GB 配置预算，write_through / kernel / page_first

NVMe
  checkpoint 分片、PLE packed 原件、模型索引与装载清单
  PLE 启动时钉载；现役不依赖持续 SSD 行读取
```

来源：`CURRENT-PRODUCTION.md`；`launch-silver-native-20slot-cap1111168.sh`；`astra_impl_final.md`。这里 13GB 是 HiCache 参数预算，不是宣称实分配恰好 13GB，更不属于 GPU 池。早期 16GB 配置的 14.35GB host KV 记录仍保留在 §4.4，不能拿来填现役账。

### 5.1 原生、量化、共享：三件事分别验

“原生 NEXTN”解决 MTP 权重在主索引组织的问题；“在线 NVFP4”解决 routed experts 的权重格式；“early-share”解决重复 embed/head 在定价前是否已释放。三者能够组合，但收益不能各领一遍再叠加一遍。

早期草稿曾写原生 NEXTN + FP8 + 共享让 draft“净成本 3.74G → 2.21G”，也曾把独立实例的 KV/图说成会随原生组织消失。晚间审计把这两句收回：**原生仍有 draft 模型执行、draft KV 与图；一本索引不等于没有 draft 运行成本。** 同样，日志 `Load weight end` 是加载前后可用显存差，发生在共享前，不能直接称共享后的常驻独占。

| 阶段 / 路线 | 留档读数 | 现在怎样解释 |
|---|---:|---|
| 早期外挂在线 NVFP4 | 3.74GB | 当轮 draft load-end；`sglang-silver-gcfix.log` |
| BF16 直载对照 | 7.06G | 旗标路径不同；`2026-09-08.md` 白天对照 |
| 原生在线 FP8 历史 | 4.75GB；审计另记 4.75–4.77G | 加载阶段历史口径；`astra_native_nvfp4_final2.md`、`sol_threemodel_final.md` |
| mixed 外挂 / 原生在线 NVFP4 | 两者均 3.92GB | 同精度日志打平；原生没有再减成约 2.6GB |
| 原生共享后独有权重 | 约 1.49GiB | 张量审计；不含 KV、图及未细分加载余额；`astra_fixed_residency_plan.md` |

早期 FP8 档需 draft runner `triton`，当时 `Fp8MoEMethod` 不支持 CUTLASS 的分派；**最终 NVFP4 档显式用 `flashinfer_cutlass`**。旧红线属于旧量化路径，不能原封不动搬进最终启动脚本。附录 A 保留两档补丁，§10 给最终参数。

### 5.2 YaRN 与长前缀：窗口、池、有效输入各管一笔

```json
{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}
```

原始位置 262,144 × factor 4.0 = 1,048,576，与 context 参数自洽。这是配置复算；不等于全长输入已经测过。严格长测取 **786,432 输入、cached 0、92% 针深**，最终短答案返回准确；详细成绩与实例归属见 §7.5–7.7。

早期“768K”日志则有 prompt **768,306 / 768,345** 的口径（`longctx_resp.json`、`mamba10_resp.json`、`mamba10_resp2.json`；草稿附录 B），不是晚间严格的 768×1024。两类成绩都保留，长度不冒充一致。续问前缀命中说明缓存复用；冷输入冒烟说明新 prefill 可以走到底，各自回答一个问题。

---

<a id="section-6"></a>

## 6. PLE 钉内存分支线：三连败与重生

> 主线（§2–§5）回答"池子怎么变大"；本节补一条平行分支线：**PLE 大表放哪**。这条线在 24 小时内走了三幕——fp8 钉内存三连败（Route A）、页缓存流式转正（Route B）、NVFP4 量化钉内存重生（分支轮，09-08 凌晨追加 cap 1.2M 试探轮，见 §6.4）——第四幕已在 09-08 晚间接成 mixed + pinned 终态（§7），以下保留分支起点。

### 6.1 Route A：钉 fp8 PLE 三连败（法证章）

官方 qualified 路线（Route A）要求 PLE 大表 47.68 GiB 常驻 pinned host memory。09-06 23:25 – 09-07 00:07，同一条路连射三轮，死在同一堵墙上：

| 轮 | 时刻 | 钉法 | 结局 | 佐证 |
|---|---|---|---|---|
| 1 | 09-06 23:31 | 默认 `cudaHostAlloc` 整块钉 | host OOM | 日志被重试覆盖，数字入值班台账 |
| 2 | 09-06 23:42 | `malloc + cudaHostRegister` 逐页注册 + 低脏页窗口 | 被杀，anon-rss 60.0 G | `sglang-silver.log` |
| 3 | 09-07 00:07 | 同第 2 轮，缓存干净、无下载干扰 | 被杀，anon-rss 60.9 G，kernel 全局 OOM（cgroup 全 max 已排除） | `sglang-silver2.log` + `ram-monitor.log` |

留档日志的结尾一模一样——不是 sglang 报错，是内核动的手：

```text
RuntimeError: Rank 0 scheduler died during initialization (exit code: -9).
If exit code is -9 (SIGKILL), a common cause is the OS OOM killer.
```

Route A 启动脚本（`launch-silver.sh`）当时已带全套钉内存参数，"姿势不对"的解释先排除：

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
... --ple-offload-embedding
```

**根因账**：钉 fp8 PLE 47.7 G + 权重加载瞬态约 13 G + 系统及其他占用 → 撞上 62 G host 限制，kernel OOM（47.7+13 本身并不大于62；必须把第三笔账写出来）。pinned页不可换出；加载缓冲和系统仍要RAM。第3轮排除下载干扰与cgroup限制后仍失败。

结论一句话：**Route A（fp8 尺寸钉内存）三连败证死，勿回退**。这不是调参问题，是算术问题。

### 6.2 Route B 为什么赢：SSD-Stream 页缓存流式

**51.2G PLE留在NVMe外置bin，OS页缓存按需供给。**

| 维度 | Route A（pinned PLE） | Route B（SSD-Stream） |
|---|---|---|
| host 侧需求 | 47.7 G 钉死 + 加载瞬态 ~13 G | staging 2×320 MiB registered buffers + 行索引，合计 <1 G |
| 内存语义 | pinned 页不可换出，OOM 即死 | 页缓存可回收，随内存压力弹性让路 |
| 62G 排布 | 47.7 + 约13 已逼近物理 RAM，叠加系统等占用后 OOM | PLE 页缓存弹性 + L2 16G pinned + 系统 ~5G，实测稳态可行（§4.4） |

落地三件套：`sglang_ssd_stream` **0.2.0 wheel**（9 模块 = 8 Python + 1 编译 `_io` 扩展）；**SHA-256 哈希门**（manifest 摘要格式与文件长度强校验，启动日志逐表出示 `sha256=` 指纹）；**staging 2×320 MiB registered buffers**（320 MiB 为金线实测后移植的值，上游默认仅 16 MiB）。`SGLANG_PLUGINS=ssd_stream` 一键启用，与 Route A 的 `--ple-offload-embedding` 互斥。Route B 转正后的九级点火全记录见 §2。

### 6.3 NVFP4 钉内存重生（09-07 晚间分支轮）

三连败杀死的是"**47.7 G 的 fp8 表**钉不进 62 G"，不是"钉内存"这个思路本身。晚间分支轮把表砍小：**PLE 从 fp8（160 B/行）换 NVFP4（94 B/行），47.68 GiB → 28.01 GiB（−41%）**——28 G 钉进 62 G，host 预算第一次有了空间；还须叠加 HiCache、加载瞬态与系统占用。

repack 实录（`repack_nvfp4_ple.py`，逐字日志 `repack_nvfp4.log`）：primitive-ai 发布的 NVFP4 PLE 128 片 safetensors 重打包为行交错单 bin——26.7 s 完成（1074 MiB/s），SHA-256 落盘，金样行复读校验。94 B/行 = 80 B e2m1 打包码 + 10 B e4m3 逐 16 列 scale + 4 B float32 scale₂；与 fp8 bin 逐行交叉核对（覆盖全部 128 片边界的 17 行）余弦 **0.995**——NVFP4 输出已是最终精度，插件侧 scale 需中性化（置 1.0）。

钉载实测，以下沿用草稿的字段节录（路径、摘要有省略）：

```text
# sglang-silver-nvfp4.log（22:43 修正轮；22:33 首轮 16.6 s，同一量级）
SSD Stream NVFP4 resident load: 28.01 GiB pinned in 16.1s
SSD Stream enabled (NVFP4 pinned-resident): backing=...-ple-nvfp4.bin sha256=930da470...
  rows=320001536 row_bytes=94 size=28.01 GiB pin=cudaHostRegister staging_slots=2 capacity=3569620 rows
SSD Stream NVFP4: checkpoint PLE weight_scale neutralized to 1.0
Loaded 1 immutable SSD Stream table(s)
```

**机制全绿：28.01 GiB 一次性 pin 进 host RAM，约 16 秒完成；驻留路径不再逐请求从 NVMe 取表。两轮启动未闭环，不能据此声称做过长时间 IO 实测。** 但两轮启动都没走到冒烟——倒在 mamba 槽位配置上，不是倒在钉内存上：

- 第 1 轮（22:33）：重写启动脚本时 `--max-mamba-cache-size 12` 被弄丢，sglang 自动贪到 166 槽（ssm_state 8.81 G）→ 池子压到 840,064 → 余量 0.13 G → warmup OOM（首轮日志被重试覆盖，数字出自值班台账）
- 第 2 轮（22:43）：mamba 槽位落在 3，`mamba_ratio=5` 之下连一路请求都排不出，调度器拒绝启动：

```text
RuntimeError: Hybrid (mamba/linear-attention) state cache is too small to serve any requests.
max_mamba_cache_size=3, mamba_ratio=5, resulting max_num_reqs=0.
```

- 22:48 回退轮：恢复 fp8 流式配置，池子 1,136,960 复点（`sglang-silver-nvfp4-rollback.log`）——两轮未闭环，收兵回城

**分支定位（09-07 22:35 拍板）**：NVFP4 PLE pinned 是一个**分支**，不是替代品——fp8 页缓存流式、RadixArk+ple-offload 等旧配置全部保留可回退，按场景选用、互不排斥。当时状态：钉载机制已实证，端到端冒烟尚未跑到；随后完成的干净重跑和晚间组合验收续在下文。顺手入账的教训：改启动脚本必须全量 diff——丢一行 mamba 槽位，就是一轮点火。

### 6.4 cap 1.2M 试探轮（09-08 01:30，10 槽 + context 1,048,576 不变，与当时基线同参）

09-08 01:30重跑：显式mamba10、context1,048,576，cap从925,000拉到1,200,000。boot-only探针仍没走完启动，留下三组证据：

**第一件，mamba 槽位修复实证。** §6.3 崩轮的 `max_mamba_cache_size=3`（贪心联合求解被 28G pinned 挤缩）没有复发，显式 flag 生效：

```text
# sglang-silver-nvfp4-1p2m.log
[01:33:10] max_running_requests is capped to 2 by the mamba state cache (max_mamba_cache_size=10, 5 state slots per request). To raise it: increase --mamba-full-memory-ratio or --max-mamba-cache-size, or halve the state size with --mamba-ssm-dtype bfloat16.
```

10槽按5槽/路钳到两路，这一次没有复发3槽启动拒绝。

**第二件，NVFP4 pinned 机制复证 + host 账本实拍。** 28.01 GiB 照常钉入，本轮 18.6 s（前夜 16.1 s，+2.5 s 属页缓存状态差异）：

```text
[01:30:48] SSD Stream NVFP4 resident load: 28.01 GiB pinned in 18.6s
```

当轮host台账：used18→40G、buff/cache49→22G、available尚余21G（草稿§6.4）。

**第三件，1.2M 池子建成，GPU 物理天花板现形。** 池子定价走完：

```text
[01:33:10] Memory pool end. avail mem=1.27 GB
```

| 口径 | cap | Memory pool end 余量 |
|---|---|---|
| 当时基线（09-08 00:17 复启轮，同口径） | 925,000 | 4.95 GB |
| 本轮 | 1,200,000 | 1.27 GB |

+275K tokens 实耗 ~3.68 GB ≈ **13.4 KB/token 实测斜率**，与 §4 定价公式的 13,312 B/token 相互印证。池子装得下，graph 系统答不答应是另一回事——三段捕获，二过一爆：

```text
# 图捕获三段实录
[01:33:12] Capture target verify CUDA graph end. elapsed=0.78 s, mem usage=0.12 GB, avail mem=0.51 GB.
[01:33:15] Capture draft decode CUDA graph end. elapsed=1.22 s, mem usage=0.36 GB, avail mem=0.14 GB.
[01:33:15] Capture draft extend CUDA graph begin. backend=full, num_tokens_per_req=4, bs=[1, 2], avail mem=0.14 GB
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of 95.01 GiB of which 125.56 MiB is free.
```

verify ✅ → draft decode ✅ → draft extend ❌，要 128.00 MiB 时只剩 125.56 MiB——**差约 3 MB**；01:33:16 调度器收到子进程 sigquit，本轮到此为止。

**本轮结论（最重要的数据点）**：该配置下 **GPU 物理天花板在 1.15M~1.2M 之间**——池子装得下 1.2M，graph 系统差一口气；同时钉死一条边界认知：**NVFP4 pinned 与“池子上限”无关**，它救的是 host RAM 与 NVMe IO，不救 GPU 显存。后续阶梯结果补在 §6.5；这个天花板只约束当时旧栈，不能宣布所有实现的永久上限。

### 6.5 09-08 白天续完：阶梯失败、晨崩、unquant 反转

凌晨的阶梯并没有因为“只差 3MB”就顺利收官。日记覆盖的窗口不同，`2026-09-07.md` 记 01:15 起的过程，`2026-09-08.md` 汇总跨 01:55 的追加轮；全程复盘将它们记为六轮失败／未过闸，不写成六次同一死点。

| 轮次 / 时段 | 留档结果 | 判决 |
|---|---|---|
| cap 1.2M | graph 要 128MiB，余 125.56MiB | 启动未成；“差约3MB”是原日志两值的近似差 |
| 1.15M | host 预算差约 0.48GB | 不把 host 失败写成 GPU graph 失败 |
| 1.125M | warmup 差约 0.92GB | 池与图之后仍有一关 |
| 1.1M | ready 后大 prefill OOM | 启动成功不能替代长请求验收 |
| 02:32:31 | 恢复旧 1.1M fired up | D08 简记 02:29 为恢复操作；完成时刻采用 D07 留档 |
| 05:00 | GDN/mamba forward 要 80MB，余 4.25MB，SIGQUIT | 真实流量触发边界；来源未核，不坐实某个定时任务 |
| 08:49–09:18 | 回 512K，无 flag 的 draft 仍 7.06G | “上下文翻倍让 draft 变重”被证伪 |
| 09:16 C 方案 | 补 unquant 后 draft 3.74G，池 1,140,672，图后 1.86GB | 在线 NVFP4 路径回来了 |
| 10:07–10:26 | PIL / CPU transport / chunk2048 组合仍要 128MiB、余 125.56MiB | 省错阶段，没救启动关 |

出处：`2026-09-07.md`、`2026-09-08.md`；汇总见 `复盘-0907至0909-全程review.md` §1.2。C 方案时刻采用 `sglang-silver-ext-512k-unquant.log` 的 09:16:33、09:16:38、09:16:51；D08 曾误记“00:1x”，在这里保留冲突并订正。C 的图后采用 **1.86G**，旧 **2.06G** 不再作比较基准。

午前审计终于把调用链对上：`server_args.py:4674` 把 unquant 归一为 None，checkpoint 识别与显式指定状态继续影响 `weight_utils.py:241` 的在线 FP4 分支。“不量化”旗标反而引出小 draft；无 flag 则直载 BF16。专家 BF16 约 4.69G、NVFP4 packed 约 1.32G，理论差 3.369G，与当轮 7.06→3.74 的实测差 3.32G 相容。**旗标名字会骗人，量化工厂和 dtype 不会替名字圆场。**（D08 白天双问审计；复盘 §3.1。）

这也订正了旧稿的“内生唯一”“与旗标无关”：本机已有可用外挂在线 NVFP4。晚间选择原生是为了同一索引管理，而非继续追一个不存在的实例减重承诺。

PIL 组合的失败同样值得留：tokenizer 的约 676MB CUDA context 没随 PIL 消失，CPU transport 本就默认，减 chunk 针对的是运行期 scratch。日记记下 **+100K 池预算的启动实价 1.30GB**；拿尚未在启动时分配的激活去抵图捕获账，算得再漂亮也救不了那 128MiB。（`2026-09-08.md` pil2048 段。）

### 6.6 转换设计走到了哪一步

旧stage-3方案的索引闭合、PLE manifest校验与定价前共享要求保留；实际选择mixed软链顶屋，复用NVFP4 PLE表，主索引保留MTP。预计工时和目标池不再列战果，没有执行磁盘清理。（`sol_threemodel_final.md`；`gdn_fp8_impl.md`；N。）

---

<a id="section-7"></a>

## 7. 晚间战役：18:14 → 00:08，九幕收官

以下局部引用缩写：D08＝`2026-09-08.md`；A＝`astra_impl_final.md`；N＝`astra_native_nvfp4_final2.md`；NP＝`astra_native_nvfp4_progress.md`；P1＝`astra_fixed_residency_plan.md`；CP＝`CURRENT-PRODUCTION.md`。报告均列于附录 B，冒号后的数字是素材行号。日记转述与源码审计不冒充本篇重新跑测。

| 幕 | 时段 / 锚点 | 这一幕真正改变了什么 |
|---|---|---|
| 一 | 18:14 五问收口 | 三套仓的职责、组合可用性先分清 |
| 二 | Round 2，19:12 加载 | 索引可用了，“packed 必省显存”却被否定 |
| 三 | E1 | early-share 兑现；cap 把安全垫吃掉 |
| 四 | GDN FP8 审计与落地 | 复用已有 FP8 GEMM，装配比造新内核重要 |
| 五 | 20:25–21:34 | mixed 栈真 768K；1.15M 未过正式 soak，回 1M 拿 991/991 |
| 六 | 22:07–22:46 | 原生 NVFP4 MTP 补丁、12 轮 bench、长测与 soak，然后按约还原 |
| 七 | 23:09–23:29 | 切换后的 mamba 崩溃与热身纪律；前身再过独立长测 |
| 八 | 23:23 结案 | 与恢复线交错进行的 P1 解剖，3.1G 幻影收成约 1.49GiB 独有权重 |
| 九 | 00:03:50 / 00:08 | 20 槽 / 4 路 / 1,111,168 点火与定版；长压未验 |

### 7.1 第一幕：三模型五问审计，先排除错误路线

审计把 plain、mixed、PLE-quant 的定位与 target×draft×PLE 选择组织成五问、60 格矩阵：组合可用性、GDN FP8、early-share、磁盘账、分期迁移。这里三套素材仓与矩阵里的三个 target 骨架不是同一个分类：PLE-quant 是表资产仓，Garner 才是已剥离 PLE 的服务骨架。

五问交付见 D08:182；`sol_threemodel_final.md:21–79` 为矩阵，`:82–135` 为 GDN/share 核查，`:137–211` 为磁盘与分期计划；`silverline_audit_findings.md:6–16` 给可运行路径和只读审计锚点。

**判决：审计·成。** plain 迁移不划算，mixed 投影值得接，PLE-quant 不是完整服务模型。60 格不是本机 60 组跑分；packed 候选排名受后续 Round 2 反证约束。

### 7.2 第二幕：Round 2 预打包反重，索引成功不等于减重成功

把官方 `mtp_nvfp4/` 全模型 overlay 收成独立 draft 加载视图。`weight_map` 从 **302,772→7,041**，引用 shard 从 **97→4**；裁掉文件后缺的 **5 键**从父仓软链补齐。先解决“能不能读”，再看“读完是什么代价”。（D08:188；`gdn_fp8_impl.md:15` 记 19:12 r2 加载。）

packed draft 加载差 **3.88G**；日记对照在线 experts **1.26G**。这两个不是同一统计范围，3.88G 包含加载阶段其他项，1.26G 是专家路径近似口径；不能直接相减宣布多吃了多少独占显存。后续张量审计给在线 experts+scales **1.318359375GiB**，单位与包含范围也须另列。真正操作结果是：cap 顶格实际池 **1,189,952**，图后仅 **0.98G**，约 **497K** 请求首 chunk OOM；accept **0.39 vs 0.36** 未显示明显劣化。（D08:188；P1:9。）

**判决：索引成、容量闸败、预设收益证伪。** “官方已经 packed”解决的是文件表示与可审计性，不是自动胜过当前在线量化。所谓“质量无罪”限于当轮 accept 对照；失败归于该配置内存水位，不据此证明跨任务质量等价。

### 7.3 第三幕：E1 early-share 兑现，cap 顶格把安全垫吃回去

E1 回到 share＋unquant 正主路径，cap **1,190,000**，检验名义共享收益能否支持大请求。结果图后 **1.21G**，768K 闸门约 **6 秒**就 OOM：要 **128MiB**，只余 **122MiB**。（D08:189。）

**账要这样算。** E1 池后 **3.78G**，C 池后 **2.01G**，裸差并不是 2.37G；两者池规模不同，要加回转入池子的那部分。更直接的无 cap A/B 是 **1,140,672→1,320,704**，增加 **180,032 tokens**，与约 **2.37G** 在当轮定价斜率下相容。共享函数已有删除旧权重、empty_cache 和 synchronize，“还缺一次清缓存所以收益没兑现”被源码与池增量共同反证。（`silverline_audit_findings.md:12–13`。）

**判决：share 成、顶格策略败。** 当轮归档把图后 **1.21～1.86G** 记为死亡边界观察区间；C 图后必须用 **1.86G**，**2.06G 是旧记录**。这不是任意模型、任意负载都适用的精确安全阈值，尤其 C 仍是 512K 上下文，不能伪造它也过了严格 768K 对照。

**闸门长度补注。** E1 启动日志 `sglang-silver-e1-earlyshare-unquant.log:97`仍是 context **524,288**；D08 称“768K 闸门”，只按实验名保留。未核输入是否被截断，不能把 E1 写成严格 786,432-token 零缓存验收。

**E1b 留白。** cap 约 **1,130,000**、让利转安全垫约 **2.4G** 是待验分支，晚间没有跑。后面的 mixed 终态取得更好的组合结果，不等于 E1b 的独立假说已被验证。（D08:190；N:22,25。）

### 7.4 第四幕：GDN FP8 零新增补丁，拼装 mixed 顶屋

两轮源码核查把 FP8 的边界钉在投影 GEMM：SSM 消费投影输出与状态，不直接解释 FP8 权重；SM120 有 `launch_sm120_fp8_scaled_mm`。mixed 自带 compressed-tensors，旧 target `--quantization modelopt_fp4` 必须去掉，否则兼容表冲突直接 ValueError。draft 量化是另一作用域，不应跟着 target 旗标一起误删。（`gdn_fp8_impl.md:9–23,35–43`。）

最小路线是 mixed 权重软链成新顶屋，编辑 index 去掉内嵌 PLE 条目，复用已有 manifest 与 pinned 表。报告估工约 **30 分钟**，这是方案估计，不是全战役实耗。投影文件约 **2.68GB** 对 BF16 约 **5.3GB**，审计净省约 **2.6GB**；战略报告同资产给 **2.487GiB**，属于不同单位/精度口径。（同稿:47–56,78；D08:117。）

**判决：审计可行，随后实测落地。** GDN 本身不需要新补丁；新顶屋完成 **64 个软链总数**，其中台账先记 **53 个权重软链**，并非互相矛盾。原仓只读零删改。也不能把“GDN 零新增补丁”扩大成整个 silverline 工作树无本地补丁。（D08:197；`astra_impl_progress.md:3`。）

### 7.5 第五幕：终态栈真 768K 与 991/991，冲顶后纪律回退

20:25 开始分步验收后，P1 从 cap **1,050,000** 开始，实际池 **1,049,984**、图后 **2.87GB**，没过预设余量门就不提交 768K；降到 cap **1,000,000**，图后 **3.52GB**，真针通过，约 **147.681s**，但有可恢复 allocator 告警。此前“能启动就直接压满”的习惯在这里被门禁挡住。（`astra_impl_progress.md:11–16`。）

P2 首次 smoke 又死在 backend：自动选到 draft NVFP4 不支持的 `FLASHINFER_TRTLLM`。按纪律先回 C，再显式 `flashinfer_cutlass` 重试。第二轮池 **899,968**、图后 **6.83GB**，QSA12/GDN36 FP8 和 64K 针测通过；抬回 1M 后，**20:53 的真 768K PASS：786,432 输入、cached 0、143.397672s≈143.4s、spec_accept_length 3.875**。allocator 重试告警仍有 **28 次**，所以写“无致命异常”，不写“全程零 OOM 字样”。（A:10–11,128；`astra_impl_progress.md:18–23`。）

随后 **1.15M 冲顶**：21:14:45 ready，cap **1,150,000** 对齐成池 **1,149,952**，图后 **3.73GB**；768K 再过，**143.384068s**。正式 soak 21:18:53 开始，约 **4.413s** 被严格 harness 中止：匹配到 **7 条**“avoid CUDA OOM”预加载提示；复核新增真实 allocator OOM **0**、未崩未重启，当时只余 **0.27GiB**。针测阶段另有 **73 次**可恢复告警，不能用 soak 窗口的零覆盖整轮。（A:12–14,182–219。）

**判决：冲顶正式验收未完成，回退有据，不虚称顶格全绿。** 第二轮回到 1M，**21:22:58** fired up，图后 **5.68GB**；当前恢复轮补做 64K，正式 **600s soak 991/991** 成功、零致命、零重启，存在双 worker 成功重叠。该恢复 PID 没有重跑 768K；1M 同配置旧轮有证据，归属要写清。（A:121–128,223–257。）

性能用同题口径记账：外挂终态 accept 中位约 **2.207**、decode wall TPS **142.4**；C 基线约 **2.266 / 148.4**。C 与 mixed 终态还改变了 target 和 context，不能把差值全归给 GDN FP8。最终仍有 draft **3.92G＞3.8G** 的严格门禁黄灯。（N:15–25；A:5,172–178。）

### 7.6 第六幕：原生 NVFP4 MTP，2.5 分钟补丁与 p=0.63 判决

22:07:17 开始，22:09:44 配置测试由缺 override 到通过，约 **2.5 分钟**完成局部实现与接线；不是 2.5 分钟做完全部验收。原生一本索引共 **296,502** 条，含 **31 条 MTP 映射**，用显式 NVFP4 开关接已有在线量化工厂，只量化 routed experts，非专家保留 BF16。（NP:3–7；N:52–54。）

第一轮 draft 仍 **3.92GB**，未达原先约 2.6GB 目标，也没过 3.8GB 严格门，按指令还原。第二轮获准带黄灯做完整功能验证，不放宽显存门槛：同题 **12/12** bench，accept 中位 **2.20216 vs 2.20690**，decode wall **141.677 vs 142.386 TPS**；探索性置换检验 **p=0.6286≈0.63**，判为与轮间噪声相容。（NP:8–18；N:29–33。）

22:32 真 768K 通过：**786,432 tokens、cached 0、143.365s**，新随机口令准确返回；22:42 正式 **600s soak 985/985 PASS**。22:46:30 仍按本轮约定恢复外挂，**恢复文件核验一致**；这份终报截止时“未切原生”，不能跳过它，直接写原生从第一次实验一直在役。（N:56–64；NP:45–51。）

**判决：功能链成、显存打平、部署账简化。** “双过”指 768K＋soak，区别于 23:24“双 768K”。p=0.63 更不是 accept=0.63，也不证明统计等价或全任务质量无损。正式切换发生在此后后续正式切换阶段。

### 7.7 第七幕：切换与并发崩，mamba assert 找到了业务触发条件

原生终报在 22:46 仍是“测试后恢复外挂”；后续才正式切换。**23:09，服务报 `Can not alloc mamba cache`。** 已经 fired up、已经证明原生长测可用，真实请求和一条大针测叠在一起，照样可以把服务打倒。

事故原文，`sglang-silver-native-prod.crash1_20260908_2309.log:500–502`：

```text
    assert slot is not None, "Can not alloc mamba cache"
AssertionError: Can not alloc mamba cache
```

CP 把这次事故归档为 **真实流量 × 768K 针测并发 × 冷分配**。当轮启动是 **10 槽**，SSM 日志 **0.58GB**；不能把最终 20 槽的 mamba cache **1.16G**倒填回事故现场。并发槽位耗尽与冷分配压力各占多少，没有隔离实验，不装作已经测出百分比。

05:00 晨崩也是边界流量，但当时是显式 GPU OOM；23:09 则是 slot 为空后的断言。两次同属“ready 以后才支付运行峰值”的事故家族，不能合成一条假栈。修法先落在操作纪律：**冷炉先 mini 热身，再接大 prefill；768K 级测试挑没人使用的窗口。** 调度拒新请求或预留槽位的源码加护仍是待办。

恢复后，CP 记 **23:24 双 768K PASS**，属于 10 槽 / cap1M 原生前身。“双”按交接记录保留，不扩写成两个严格 786,432-token 输入同时成功。**23:29 第三次独立针测**则有更精确的原行：

```text
[2026-09-08 23:29:09] ReqTimeStats(rid=4b5a1374593f4fddbf07d48d62d67a76, input_len=786432, cached_input_len=0, output_len=33, attempts=0, type=unified): queue_duration=0.98ms, initial_prefill_elapsed=142605.21ms, post_prefill_elapsed=257.57ms, forward_duration=142862.78ms, entry_time=1788881206.969
```

出处：`sglang-silver-native-prod-r2.log:881`；CP 另记 accept rate **0.61**。**142.6s 是 initial prefill**；整段 forward 是 142,862.78ms，不能把 prefill 值冒称客户端总墙钟。第三次 PASS 补强前身证据，没有替最终扩槽版完成验收。

### 7.8 第八幕：P1 解剖，3.1G 幻影变成约 1.49GiB 独有权重

这条审计与恢复线交错进行，**23:23 结案**。当时的问题是：3.92G draft 减掉 experts 后，是否还有约 3.1G 固定项可以砍？答案先纠正题目：3.92 是共享前、缓存分配前的 load-end；连减法的统计范围都没对齐。

`astra_fixed_residency_plan.md` 按 31 条 MTP 索引、张量头部、构造和量化代码做静态审计，再对启动日志。没有附加运行中的进程取 storage 快照，因此下面高精度数字是**张量与源码复算**，不是十位小数精度的显存仪表实测。

| 项目 | GiB | 属于哪一阶段 |
|---|---:|---|
| routed experts packed 权重 | 1.171875000 | 独有权重 |
| experts FP8 block scales | 0.146484375 | 同上，已计入 experts+scales |
| experts 合计 | **1.318359375** | 不能再另外重复加一份 swizzled scale |
| 其余 29 条 MTP BF16 权重 | **0.168696880** | 非专家独有权重 |
| 构造时 embed/head 副本 | **2.368164063** | 共享前在场，early-share 后释放重复 storage |
| 共享前可解释权重小计 | 3.855220318 | 与 load-end 3.92 有差额 |
| 未细分加载余额 | 0.064779682，约 **66.33MiB** | 未逐 buffer 证明，不能指定给 RoPE 或某个可删缓存 |
| early-share 后独有权重 | **1.487056255，约1.49GiB** | experts + 非 experts |
| 余额全保守留给 draft 的阶段增量 | 约1.551835938 | 推导值；不是另采样的运行时峰值 |

日志标签写 `GB`，P1 查计量函数实际除以 2^30，故这里分解用 GiB；其他原文摘录保留原标签，不擅改日志。

共享是真的别名绑定：先删 draft 的旧 embed/head 参数，再绑定 target 的现成参数，随后 empty_cache 与 synchronize。target 那两块权重当然仍在；消失的是 draft 重复副本。共享发生在图捕获前，不能再把“图锁着旧 head”当没有证据的解释。

**专家以外只有约 0.169GiB 独有权重，哪里还藏着可安全拿走的 3.1G？** 相对保守的 `o_proj` FP8 候选也只约 **15MiB**理论权重收益，accept 风险与新增 workspace 尚未实测，不值得为了 GiB 目标动刀。其余 norm、router、HC、QSA 离散选择都有各自语义，不能把 ignore 清单一删就宣布全模型量化。

KV、mamba、图是后续阶段另一笔账。P1 在 10 槽 / cap1M 原生日志中列出 draft KV payload 约 **0.953674GiB**、两段 draft 图 **0.36+0.20GB**；它们说明原生仍有运行成本，却不该被拼回共享前 3.92G 的分解里。**“原生一本账”成立，“原生自动再省一大块显存”撤回。**

### 7.9 第九幕：20 槽、4 路与吉利数，00:08 收官

最后把 mamba **10→20**，请求上限 **2→4**，为新的并发预算点火。cap 先取 **1,111,143**，实际页对齐成 **1,111,104**；再改 **1,111,168**，恰好 **64×17,362**。名义只加 25，实际多拿一整页 64 tokens。这是页对齐复算，原池值分别见 `sglang-silver-native-cap1111143.log:209` 与最终日志。

最终同一轮逐字启动证据，`sglang-silver-native-cap1111168.log:142,155–156,208–209,239`：

```text
[2026-09-09 00:02:39] Load weight end. elapsed=116.95 s, type=Qwen4ExpForConditionalGeneration, quant=compressed-tensors, avail mem=22.50 GB, mem usage=71.68 GB.
[2026-09-09 00:03:25] Load weight end. elapsed=45.41 s, type=Qwen4ExpForCausalLMMTP, quant=compressed-tensors, avail mem=18.44 GB, mem usage=3.92 GB.
[2026-09-09 00:03:25] NEXTN early embed/head sharing done before KV pool sizing (draft Qwen4ExpForCausalLMMTP now aliases target tensors).
[2026-09-09 00:03:35] Capture draft extend CUDA graph end. elapsed=0.54 s, mem usage=0.22 GB, avail mem=3.57 GB.
[2026-09-09 00:03:35] max_total_num_tokens=1111168, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=4, context_len=1048576, available_gpu_mem=3.57 GB
[2026-09-09 00:03:50] The server is fired up and ready to roll!
```

注意 draft 外层 load-end 仍写 `quant=compressed-tensors`，原生局部 override 与在线 experts 的 NVFP4 由另一条路径证明，不能为了图好看把这行改成 `quant=modelopt_fp4`。

00:08 的 `CURRENT-PRODUCTION.md` 定版：health **200**、图后余 **3.57G**、mamba cache **1.16G**、**3×200 OK**；验收备注另记总共 **4 条 mini 热身**。两项照录，不把两种计数悄悄合成“四次新增独立测试”。该文档风险备注还留 **3.53G**旧水位，正文用最终日志支持的 3.57G，风险水位保留 **3.53～3.57G**双口径。

收官的 PASS 是点火和 mini 热身。**768K / soak 未在 20 槽、4 路、1,111,168 上跑；前身双 PASS 不转借。** 扩槽本身也没有修好 assert：更多槽同时允许更多请求，激活峰值跟着变。下一轮先补并发长压与持续负载，再决定是否继续抬 cap。

---

<a id="section-8"></a>

## 8. 坑清单：现象 → 根因 → 修法

| 坑 | 现象 | 根因 / 证据边界 | 修法与当前状态 |
|---|---|---|---|
| **unquant 语义反转** | 无 flag 7.06G，补回后 3.74G | `server_args.py:4674` 归一化、显式标记与 `weight_utils.py:241` 在线分支共同作用；不是 context 翻倍 | 外挂补回原 flag；原生用 `SGLANG_NEXTN_MTP_NVFP4=1` 明确局部意图。通用参数语义尚未修复 |
| **预打包反重** | Round 2 3.88G > 日记在线 experts 1.26G，顶格后首 chunk OOM | 3.88G 是加载段，1.26G 是 experts 近似；文件 carry、非专家、构造开销与在线量化基线混账 | 索引修好再测；放弃“packed 自动更省”的预期，回已有在线路径。不得直接相减称独占增重 |
| **cap 顶格物理** | packed 图后0.98G、E1图后1.21G失败 | 池与运行工作区共用显存；共享收益全部转池，安全垫不会再多一份 | cap 降档、先验余量，再热身与长测。死亡区间是观测归纳，不是通用安全常数 |
| **冷炉大 prefill** | ready 后晨崩；夜间并发再次暴露边界 | 惰性 kernel / workspace 与长输入峰值尚未在启动阶段付清 | mini 热身后再接长输入，压测选空闲窗口；热身不保证覆盖所有 shape |
| **并发 mamba 断言** | `Can not alloc mamba cache` | 当树5槽/路；真实流量、投机状态和缓存生命周期交错，分配返回 None | 20槽/4路已配置；调度拒新或预留槽位待做，扩容未等于断言修净 |
| **draft 3.92G 口径** | 以为还有约3.1G固定驻留可砍 | 3.92G是共享前 load-end，2.368GiB embed/head稍后已释放；KV/图在更后阶段 | 共享后独有权重约1.49GiB；66.33MiB余额保留未细分，15MiB候选不预支为净收益 |
| **mixed 全局 quant flag** | 保留 `--quantization modelopt_fp4` 会 ValueError | target config 是 compressed-tensors，兼容表不接受强行覆盖 | 去掉旧 target flag，靠 config 自动识别；draft 局部量化分开验 |
| **auto backend** | P2首次选 `FLASHINFER_TRTLLM`，NVFP4 draft smoke失败 | 自动分派与当前 draft 路径不兼容，不是 GDN FP8 recurrence 坏了 | 按纪律回C，再显式 `flashinfer_cutlass`。旧FP8 draft的triton规则不搬到新NVFP4档 |
| **OOM 字符串误判** | 1.15M soak 4.413s停，命中7条提示 | “avoid CUDA OOM”预加载文字被宽泛正则视为致命；但0.27GiB低水位也确实存在 | 保留原FAIL并追加根因更正；回1M。未完成600s不补发PASS；harness分类修正待办 |
| **PIL / 小 chunk 省错阶段** | chunk2048仍申请128MiB、余125.56MiB | tokenizer context未消失；CPU transport原就如此；运行期scratch不能抵未发生的启动账 | 先标设备与阶段，以+100K启动实价1.30GB校准，不再用纸面激活收益救graph |
| **自动 mamba 槽位** | 丢flag后166槽/8.81G；另轮只3槽，一路都排不出 | 默认求解受整体预算影响，未按业务并发固定 | 脚本全量diff，显式槽位，启动核验5槽/路与实际max requests |
| **FR map 语料** | 中文accept降到0.06–0.28，106.5 tok/s | 64K map仅13个CJK token，压缩候选覆盖错误 | 现役关闭；中文/领域map重建后独立A/B；机制未被判死 |
| **pkill 模式自匹配** | 日记记两次把自己停掉 | `-f`模式也出现在执行shell命令文本里 | CP采用方括号模式 `[s]glang serve`；还须核目标端口、模型路径和进程树。方括号只解自匹配，不保证不会误停别的同名服务 |

逐项出处：前六项见 D08:38–42,188–190、CP 事故记录、P1；mixed/backend见 `gdn_fp8_impl.md` 与 A；harness见 A:212–219；PIL见 D08:53–64；槽位/FR/pkill见 `2026-09-07.md:55,64,116`、CP 回滚模板。这里保留的是工程坑，不根据未隔离的现象编造唯一底层原因。

gcfix证明有可回收的加载期占用；空帧快照未逐对象指认全部引用环。

---

<a id="section-9"></a>

## 9. 方法论：让下一轮从账本继续

### 9.1 对照实验可以推翻昨天的结论

fp8→unquant 起步池不动，否定当轮瞬态猜想；512K 无 flag 仍7.06G，否定 context 膨胀；补 unquant 恢复3.74G，把变量收敛到加载路径。PIL组合失败推翻纸面节省；原生NVFP4打平外挂，推翻“一本索引自然省一大块”的承诺。报告可以错，后面的实验必须有权改判。（两日日记；N。）

组合实验也能做，但得承认它回答什么。C 与 mixed 终态同时改 target、context 和内存安排，148.350→142.386 TPS 不能全归给 GDN FP8；原生与外挂的同题比较更接近问题本身，但基线只有3轮，原生12轮连续同题，也不能升级成全面质量等价。

### 9.2 净值公式：accept 要乘执行频率，显存收益只能花一次

投机解码的直觉账是：

```text
TPS ≈ accept_length × steps/s
```

这里 `accept` 指每个验证步实际贡献的平均输出 token 数，**不是 accept rate**；steps/s 与 accept_length 必须来自同一负载、同一统计窗口。这是解释吞吐的近似关系，不是拿任意日志两列相乘就能算出的测量值。draft 更激进可能降低 accept，也可能更快；只盯其中一边，净值就会算错。本篇性能表直接用报告定义的 `completion_tokens / (request_finished_ts - prefill_finished_time)`，不拼造未记录的 steps/s。（N 的 bench 口径；公式为本文方法说明。）

显存也有一条同样朴素的账：

```text
可分配净让利
  = 新增池子的显存成本 + 留作余量的增量 + 新增固定/运行开销
```

early-share 约2.37G、FP8投影约2.6GB／2.487GiB，不能既全算入池增量，又全算成安全垫。每笔先标设备、阶段、GB/GiB、是否共享，再比较。旧13,312 B/token简式和晚间+100K实价1.30GB各有环境；不挑一个对自己结论最有利的斜率。（sol五问；D08；P1。）

### 9.3 量化死亡边界，别把 ready 当生死线

packed图后0.98G失败、E1图后1.21G失败、C归档1.86G，留下的是1.21～1.86G观察区间。C只有512K、E1长测的实际输入未核，所以它不是严格二分搜索得到的临界值。它的作用是让后续实验先看水位，知道“共享成功”以后还得决定钱花在哪里。

最低验收顺序落成：配置/容量门 → mini热身 → 真长输入 → 持续负载。**786,432输入、cached0、新随机针、92%深度、答案准确**才是晚间768K生死线。续问命中再快，也不能替冷输入过关；短答案成功不能替满1M输入、长篇输出或两条768K并发过关。（A、N。）

### 9.4 门禁三方一致：脚本、日志、live

`check-silver-astra-final.py` 核 cap、池和 live 配置，防脚本名与实际参数不一致、防页对齐丢尾页、防检查到了别的实例。报告记 parser修正后 **37/37**用例通过，那是门禁解析测试通过，不是模型所有业务通过。（`astra_impl_progress.md`；D08:197。）

每轮证据绑定配置和进程身份，保存测试窗口日志offset。外挂991/991、原生985/985分别入账，旧PID的768K不能说新PID刚测过。最终20槽版的mini同样只能证明mini。**功能PASS与draft≤3.8GB严格门黄灯可以并存**，不能偷偷抬门槛让报表全绿。

### 9.5 回滚三档，要能恢复到已知配置

按00:08定版，由近到远保留：

1. 原生10槽/cap1M/两路：`launch-silver-native-nvfp4-mtp.sh`，前身有长测证据。
2. 外挂mixed终态：`launch-silver-astra-final.sh`，22:46曾逐项恢复核验。
3. C方案512K基线：`.bak_20260908_cplan_live_preR2`及对应unquant配置。

备份不止留文件名：记录模型索引、补丁、启动参数和日志归属，恢复后核health、池、context与实际路径。1.15M因正式soak未完成而退回1M；后验发现预警误匹配，可以更正失败原因，不能倒改成当时已经PASS。（CP；A；N。）

### 9.6 边写边落盘，诚实验收

每一节写完即存，实验也按同样粒度留档：配置、日志名、阶段水位、判决、回滚状态。结论变了追加更正，别覆盖事故原件。这样下一次接手能看清“先误判、后纠正”，不用从一段残缺回忆重新猜。

---

<a id="section-10"></a>

## 10. 复现指南：最终原生 NVFP4 路线

### 10.1 环境与资产前置

本文复现的是本地补丁树 `b21f03306` 的历史环境，不是承诺任意最新版直接兼容。草稿环境为 torch **2.13.0+cu130**、flashinfer **0.6.17**、sgl-kernel **0.4.6.post1**、triton **3.7.1**、xgrammar **0.2.1**；CUDA toolkit **13.3**、GCC **15.3**。`MAX_JOBS=1` 是本机 JIT 内存约束的实证选择。版本出处为 `README.md` §7.1，本次写作未安装或更新依赖。

模型准备按已有装配报告复现：以 mixed 的 compressed-tensors config、carry/fp8-tail/ct-experts 分片组成新顶屋；index去掉内嵌PLE大表键，保留PLE支撑张量；`ple/`和`ssd-stream.json`复用已验证NVFP4 packed表。原生索引最终 **296,502条，其中31条MTP映射**，31条磁盘MTP权重是BF16源，在线只量化routed experts。（`gdn_fp8_impl.md` 路a；N；P1。）

装配检查必须核软链目标存在、index映射闭合、manifest dtype为nvfp4、packed bin长度 **30,080,144,384字节**、摘要与manifest一致。不要把 `mtp_nvfp4/` overlay原目录直接当独立模型，也不要把原生索引裁掉31条MTP。最终QSA/GDN FP8依赖原mixed分片与scale，不能凭参数名现场重新量化替代。（最终启动脚本预检；N。）

### 10.2 环境变量清单

以下按 `launch-silver-native-20slot-cap1111168.sh` 抄出最终值，路径变量由复现者填自己的安装位置。`CKPT_DIR`指原生mixed顶屋，`VENV_DIR`指flash环境，`TOOLKIT_DIR`和`GCC_DIR`为工具链，`SILVERLINE`为服务工作目录。这里只展示复现材料，本次写作没有运行这些命令。

```bash
export CUDA_HOME="$TOOLKIT_DIR"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC="$GCC_DIR/bin/gcc" CXX="$GCC_DIR/bin/g++"
export CUDAHOSTCXX="$CXX"
export PATH="$CUDA_HOME/bin:$VENV_DIR/bin:$GCC_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$GCC_DIR/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=12.0

CACHE_BASE="$SILVERLINE/cache"
export XDG_CACHE_HOME="$CACHE_BASE"
export TORCH_HOME="$CACHE_BASE/torch"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torchinductor"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export CUDA_CACHE_PATH="$CACHE_BASE/cuda"
export FLASHINFER_WORKSPACE_BASE="$CACHE_BASE/flashinfer"
export SGLANG_CACHE_DIR="$CACHE_BASE/sglang"
export SGLANG_JIT_CACHE_DIR="$CACHE_BASE/sglang/jit"
# 最终脚本后一次赋值覆盖此前 CACHE_BASE/huggingface。
export HF_HOME="$SSD_STREAM_HF_CACHE"

export PYTORCH_CUDA_ALLOC_CONF="pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
export SGLANG_NUMA_BIND_V2=false
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MAMBA_CONV_DTYPE=bfloat16
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export MAX_JOBS=1
export SGLANG_NEXTN_EARLY_SHARE=1
unset SGLANG_NEXTN_MTP_FP8
export SGLANG_NEXTN_MTP_NVFP4=1
export SGLANG_LOGGING_CONFIG_PATH="$SILVERLINE/astra-logging.json"
export SGLANG_PLUGINS=ssd_stream
```

`SSD_STREAM_HF_CACHE`对应最终脚本已有的SSD-Stream Hugging Face缓存目录；不是另一个模型版本。缓存子目录须预建，日志配置文件须存在。allocator不带`expandable_segments`；NVFP4与FP8两个MTP开关不同时开。模型、manifest和工具链预检都通过后，最终脚本先调用`astra-verify-rope.py`核effective HF配置，再启动服务。

### 10.3 原生启动全参数

这是路径参数化后的参数清单，模型与推理数值取自最终脚本。认证变量由部署者提供，不在文中保存值。没有外挂draft路径，也没有旧`--speculative-draft-model-quantization unquant`；target不再强制`--quantization modelopt_fp4`。

```bash
TARGET_OVERRIDES='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'

"$VENV_DIR/bin/sglang" serve \
  --model-path "$CKPT_DIR" \
  --load-format safetensors \
  --served-model-name Qwen3.8-27B \
  --api-key "${SERVICE_API_KEY:?}" \
  --host 0.0.0.0 --port 8000 --tp 1 \
  --dtype bfloat16 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.992 --max-total-tokens 1111168 \
  --context-length 1048576 --json-model-override-args "$TARGET_OVERRIDES" \
  --page-size 64 --max-running-requests 4 --sleep-on-idle \
  --chunked-prefill-size 4096 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 \
  --enable-hierarchical-cache --hicache-size 13 --hicache-host-memory-mode cache \
  --hicache-write-policy write_through --hicache-io-backend kernel \
  --hicache-mem-layout page_first --hicache-storage-prefetch-policy timeout \
  --max-mamba-cache-size 20 --gdn-mtp-cache-mode none \
  --moe-runner-backend flashinfer_cutlass \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --mamba-track-interval 64 \
  --trust-remote-code \
  --chat-template "$CKPT_DIR/chat_template.jinja" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --watchdog-timeout 1800
```

最终日志另显示 `max_prefill_tokens=16384`，这是该轮运行时值，不虚增一条原脚本没写的显式flag。`max_running_requests=4`与mamba20对应；如果只改请求数不补状态预算，就会回到早期被钳路数或状态不足的坑。

### 10.4 补丁怎么对应到这条路线

| 补丁 | 文件 / 作用 | 验证与边界 |
|---|---|---|
| poolfix / draftfix | `load_model_utils.py`：插件与draft路径也在定价前清瞬态 | L3合并去expandable收益+38,848 |
| estfix | `kv_cache_configurator.py`：none不扣未分配的intermediate SSM | 三处分支，L4 +49,920 |
| gcfix | 同load工具：empty_cache前gc.collect | L5 +528,640；不声称找齐每一个引用环 |
| fp8mtp历史档 | `qwen3_5_mtp.py`：局部FP8配置、ignore映射，保护KV量化路径 | 旧FP8运行条件与backend，最终不开 |
| early-share | `eagle_worker_v2.py`：定价前调用共享；后续避免无意义重绑 | 约2.368GiB重复embed/head只释放一次 |
| **NVFP4新档** | `qwen3_5_mtp.py::_mtp_quant_config`：显式开关优先接已有在线工厂 | BF16-source routed experts走W4A4，其余BF16；原生显存未优于外挂 |

关键新hunk调用`make_modelopt_fp4_online_config_from_fp8`，工厂名字含fp8，但传入`quant_method="unquant"`是为了选择BF16-source在线路径。ignore保留attention、embed/head、router、HC、shared expert和PLE；没有改target的compressed-tensors config。不能从函数名推断整个draft都成了FP8，也不能从NVFP4开关推断所有dense层都4bit。（`astra_native_nvfp4_mtp.diff`；P1工厂覆盖性审计。）

---

<a id="section-boundary"></a>

## 11. 池的边界模型：fraction 退化为护栏（09-10/09-11 续测）

> 前九级点火解决“怎么把池做大”；本节解决“池该停在哪、余量该给谁”。全部数字来自 docker 化现役栈（`sglang-qwen3.8-flash`）的真实重启轮，预测先行、实测在后。

### 11.1 引信：0.992 下的视觉请求崩循环

现役 0.992 时一张 2560² 画板图进入 transformers 图像预处理，瞬时分配失败反复 500，最终拖崩 scheduler 主循环 → SGLang 自 shutdown（SIGQUIT → 等 60s coredump → `kill_process_tree` → **exit 0**）→ `restart: unless-stopped` 自动重启循环。取证口径：`docker inspect --format '{{.RestartCount}} {{.State.StartedAt}}'` + `docker events`——**exit 0 紧跟 start 是应用自杀，不是人为重启、也不是容器 OOM-kill**；根因看超时块之前最后一条非噪音异常，不在超时行本身。

### 11.2 边界公式

`池 = min(token锁, (95.59 GiB × fraction − 固定开销) ÷ 单token KV)`

固定开销 = 权重 75.60 GiB + CUDA graphs/杂项 1.46 GiB，另有 ~1.9 GiB 池外占用者（驱动 context、mm worker）——反推空闲必须减掉。单 token 实测 13.05 GiB ↔ 1,111,168（≈12 KB，fp8 KV + Mamba/GDN 态）。

### 11.3 fraction 扫描：锁存在时它是护栏不是旋钮

| fraction | 池 (GiB) | 池 (tokens) | 启动动态空闲 | 结果 |
|---|---:|---:|---:|---|
| 0.992（事故态） | 13.05 | 1,111,168 | ≈0（理论 ~1.9，graphs 后归零） | ❌ 大图即崩 |
| 0.98（现役基线） | 13.05 | 1,111,168 | 3.57 GiB | ✅ |
| 0.97 / 0.96 | 13.05 | 1,111,168 | 3.57 GiB（与 0.98 完全等价，各实测一轮） | ✅ 无增益 |
| 0.943 | 13.05 | 1,111,168 | ~0 | 临界擦线 |
| 0.94 | 12.80 | 1,089,643 | — | 池开始等比缩水 |
| 0.93 | 11.84 | 1,008,249 | — | −10 万 token |

显式 `--max-total-tokens` 压过 fraction 的自动推导后，**池只认锁，fraction 只在跌破 ~0.943 后才开始削池**。事故修复（0.992→0.98）的实质是恢复动态余量到 3.57 GiB 而池分文未动；继续往下降 fraction 换不来余量。基线定 **0.98**：高于临界点、给 PLE LUT / hicache 元数据 / Mamba buffer 留够 SGLang 自身预留。

### 11.4 锁扫描：提锁 1:1.3 GiB/10万token，「第一张能过」不是验收

fraction 0.98 定值：

| 锁 | 池 (GiB) | 启动空闲 | 大图前空闲 | 结果 |
|---|---:|---:|---:|---|
| **1,111,168（现役）** | 13.05 | 3.57 | 2.62 | ✅ 连打 7 张全过（含 2796×2002、2.2 MB），图后钉住 909 MiB |
| 1,170,006 | 14.12 | 2.81 | 1.86 | ⚠️ 第 1 张过但图后仅剩 **131 MiB**，第 2 张（不同宽高比）500 |
| 1,211,168 | 14.22 | 2.27 | 1.32 | ❌ 第 1 张即 500 + scheduler 自杀重启 |
| 1,311,168 | 15.39 | 1.24 | — | 贴图必炸 |
| 1,572,864 / 2,097,152 | 18.47 / 24.62 | −1.83 / −7.99 | — | 起不来 |

- 边际成本由两组实测点反推：**每 +10 万 token = −1.30 GiB 空闲**（≈13 KB/tok，含池内 Mamba/GDN 缓冲，比纯 KV 口径略陡），启动/运行两口径一致。
- **多图可用的真实判据是「图后剩余 ≥ ~0.8 GB」**——第一张图的缓存高水位块不能被形状不同的第二张完全复用，所以 1,170,006 数学上单张通过也不算可用。现役 1,111,168 恰是这条线的上界。
- 真实流量峰值仅占池 5%（~5.5 万 token）：提锁不服务日常，只服务「单条干满 1M 且并行还有长会话」，而那需要 +11.6 GiB 池，单卡物理不可行。
- **摘锁是回退不是释放**：自动推导吃满 fraction 预算 → 动态余量塌到 ~0.5 GB → warmup CUDA OOM。锁必须保留。
- 高水位不随请求释放（torch caching allocator 复用），只有重启回启动值——所以报余量必须「启动/运行」两口径分开。

### 11.5 HiCache 的天花板是 host 内存不是显存

宿主 62 GiB 三分账：PLE 表钉死 28 + hicache L2 + ~21（OS/worker/page-cache）。`--hicache-size 16` 在 pool 构建期直接 `ValueError: Not enough host memory available`——**13 GiB 封顶**（swap 已用 ~3 GB，host 侧本就偏紧）。L3 file 后端另挂 150 GiB SSD 目录（env 变量配路径，`--file-storage-path` 是另一个参数别混用），写策略必须 `write_through`（write_back 重启后命中率归零）。附一条容器坑：runtime 镜像缺 openssl 头会让 HiCache 的 native_hash JIT 编译崩在 scheduler 加载期，表象是 warmup 600s 超时重启循环——ro 挂宿主 `/usr/include/openssl` 与 `libcrypto.so` 即愈。

### 11.6 大图的根治旋钮

不是 fraction、不是锁，是 **`SGLANG_IMAGE_MAX_PIXELS`**（fork 内 `qwen_vl.py` 读取，编码前把尺寸夹到上限）。默认高，故 2560² 满配吃 ~1.8 GB 瞬时；压到 ~602112（≈777²）峰值降到几百 MB，代价细字发糊。本轮结论（作者拍板）：88 视觉在 0.98/1,111,168 下已能扛大图连打，识图模型引进线终止；密集小字画板改由本机 oMLX 的 Qwen3-VL-8B 承担（三方同图对比 11/12 > 8/12 > 7/12）。

### 11.7 方法论补条

扫任何旋钮前，先用 11.2 公式把「池 / 启动空闲 / 图前空闲」三个预测值写进记录再重启——本轮 fraction 两点、锁两点预测全命中，模型才继续可信；事后拟合的解释不算结论。响应事故的调参，必须重放原失败请求类才算修好（当年只测 /health 就报“修好了”，重放现形）。

---

<a id="section-11"></a>

## 12. 现役配置快照与未完成项

**唯一现役口径：`CURRENT-PRODUCTION.md`，2026-09-09 00:08定版。** 本节是这份归档的公开快照，不根据文件名猜配置，也不把写稿过程中后续实时流量纳入截止00:08的战果。

| 项目 | 最终值 / 状态 |
|---|---|
| 启动脚本 / 日志 | `launch-silver-native-20slot-cap1111168.sh` / `sglang-silver-native-cap1111168.log` |
| fired up | **00:03:50** |
| 主模型 | mixed软链顶屋；**FP8 QSA12层、GDN36层投影**，专家NVFP4 |
| PLE | **NVFP4 28.01G pinned**；原日志单位28.01GiB，packed原件保持 |
| MTP | 原生在线NVFP4，`SGLANG_NEXTN_MTP_NVFP4=1`；一本索引，early-share |
| 上下文 | YaRN **1,048,576**，factor **4.0** |
| cap / 实际池 | **1,111,168 / 1,111,168**，64-token页零尾页损耗 |
| mamba / 并发参数 | **20槽 / 4路**；mamba cache归档 **1.16G** |
| 图后 / 健康 | **3.57G / health200**；风险备注旧3.53G另保留 |
| mini | 归档 **3×200 OK**；验收备注 **4条mini热身**，双口径照录 |
| 最终版768K / soak | **未跑**；长测与soak双PASS在10槽/cap1M原生前身 |
| 补丁状态 | `qwen3_5_mtp.py`含NVFP4档，git M；diff已留档 |

最小后续清单按证据缺口排：

- 最终20槽/4路/cap1.11M的长输入并发与多小时soak；当前只有mini，运行峰值不能从启动余量推出。
- mamba调度侧拒新或槽位预留加护；扩槽不算实现该修复。
- 1.15M候选补正式soak；先分清预警、allocator重试与致命栈，再在对应配置补证。
- 中文FR map重建后公平A/B；现役关闭。`mtp_fp8`候选的上游2.52G是专家档口径、本机未测，不能与3.92G整段load-end直接比。
- E1b约1.13M和留余量方案未独立试；统一PLE reader、磁盘清理都未实施，不列作本轮收益。

---

<a id="section-12"></a>

## 13. 发布前检查清单

- [x] 终态采用00:08定版；头部、快照、复现参数一致。
- [x] 早期九级与晚间九幕分开；失败、回退、统计黄灯和未验项明确。
- [x] 目录锚点与实际章节核对；关键数字抽查不少于15项，核对记录见附录C。
- [x] 日志引文保留原单位；报告转述、张量审计、复算有标记；不跨实例借用PASS。
- [x] 正文无凭据值、无内部基建花絮；公开参数路径使用变量，日志只选所需原行。
- [x] README草稿原件MD5前后相同；只写新稿，没有执行服务切换、测试或补丁。
- [ ] 原草稿所列上游链接做发布时可达性复核；本次只核本地素材，未联网更新上游结论。
- [ ] 若随文分发代码补丁，由仓库所有者确认LICENSE及上游声明的保留方式；本任务未新增许可文件。
- [ ] 作者署名、致谢口径由老板定，文末留位；GitHub实际发布由仓库所有者执行。

本文写作检查完成与服务验收完成是两件事。前六项是稿件检查，不能用复选框替最终四路版补长压证据。

---

<a id="appendix-a"></a>

## 附录 A：补丁 diff 与适用范围

以下历史三补丁沿用草稿摘录，early-share来自本地工作树只读diff，FP8/NVFP4来自`astra_native_nvfp4_mtp.diff`。**标为摘录的hunk只供审阅，省略处不应直接拿去git apply**；完整NVFP4合并档留在上述报告文件旁，新增档另有`astra_native_nvfp4_mtp_incremental.diff`。本次只展示，没有应用补丁。



### A.1 poolfix / draftfix / gcfix（草稿同址摘录）

```diff
--- a/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py
+++ b/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py
@@ -340 +340,12 @@
-    if not is_draft_worker and server_args.ple_offload_embedding and device == "cuda":
+    # poolfix: SGLANG_PLUGINS（ssd_stream）路径下权重加载完同样清瞬态，
+    # 否则加载缓存残留压低 KV 池定价
+    # draftfix: draft 路径也清——draft 加载完同样在池子 sizing 前
+    _plugins_enabled = bool(os.environ.get("SGLANG_PLUGINS", "").strip())
+    if device == "cuda" and (
+        server_args.ple_offload_embedding
+        or _plugins_enabled
+        or is_draft_worker
+    ):
+        torch.cuda.synchronize()
+        import gc; gc.collect()  # gcfix: 破装载期引用环，放出循环垃圾持有的 ~7GB
```

### A.2 estfix（三分支同款，草稿摘录）

```diff
--- a/python/sglang/srt/mem_cache/kv_cache_configurator.py
+++ b/python/sglang/srt/mem_cache/kv_cache_configurator.py
@@ -2144 +2144,9 @@
-            if has_spec_dec and not replayssm_active:
+            # estfix: also skipped when gdn_mtp_cache_mode="none" --
+            # memory_pool.py sets intermediate_ssm_state_cache=None there (recovery
+            # reconstructs h_K from committed h_0), mirroring replayssm.
+            # The conv intermediate windows DO remain (~0.01GB; left un-modeled:
+            # <0.2% of the recovered budget).
+            if has_spec_dec and not (
+                replayssm_active
+                or get_exec().mamba.gdn_mtp_cache_mode == "none"
+            ):
```

> 注意：该判断在文件中同款出现 **3 处**（含休眠分支），只改活跃分支会在切换配置时复活漏账——estfix 轮后专门做了"三分支全部豁免"补齐轮，复点火池子 608,320 不动，账目自洽。

---

### A.3 fp8mtp 历史档关键摘录

`qwen3_5_mtp.py`，来源`astra_native_nvfp4_mtp.diff`。完整ignore转换与局部配置构造见原diff；这里保留KV不引入量化方法、非序列化BF16源及动态activation的关键片段。最终原生NVFP4配置关闭这个FP8开关。

```diff
+class _NextnMtpFp8Config(Fp8Config):
+    """Fp8Config variant scoped to the embedded NEXTN MTP module.
+
+    Built with ``is_checkpoint_fp8_serialized=False`` so linear/MoE weights
+    construct in source BF16 and convert to FP8 per-tensor during
+    ``process_weights_after_loading``. Attention layers deliberately get no
+    quant method: the draft's KV-cache path must stay bit-identical to the
+    unquant draft (same fp8_e4m3 KV dtype, no k/v scale plumbing).
+    """
+
+    def get_quant_method(self, layer, prefix):
+        from sglang.srt.layers.radix_attention import RadixAttention
+
+        if isinstance(layer, RadixAttention):
+            return None
+        return super().get_quant_method(layer, prefix)
+
+
+    return _NextnMtpFp8Config(
+        is_checkpoint_fp8_serialized=False,
+        ignored_layers=ignored_layers,
+        weight_block_size=None,
+        activation_scheme="dynamic",
+        packed_modules_mapping=getattr(quant_config, "packed_modules_mapping", None),
+    )
+
+
```

以上两段在完整diff中不相邻；忽略层列表与匹配转换有意省略，不把它们拼成可直接应用的补丁。

### A.4 early-share 关键摘录

`speculative/eagle_worker_v2.py`的工作树diff。共享调用移到池定价前；EAGLE3保持原路径，后续init_lm_head避免重复绑定。FP8审计辅助函数与import在本摘录外，完整工作树也包含它们。

```diff
+        self._embed_head_shared = False
+        if get_bool_env_var("SGLANG_NEXTN_EARLY_SHARE"):
+            self._early_share_target_embed_and_head()
+        if get_bool_env_var("SGLANG_NEXTN_MTP_FP8"):
+            self._audit_nextn_mtp_fp8_weights()
+
+    def _early_share_target_embed_and_head(self):
+        """NEXTN: alias draft embed/lm_head weights onto the target's, early.
+
+        Mirrors the non-EAGLE3 branch of ``init_lm_head`` but runs before pool
+        sizing. EAGLE3 is skipped (hot-token remap may clone/reindex the head);
+        its flow is unchanged. ``init_lm_head`` still runs later and becomes a
+        no-op rebind (guarded by ``_embed_head_shared``).
+        """
+        if self.speculative_algorithm.is_eagle3():
+            return
+        draft_model = self.draft_runner.model
+        if not hasattr(draft_model, "set_embed_and_head"):
+            logger.warning(
+                "SGLANG_NEXTN_EARLY_SHARE=1 but draft model %s has no "
+                "set_embed_and_head; falling back to post-pricing sharing.",
+                type(draft_model).__name__,
+            )
+            return
+        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
+        if embed is None or head is None:
+            return
+        draft_model.set_embed_and_head(embed, head)
+        self._embed_head_shared = True
+        logger.info(
+            "NEXTN early embed/head sharing done before KV pool sizing "
+            "(draft %s now aliases target tensors).",
+            type(draft_model).__name__,
+        )
+
@@ -335,8 +398,12 @@ class EagleDraftWorker(EagleDraftWorkerBase):
                 self.hot_token_id = self.hot_token_id.to(head.device)
                 head.data = head.data[self.hot_token_id]
 
-            # Share the embedding and lm_head
-            self.draft_runner.model.set_embed_and_head(embed, head)
+            # Share the embedding and lm_head. With SGLANG_NEXTN_EARLY_SHARE=1
+            # the same alias was bound before pool sizing; rebind here is a
+            # no-op worth skipping so the free stays accounted once.
+            if not (self._embed_head_shared and self.hot_token_id is None):
+                self.draft_runner.model.set_embed_and_head(embed, head)
+                self._embed_head_shared = True
             maybe_share_target_lm_head()
 
     def init_attention_backend(self):
```

### A.5 原生 NVFP4 新档（增量 diff 全文）

来源`astra_native_nvfp4_mtp_incremental.diff`；对应合并diff `astra_native_nvfp4_mtp.diff` 的 `_mtp_quant_config` 新分支。基线已含FP8档，这里新增显式NVFP4档，不另造量化kernel。

```diff
--- sglang-rtxpro6000/python/sglang/srt/models/qwen3_5_mtp.py.bak_astra_nvfp4_20260908_220834
+++ sglang-rtxpro6000/python/sglang/srt/models/qwen3_5_mtp.py
@@ -127,6 +127,35 @@
     quantized; the loader's fusion gate has to see the same normalization the
     constructor applies, or it would answer for the target's quantization.
     """
+    # Native NEXTN only: use the same BF16-source ModelOpt W4A4 adapter as
+    # the external draft. Dense/attention layers stay unquantized; no target
+    # configuration is mutated or inherited into the draft expert namespace.
+    if get_bool_env_var("SGLANG_NEXTN_MTP_NVFP4"):
+        from sglang.srt.layers.quantization.nvfp4_online import (
+            make_modelopt_fp4_online_config_from_fp8,
+        )
+
+        config = make_modelopt_fp4_online_config_from_fp8(
+            {
+                "quant_method": "unquant",
+                "ignore": [
+                    "attn", "self_attn", "embed_tokens", "lm_head",
+                    "shared_head.head", "fc_embedding", "fc_hidden",
+                    "hyper_connection_mixer", "attn_hyper_connection",
+                    "mlp_hyper_connection", "mlp.gate", "shared_expert",
+                    "shared_expert_gate", "ple",
+                ],
+                "packed_modules_mapping": getattr(
+                    quant_config, "packed_modules_mapping", None
+                ),
+            }
+        )
+        logger.info(
+            "NEXTN native MTP online quant=modelopt_fp4, quant_algo=NVFP4; "
+            "BF16 source routed MoE experts only"
+        )
+        return config
+
     # Serialized Qwen3.5 ModelOpt checkpoints keep embedded MTP weights in
     # BF16. Disable quantization for those checkpoints; non-serialized
     # modelopt_fp4 still converts MoE expert weights on load.
```

---

<a id="appendix-b"></a>

## 附录 B：日志与报告索引

日志简称位于服务工作目录；报告简称中，专项报告原件在`/tmp/`，两日日记在memory目录，README与全程复盘在本仓库。索引给的是留档出处，不表示这些本地文件已随GitHub帖子上传。已覆盖的首轮日志若曾被覆盖，正文明确只引用日记，不制造一份不存在的原行。

| 日志 / 产物 | 对应证据 |
|---|---|
| `sglang-silver-b2-retry3.log` / `sglang-silver-unquant.log` | L1 374,208；单变量池不动 |
| `sglang-silver-m12.log` / `sglang-silver-poolfix.log` | L2 519,552；L3 558,400 |
| `sglang-silver-estfix.log` / `sglang-silver-estfix2.log` | L4 608,320；三分支补齐复启 |
| `sglang-silver-nodraft.log` | 968,832，无draft诊断 |
| `sglang-silver-gcfix.log` / `sglang-silver-gcfix2.log` | L5 1,136,960；HiCache host池 |
| `sglang-silver-fp8mtp*.log` | 1,168,704→1,149,952失败；1,099,968定额 |
| `sglang-silver-1mctx*.log`（r1–r6） | 1M窗口六轮，r6池869,952 |
| `sglang-silver-mamba10.log` / `sglang-silver-mamba10-r2.log` | 早期924,992与续问；不是00:08现役 |
| `sglang-silver-frspec.log` | map与accept崩落；FR冒烟 |
| `sglang-silver.log` / `sglang-silver2.log` / `ram-monitor.log` | Route A host OOM，第2/3轮 |
| `repack_nvfp4.log` / `sglang-silver-nvfp4.log` | repack、28.01GiB钉载、槽位失败 |
| `sglang-silver-nvfp4-rollback.log` / `sglang-silver-nvfp4-1p2m*.log` | 早期回退、1.2M阶梯探针 |
| `longctx_resp.json` / `mamba10_resp.json` / `mamba10_resp2.json` | 早期针测和续问长度口径 |
| `mem-snapshot.log` | gcfix前的13.42→20.35GB探针 |
| `sglang-silver-ext-512k.log` / `sglang-silver-ext-512k-unquant.log` | 白天unquant反转；C方案时刻与池 |
| `sglang-silver-nvfp4ct-r2.log` | packed Round 2索引修复、反重与运行失败 |
| `sglang-silver-e1-earlyshare-unquant.log` / `sglang-silver-earlyshare-r2.log` | share+unquant、cap与水位；E1仍是512K启动窗口 |
| `sglang-silver-e1-gate.log` / `sglang-silver-e1-gate.attempt1-launchdied.log` | E1门禁过程；不视作严格零缓存768K成功证据 |
| `sglang-silver-astra-p1.log` / `sglang-silver-astra-p1-r2.log` | 1.05M余量门失败，1M pinned基线 |
| `sglang-silver-astra-p2-smoke.log` / `sglang-silver-astra-p2-smoke-r2.log` | auto backend失败，显式CUTLASS后mixed smoke |
| `sglang-silver-astra-final.log` | mixed 1M，图后5.56GB，真768K对应轮 |
| `sglang-silver-astra-cap1150.log` | 池1,149,952，143.384068s针测与未完成soak |
| `sglang-silver-astra-cap1000-restore.log` | 回1M、5.68GB与991成功请求恢复轮 |
| `astra_native_nvfp4_serve2.log`（`/tmp/`） | 原生局部NVFP4、3.92GB、共享顺序与5.56GB |
| `sglang-silver-native-prod.log` / `sglang-silver-native-prod.crash1_20260908_2309.log` | 正式切换与23:09 mamba assert；P1审计阶段锚点 |
| `sglang-silver-native-prod-r2.log` | 23:29，786432输入、cached0、prefill142605.21ms |
| `sglang-silver-native-20slot-cap1100.log` / `sglang-silver-native-20slot-cap1100-v2.log` | 扩槽过渡轮；不混入最终3.57G |
| `sglang-silver-native-cap1111143.log` / `sglang-silver-native-cap1111168.log` | 页对齐1111104→1111168；最后一轮00:03:50 ready |

报告与判决产物：

| 文件 | 查什么 |
|---|---|
| `sol_threemodel_final.md` | 三模型五问、60格审计；候选排名须结合Round2后验 |
| `gdn_fp8_impl.md` | FP8加载/分派、软链顶屋、target旗标与零新增内核补丁判决 |
| `astra_impl_progress.md` / `astra_impl_final.md` | P1/P2、真768K、1.15M冲顶纠错、回退soak |
| `astra_cap1000_soak_result.json` / `astra_cap1000_bench.json` | 外挂恢复轮991/991与同题性能 |
| `astra_native_nvfp4_progress.md` / `astra_native_nvfp4_final2.md` | 原生接线、12轮bench、p值、功能PASS与严格黄灯 |
| `astra_native_nvfp4_gate768k.json` / `astra_native_nvfp4_soak_result.json` | 原生786432零缓存与985/985；前身验收 |
| `astra_fixed_residency_plan.md` | 23:23张量账；1.49GiB独有权重与未细分余额 |
| `astra_native_nvfp4_mtp.diff` / `astra_native_nvfp4_mtp_incremental.diff` | FP8合并档与新增NVFP4档 |
| `2026-09-07.md` / `2026-09-08.md` | 两日蒸馏；存在旧判决者以正文冲突注释为准 |
| `复盘-0907至0909-全程review.md` | 全程时间线、失败边界与素材行号 |
| `CURRENT-PRODUCTION.md` | **00:08最终现役唯一依据**，三档回退与未验状态 |

---

<a id="appendix-c"></a>

## 附录 C：关键数字抽查记录

本次写作只读核对；日志仅用grep。下面16项已对到原日志行，报告项另核原文／结构化字段。行号以写稿时留档为准，持续追加不改变既有行号；未重新跑服务测试。

| 数字 | 原日志定位 |
|---|---|
| 374,208 / 519,552 / 558,400 | b2-retry3:109 / m12:94 / poolfix:94 |
| 608,320 / 968,832 / 1,136,960 | estfix:94 / nodraft:59 / gcfix:94 |
| 869,952 / 924,992 / 1,140,672 | 1mctx-r6:92 / mamba10:92 / ext-512k-unquant:96 |
| mixed图后5.56GB / 冲顶池1,149,952 | astra-final:219 / astra-cap1150:212 |
| 恢复图后5.68GB | astra-cap1000-restore:212 |
| 1,111,104 / 1,111,168 / 最终3.57GB | native-cap1111143:209 / native-cap1111168:209（后两项） |
| prefill142605.21ms | native-prod-r2:881 |

上表短名统一补`sglang-silver-`前缀与`.log`后缀。报告另核：A结构化块`requests=991`且`successful_requests=991`；N中985/985、p=0.6286、143.365s；P1中1.487056255GiB与2.368164063GiB；CP中20槽、4路、1.16G及mini边界。算术另核1,111,168=64×17,362、头部比值约2.97；旧172×与显示值复算约173×并列保留。

---

完稿时间：2026-09-09 00:58:51（Asia/Shanghai）；战果截止：2026-09-09 00:08。

## 鸣谢

- **jpezzulli** — Pennyroyal / sglang-rtxpro6000 树（本文全部实验的战场）、Route A 824K 对照基准、early-share / fp8mtp 补丁思路的源头
- **Garnermccloud** — SSD-Stream 拆分式封装 checkpoint（现役 ple/ 与 manifest 壳）
- **Primitive-ai** — plain / mixed / PLE-quant 三套模型仓与 PLE 量化工具链（mixed FP8 GDN/QSA 顶屋的全部用料）
- **LMSYS / SGLang 团队** — SGLang 框架本体与 Qwen3.8-Flash-Next day-0 支持文档
- **RadixArk** — day-0 NVFP4 官方量化（一体式对照组）
- **thunlp** — FR-Spec 词表压缩（实验线判了回退，机制无罪，中文 map 重建仍是待办）

没有这些上游与社区的肩膀，单卡 96G 的这场战役无从打起。文中所有错误与未竟事项，责任在本文作者。

---

**作者：Eddy**（GitHub [@AntigravityAI](https://github.com/AntigravityAI)）· 主战役完稿 2026-09-09，边界模型续测 2026-09-11 · 全部数字可由附录 B 日志索引复核
