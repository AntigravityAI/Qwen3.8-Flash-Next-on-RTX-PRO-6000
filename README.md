# Taming Qwen3.8-Flash-Next on a single RTX 6000 Pro 96GB: KV pool 374K → 1,111,168 (1M context, 4-way concurrency, 62GB system RAM)

> *Qwen3.8-Flash-Next (180B MoE) on one RTX 6000 Pro (96GB) + 64GB system RAM — 1.11M-token KV pool, 1M context, native NVFP4 MTP.*
>
> **单卡服役实录**：96G 显存 + 62G 内存，SSD-Stream 双通 PLE，768K 长文实测过闸，全数字带日志出处。
>
> **[完整中文账本 → README.zh-CN.md](README.zh-CN.md)** (full Chinese write-up — every number traceable to verbatim logs)

A two-day engineering campaign (Sep 7 → Sep 9, 2026) growing the KV pool **374,208 → 1,111,168 tokens (×2.97)** on a single RTX 6000 Pro 96GB with only ~62GB usable host RAM, plus a Sep 10–11 boundary-model round establishing **why that cap is the stop line** (see below). All numbers come from archived logs; nothing is extrapolated.

## TL;DR

| | |
|---|---|
| GPU | RTX 6000 Pro 96GB (Blackwell SM120), TP=1, driver `nvidia-driver-610-open` (open kernel module is mandatory on Blackwell) |
| Host RAM | 62GB — this constraint shaped the whole architecture |
| Model | Qwen3.8-Flash-Next, NVFP4 **SSD-Stream** split checkpoint: 78G weights on GPU, 51.2G PLE table streamed from NVMe (2×320 MiB staging buffers) |
| KV cache | `fp8_e4m3`, mamba SSM state `bfloat16`, page size 64 |
| Speculative decoding | native NEXTN MTP (`steps=3, topk=1, draft_tokens=4`) |
| Stack | SGLang (jpezzulli/sglang-rtxpro6000 `pennyroyal-v2.3.0`) |
| Result | KV pool **1,111,168 tokens** (page-aligned, 64 × 17,362), context **1,048,576** (YaRN ×4), 768K long-context run passed |

## The campaign: nine ignition levels

| # | Change | KV pool `max_total_num_tokens` |
|---|---|---:|
| L1 | SSD-Stream baseline (mem-fraction 0.981) | 374,208 |
| L2 | mem-fraction 0.992, mamba 24→12 | 519,552 |
| L3 | **poolfix**: drop `expandable_segments` + `empty_cache` | 558,400 |
| L4 | **estfix**: estimator under-accounting fix | 608,320 |
| D | diagnostic: strip all speculation | 968,832 (proves the MTP tax: −360K) |
| L5 | **gcfix**: one `gc.collect()` before pool sizing | **1,136,960** ← biggest single jump, +528,640 |
| L6 | native NEXTN + FP8 MTP | 1,099,968 |
| L7 | 1M context (YaRN ×4), cap 870,000 | 869,952 |
| L8 | mamba 16→10 retune, cap 925,000 | 924,992 |
| Final | native NVFP4 MTP + early sharing, cap 1,111,168 | **1,111,168** |

One line of `gc.collect()` — breaking reference cycles left over from weight loading — recovered ~7GB and bought +528K tokens. Every gigabyte squeezed out was budget, not windfall: half went to the pool, half to runtime headroom.

## Key findings

1. **The pool gap is a "memory-at-pricing-time" problem.** The 450K-token gap vs upstream's 824,384 was transient allocations polluting the pool-sizing moment — not bigger reservations. Single-variable controls killed the easy explanations first.
2. **KV pricing is a constant 13,312 B/token.** The entire capacity problem reduces to `available bytes ÷ 13,312`. Fully algebraic.
3. **Pinned PLE is physically impossible on 62GB host RAM.** Official full config wants 47.68 GiB pinned PLE + 32G HiCache ≈ 79.7G — `cudaHostAlloc` doesn't swap, three boot failures proved it dead. SSD-Stream streaming cut host-side need from 47.7G to under 1G. (NVFP4 quantization later shrank the table to 28.01 GiB, reviving a pinned branch — dual-path, see §6 of the Chinese write-up.)
4. **`expandable_segments` silently steals ~7% of the pool.** Delete it at extreme mem-fraction.
5. **Nominal 1M ≠ usable 1M.** QSA extend gathers over full sequence depth (O(D) memory); the real single-request ceiling was ~850K at that config. Strict zero-cache 786,432-input is what was actually verified.
6. **MTP costs 5 mamba slots per request** (4 draft + 1 main). Concurrency = `floor(max_mamba_cache_size / 5)`.
7. **FR-Spec vocab compression: mechanism fine, corpus decides.** Official 64K map has 13 CJK tokens (0.02%) — Chinese accept rate collapsed 0.57→0.06. English coding speed was unchanged (177.2 tok/s). Not a verdict on the mechanism; a Chinese map rebuild is pending.
8. **Cache continuation works: 99.99% prefix hit, second-round prefill 172× faster** (radix reuse on a 768K needle-in-haystack follow-up).

## Top pitfalls (full table in Chinese version)

- Blackwell requires **open** kernel modules — closed 610 variant dies at `RmInitAdapter failed`
- Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at mem-fraction 0.992
- chunked-prefill 8192 + mem-fraction 0.992 = OOM: activation peak (~2.5GB) eats the graph/warmup living space — stick to 4096
- TP3 doesn't exist: GDN `linear_num_key_heads=16` must be divisible by TP (1/2/4/8/16)
- The pool cap must be page-aligned — the final 64 × 17,362 = 1,111,168 costs zero bytes
- Warm up briefly before long inputs: lazy kernels/workspace aren't paid for at boot ("cold furnace" big-prefill crash)

## The boundary model (Sep 10–11): where the pool should stop

The nine-level campaign grew the pool; the follow-up round established **where it must stop**, on the dockerized production stack — every sweep point predicted from the boundary formula first, verified after restart.

- **Formula**: `pool = min(cap, (95.59 GiB × mem-fraction − fixed overhead) ÷ ~12 KB/token)`; fixed overhead = 75.60 weights + 1.46 CUDA graphs + ~1.9 GiB out-of-pool residents.
- **With an explicit `--max-total-tokens`, mem-fraction degrades from allocation knob to safety guardrail**: 0.99/0.98/0.97/0.96 measured identical (pool 1,111,168, startup headroom 3.57 GiB). It only bites below ~0.943, where the pool shrinks proportionally.
- **Raising the cap costs 1.30 GiB headroom per +100 K tokens** (measured on two points, startup and runtime agree). At cap 1,170,006 the *first* big image passes but leaves 131 MiB and the *second* one (different aspect ratio) 500s — **single-image pass ≠ usable**; the acceptance test is post-image headroom ≥ ~0.8 GB. Current cap 1,111,168 sits exactly on that line and survived 7 images back-to-back (incl. 2796×2002).
- **0.992 crash chain**: vision pre-transient failed → scheduler self-shutdown → `exit 0` → `restart: unless-stopped` loop. A clean exit-0 restart in docker events is application suicide, not a manual restart and not a container OOM-kill.
- **HiCache ceiling is host RAM, not VRAM**: 62 GB = 28 pinned PLE + L2 + ~21 OS/workers; `--hicache-size 16` dies with `Not enough host memory` — 13 GiB is the cap. L3 (file backend, 150 GiB SSD) requires `write_through` or hit-rate reads back ~0 after restart.
- **The real knob for big images is `SGLANG_IMAGE_MAX_PIXELS`**, not fraction or cap (clamps pre-encode; ~602 K pixels → few-hundred-MB transient).
- Removing the cap is a regression, not a freeing: auto-derivation fills the whole fraction budget → ~0.5 GB slack → warmup OOM. Keep the cap pinned.

## Reproduce

See the final production snapshot in [README.zh-CN.md §11](README.zh-CN.md) — launch flags, patch diffs (poolfix / estfix / gcfix / fp8mtp, Appendix A), and the log index (Appendix B) for every number above.

## Acknowledgements

- **jpezzulli** — the Pennyroyal / sglang-rtxpro6000 tree (the battlefield for every experiment here), the 824K Route-A baseline, and the origin of the early-share / fp8mtp patch ideas
- **Garnermccloud** — the SSD-Stream split checkpoint shell (the active `ple/` + manifest packaging)
- **Primitive-ai** — the plain / mixed / PLE-quant model repos and PLE quantization toolchain
- **LMSYS / SGLang team** — SGLang and day-0 Flash-Next support docs
- **RadixArk** — day-0 official NVFP4 quant (the one-piece control group)
- **thunlp** — FR-Spec vocab compression (experiment line rolled back; mechanism unindicted)

Without these upstream and community shoulders, none of this happens on a single 96G card. All errors and unfinished items are the author's.

---

**Author: Eddy** (GitHub [@AntigravityAI](https://github.com/AntigravityAI)) · main campaign 2026-09-09, boundary model 2026-09-11 · every number verifiable via Appendix B log index · **[中文完整版](README.zh-CN.md)**
