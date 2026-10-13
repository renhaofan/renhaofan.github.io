---
title: "Elastic Token Compression for Pixel-Space Diffusion Transformers"
subtitle: "以 Hilbert 连续区间构造内容自适应 region tokens，只压缩 DiT 中层并用一个 checkpoint 覆盖连续计算预算"
authors: "Eduard Zamfir, Christian Reisswig, Zongwei Wu, Yongqin Xian, Radu Timofte"
venue: "arXiv preprint"
year: 2026
date: 2026-10-11
paper_url: "https://arxiv.org/abs/2608.29281"
code_url: "https://github.com/eduardzamfir/RTI"
project_url: "https://eduardzamfir.github.io/rti"
tags:
  - Pixel-Space Diffusion Transformer
  - Adaptive Token Compression
  - Region Tokens
  - Hilbert Curve
  - Efficient Image Generation
summary: "Elastic Token Compression 提出 Region Token Interface（RTI）：沿 Hilbert 顺序在中层特征变化最大的地方切分，把相邻平坦 patch 聚为带位置的 region token，仅在 DiT 中间 blocks 上缩短序列，再把 region update 写回每个 patch；单一 checkpoint 支持 64–1024 token 预算，并在 MiniT2I-L/16 半 token 时以 2.11 倍吞吐保持接近 dense 的 GenEval。"
permalink: /papers/elastic-token-compression-pixel-space-diffusion-transformers/
---

> **阅读依据**：arXiv:2608.29281v1（2026-08-29，19 页）与[官方项目页](https://eduardzamfir.github.io/rti)；[代码](https://github.com/eduardzamfir/RTI)和模型 checkpoint 已由项目页公开。本文重点区分 per-image 指标和 distribution-level FID：在半 token 下，GenEval 接近 dense，但 FID 仍有 +2.4 的缺口。

## 一句话总结

Elastic Token Compression 先探测 pixel-space DiT 的空间、时间与深度冗余，再据此设计 Region Token Interface（RTI）：每个采样步沿 Hilbert 顺序在中层 feature gap 最大处切成相邻、内容定大小的 regions，以 learned READ 聚合、让中间 core blocks 只处理 $R\ll N$ 个 region tokens，再由 learned WRITE 把 region change 个性化写回 $N$ 个 patch；随机预算训练使一个 checkpoint 可在部署时连续调节吞吐与质量。

## 背景与核心问题

Pixel-space diffusion 没有 VAE 预先吸收低层冗余，$512^2$ 图像配 $16\times16$ patch 会产生 $32\times32=1024$ image tokens；每个 block、每个 timestep 都等量处理它们。作者对冻结 MiniT2I-B/16 的 probe 得到三条观察（Figure 2）：

- **空间**：细节高度集中，最详细的 15% patches 承载约 50% detail，前 50% 承载 88%。
- **时间**：随着噪声变成图像，content-adaptive pooling 对中层表示的保留率越来越高。
- **深度**：冗余主要在 middle blocks；首层仍在把原始 patch 变成语义 feature，末层需恢复 per-pixel output，两端都更敏感。

这把问题具体化为四项 region reduction 要求：区域应当 **contiguous、content-sized、带明确 position、summarize 而非 delete**。Feature-similarity merging 会把远处相似 patch 混在一起，latent array 没有明确位置，token skipping 丢弃而不总结；作者希望同时保留四者。

**核心研究问题：怎样在不重训 pixel-space backbone 的前提下，把中层 patch tokens 变成有空间锚点、随当前生成内容变化的区域 tokens，并让一个模型支持任意部署预算？**

## 方法详解

### 整体框架

<div class="mermaid">
flowchart LR
    A[noisy image: N patch tokens] --> B[dense prelude blocks]
    B --> C[Hilbert order and feature gaps]
    C --> D[R contiguous regions]
    D --> E[READ: learned regional pooling]
    E --> F[short core: R region tokens]
    F --> G[WRITE: region change plus patch feature]
    G --> H[dense coda blocks]
    H --> I[clean-image prediction / next step]
    I -->|rebuild partition each step| A
</div>

MiniT2I-B/16 有 17 个 double-stream blocks；RTI 把 blocks $[3,13]$ 设为 core，前后各留 3 个 dense blocks。L/16 的 23 blocks 中压缩 $[3,19]$。只有 core 在 $R$ 个 region tokens 上运行；text tokens 始终 full length，prelude / coda 始终处理全部 $N=1024$ image tokens。

### 1. Hilbert feature-gap partition

Hilbert curve 遍历 $2^k\times2^k$ grid 的每个 cell 一次，且序列中相邻位置在二维图像中也是邻居。因此任一连续区间都对应连通区域，并且可表达所有 quadtree cells 和一些非方形区域。

令进入 core 的 patch feature 为 $h_j^{(s_0)}$，$\pi(j)$ 是第 $j$ 个 Hilbert 位置。相邻 feature gap 为

$$
d_j=\left\|h_{\pi(j+1)}^{(s_0)}-h_{\pi(j)}^{(s_0)}\right\|_2.
$$

取最大的 $R-1$ 个 $d_j$ 为切点，得到 $R$ 个连续 runs。平坦区域 feature 变化小，会形成长 run；物体边界和细节处 gap 大，会被切成小 region，甚至单 patch。该规则无参数、复杂度 $O(N)$，而且每个 denoising step 都根据当前 state 重建；partition + READ/WRITE 的开销不足 forward pass 的 2%（Appendix A）。

Figure 9 显示在高噪声早期 fine/coarse regions 的 detail ratio 接近 1，后期约升到 6：细 regions 随物体成形而聚到边界和主体上。预算 $R$ 决定可用 region 数，图像 feature 决定它们放在哪里。

### 2. READ：在区域内总结而非平均或删除

对 region $c$，READ 用单个 learned vector 给各 patch 打分，仅在该 region 内 softmax：

$$
\alpha_j=\operatorname{softmax}_{j\in c}(w^\top h_j),
\qquad
z_c=\sum_{j\in c}\alpha_jh_j+e_{size}(\lfloor\log_2|c|\rfloor).
$$

因此 region token 是 learned weighted summary，并通过 size embedding 知道自己代表多少 patches。位置不是另学一个任意 latent，而是平均区域内各 patch 的 rotary embeddings：长波长位置仍清晰，短于 region extent 的频率被低通衰减，符合“一个 token 代表一片面积”的含义。

与全局 latent queries 的 $O(NR)$ cross-attention 不同，READ 只在 region 内操作，总成本 $O(N)$；更关键的是每个 token 有固定空间 support，不会出现一个 latent 只盯主体而无人代表背景。

### 3. WRITE：共享 region work，保留 patch identity

若把 core 后变化 $z'_c-z_c$ 原样广播给区域内所有 patches，会把它们推成相同更新。RTI 将 region change 与每个 patch 的 pre-pool feature 拼接，经小网络 $g$ 生成个性化 residual：

$$
h_j^{(s_1+1)}=h_j^{(s_0)}+
g\left([h_j^{(s_0)}\Vert z'_{c(j)}-z_{c(j)}]\right).
$$

所以 core 共享区域级计算，但 coda 仍收到 $N$ 个不同 patch features。这个形式也是绕过 compressed core 的 skip connection：patch 原信息不被销毁，只附加 region-level reasoning 的修正。把 learned READ/WRITE 换成 mean-pool + broadcast 会在 $R=256$ 时损失 7.2 GenEval points（Figure 8b）。

### 4. Elastic training

backbone 冻结，只训练 READ/WRITE 与 attention/MLP projections 上的 rank-32 LoRA。每个 training step 从 $R\in\{64,128,256,512\}$ 随机取预算，40k steps，数据为 MiniT2I 约 120k alignment pool。推理不仅能用训练预算，也可用未见过的 $R\in\{96,192,384\}$；Figure 8a 中这些点落在同一 quality–budget curve 上，支持把 $R$ 当作部署时连续旋钮。

## 实验关键数据

### 设置

文本生成使用 MiniT2I-B/16 与 L/16、$512^2$、100 Euler steps、guidance 5.0；GenEval 测 compositional accuracy（553 prompts），PartiPrompts 测 CLIPScore、PickScore、ImageReward（1,632 prompts）。吞吐在 RTX 4090、batch 4（CFG 后 8）测完整 generation loop，含 partition、READ/WRITE、adapters 和 text stream。

### 主结果

Table 1 的关键 operating points：

| Backbone / 方法 | Core tokens $R$ | 加速 ↑ | GenEval ↑ | CLIP ↑ | Pick ↑ | ImageReward ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B/16 dense | 1024 | 1.00× | 87.2 | **28.23** | **22.45** | **1.18** |
| B/16 Feat Sim | 512 | 1.84× | 84.7 | 28.04 | 22.04 | 1.07 |
| **B/16 RTI** | 512 | **1.84×** | **87.8** | 28.09 | 22.30 | 1.11 |
| B/16 Feat Sim | 256 | 2.15× | 80.5 | 27.73 | 21.57 | 0.93 |
| **B/16 RTI** | 256 | **2.15×** | **84.7** | 28.04 | 22.06 | 1.06 |
| L/16 dense | 1024 | 1.00× | **88.1** | 28.50 | **22.75** | **1.23** |
| **L/16 RTI** | 512 | **2.11×** | 87.5 | **28.58** | 22.63 | 1.20 |
| **L/16 RTI** | 256 | **2.59×** | 86.3 | 28.52 | 22.37 | 1.17 |

支持的结论是：

1. 同 token budget / throughput 下，RTI 始终优于 feature-similarity merging；差距在 budget 越紧时越大，符合“丢失空间位置”的机制解释。
2. L/16 用半数 core tokens 达到 2.11× throughput，GenEval 仅低 dense 0.6，落在论文报告的 paired uncertainty 内；四分之一 tokens 时仍只低 1.8 points，2.59× 加速。
3. 不能说所有质量都无损：dense L/16 的 PickScore / ImageReward 仍高于压缩模型；B/16 在 $R=512$ 的 GenEval 略超 dense 也不代表 distribution-level quality 全面提升。

### Distribution-level FID 与跨模型验证

Appendix Figure 11a 在 MJHQ-30K 上揭示了主表没有显示的损失：L/16 的 RTI 在 $R=512$ 虽 GenEval 接近 dense，FID 仍比 dense 高 2.4；不过同 budget 下比 Feat Sim 好 8.6 FID，$R=256$ 时好 13.3 FID。作者对这一不利结果有明确披露。

在 class-conditional pixel-space JiT / ImageNet-256 上，RTI 同样在每个 budget 的 FID / Inception Score 上优于 Feat Sim。JiT-L/16 在 $R=193$（75% patch tokens、88% dense token compute）达到 FID 2.42，对比 dense published 2.36；这支持方法跨 task/backbone，但 speedup 是 analytic token compute，并非实测 wall-clock。

### 消融：每个性质的作用

- **Adaptive cuts**：把 feature-gap cuts 改成 Hilbert 上均匀 cuts，在 $R=128$ 损失 2.7 GenEval；budget 大时 gap 缩小，因为 region 本来就短。
- **Learned interface**：mean pool + broadcast 在 $R=256$ 从 84.7 降到 77.5，损失 7.2 points。
- **Spatial anchoring**：换成 free latent array 后只有 45.7；但此消融同时移除了 grouping 与 position，所以 39-point gap 是 anchoring 单独作用的上界，不能完全归因于位置。
- **Ordering**：小 budget 下 Hilbert 优于 Morton，再优于 raster；三者在大 budget 接近。这说明 locality 在长 region 时才真正关键。

Table 4 的机制测量很有说服力：RTI group 中 patch 到中心的平均 spread 为 0.49 grid cell，Feat Sim 为 5.16，latent array 为 11.35；Feat Sim 对 core representation 的保留率 0.74 甚至略高于 RTI 的 0.73，却生成更差，说明问题不是“信息量不足”，而是 token 没有可信空间位置。

### 与减少 sampling steps 的关系

两者都省 wall-clock，但不是可直接叠加的免费收益。Figure 7 / 13 在相同成本下比较：1.8 秒时，RTI 以较多 steps 得到 GenEval 82.7，dense 少 steps 只有 65.9；约 5 秒后趋势反转，额外 sampling steps 比继续压 token 更划算。论文 Table 1 的 100-step point 强调质量保持，并非全预算范围的 compute-optimal setting。

## 亮点与我的洞察

### 论文亮点

- 先用 probe 定位冗余的 space/time/depth 结构，再由四条性质推导 reduction design；方法不是凭直觉拼接。
- Hilbert curve 把二维连通区域切分转成一维 top-$R$ boundary selection，简单、无参数、逐步重建仍很便宜。
- READ/WRITE 同时解决 summary、position 与 per-patch reconstruction，且只 retrofit 冻结 checkpoint 的中层。
- 报告未见训练预算的插值、FID 不利结果、ordering、sampling-step trade-off 和跨 JiT backbone，证据范围较完整。

### 我的洞察

- **个人分析：RTI 的核心不是“相似 token 合并”，而是“空间 support 成为 token 身份”。** Feat Sim 保留的 representation 甚至略多，却因一个 group 横跨画面而失败；生成模型需要知道 feature 代表哪里，不只是它包含多少信息。
- **个人分析：hourglass depth profile 给出了通用 retrofit 原则。** 输入端先让 dense patches 建立语义，输出端恢复局部自由度，只压中间 reasoning layers。这可迁移到 video DiT、多视图 transformer 或 dense prediction backbone，但具体 core 边界应先 probe。
- **个人分析：预算 elasticity 与 placement adaptivity 是分离的。** 用户决定 $R$，模型只决定这 $R$ 个 regions 放在哪里。下一步应让模型依据当前 detail / uncertainty 自动选择 $R_t$，同时约束总 latency。

## 局限与展望

### 作者承认的局限

- dense prelude、coda 与 text stream 不压缩，speedup 最终饱和；L/16 从 $R=128$ 减到 64 仅再增加约 0.15×。
- compressed core 无法区分 region 内 patches，只能在 WRITE 时结合旧 patch feature 个性化恢复。
- 并非所有二维 grouping 都能表示为单个 Hilbert interval。
- budget 由用户指定，不由图像内容自动决定。

### 独立分析

- **对更现代 latent/video backbone 的外推未验证。** Pixel space 冗余最大，latent DiT 已由 VAE 压缩；RTI 在其中可能收益更小。LoT Diffusion 正好说明 latent/video 场景可能需要显式 multiresolution projection，而非直接照搬 pooling。
- **质量保持依赖任务指标。** GenEval 接近 dense 时 FID 仍 +2.4；部署不能只看 compositional score，应同时验证分布、多样性、身份稳定性和文字细节。
- **逐 step hard partition 可能引入轨迹不连续。** 当前指标未直接测相邻 steps 的 region assignment churn；虽然输出视觉上稳定，仍值得记录 token boundary turnover，并比较 temporal smoothing / hysteresis。
- **训练仍需每个 backbone 单独微调**。所谓 elastic 指一个 checkpoint 覆盖多个 $R$，不是把接口零训练地插入任意模型；40k steps 与同域 alignment pool 是真实成本。

最值得追加的实验是学习一个有全局 compute constraint 的 $R_t$ controller：根据 timestep、feature-gap distribution 和 prompt difficulty 自动分配每步 budget，再与固定 $R$、减少 steps 和二者联合策略比较等延迟 GenEval、FID、文字准确率与能耗。

## 与 Level-of-Token Diffusion 的对比

| 维度 | RTI | Level-of-Token Diffusion |
| --- | --- | --- |
| 目标 | 无外部 layout 的自适应 acceleration | 外部 detail layout 驱动的控制 + acceleration |
| Region 决策 | 每步从当前中层 features 动态切分 | 生成前由 mask/box/VRS/depth/agent 指定 |
| Region 形状 | Hilbert 连续区间，连通但可不规则 | 支持集合内的轴对齐矩形 |
| Dense 恢复 | learned WRITE 返回 per-patch residual | asymmetric flow 解析恢复 dense velocity |
| 压缩深度 | 只压 core，首尾保持 dense | 多分辨率 tokens 贯穿 DiT 主干 |
| Elasticity | 同 checkpoint 支持连续 region count $R$ | 同模型支持不同 layout / token count，但 extent 需预先支持 |
| 实验模型 | pixel-space MiniT2I / JiT | latent-space FLUX.2 / Wan2.1 |

二者可以互补：RTI 提供在线、内容自适应的 partition，LoT 提供从 coarse regions 到 full-resolution flow 的更强生成参数化。一个自然假设是先用 RTI 风格 feature gaps 在线预测 LoT layout，再用 asymmetric recovery 在 latent/video model 中执行；但需解决 hard layout 在时间上的稳定性。

## 评分

| 维度 | /10 | 理由 |
| --- | ---: | --- |
| 创新性 | 8.8 | Hilbert 本身不新，但把连续区间作为冻结 pixel DiT 的动态 region-token unit 很有洞察。 |
| 技术可靠性 | 8.7 | probe、四性质、机制指标和消融互相吻合；FID 显示仍非真正无损。 |
| 实验充分度 | 8.9 | 两种 MiniT2I、JiT、未见预算、FID、ordering、steps trade-off 与失败案例覆盖完整。 |
| 写作清晰度 | 9.0 | 论证从冗余诊断到接口设计高度连贯，且主动限定 claim。 |
| 实用 / 研究价值 | 8.5 | 一个 checkpoint 多预算与开源实现很实用，但只验证 pixel-space 且 speedup 有结构性上限。 |

**总体推荐：值得细读。** 它最可复用的价值是证明 token compression 的关键不是“合并得像不像”，而是压缩后的单位是否仍代表一个连通、有位置、被总结而非删除的空间区域。

## 阅读结论

- **最值得记住的点**：Hilbert 连续区间把二维内容自适应区域化简为一维切分，同时保持每个 region 的空间锚点。
- **最需要怀疑的点**：GenEval 接近 dense 不等于分布质量无损；半 token 的 FID 仍差 2.4，且现代 latent/video backbone 尚未验证。
- **最值得复现或继续验证的点**：自动选择 per-step budget $R_t$，以及在 latent/video DiT 上比较 learned WRITE 与 asymmetric dense-flow recovery。

## 相关论文

- [Level-of-Token Diffusion](https://arxiv.org/abs/2610.05816) — 同期 multiresolution token 工作；接受显式 layout，并以 asymmetric flow 恢复 dense state。
- [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461) — feature-similarity merging baseline；RTI 证明空间位置比单纯表示保留率更关键。
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) — token skipping 对照；省略计算但不总结被跳过 token。
