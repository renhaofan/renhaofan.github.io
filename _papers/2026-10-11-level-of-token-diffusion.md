---
title: "Level-of-Token Diffusion"
subtitle: "用显式多分辨率 token 布局把空间细节需求转化为 DiT 计算分配，并以非对称流恢复全分辨率生成轨迹"
authors: "Kiyohiro Nakayama, Brian Chao, Jan Ackermann, Hansheng Chen, Federico Tombari, Leonidas Guibas, Lior Yariv, Gordon Wetzstein"
venue: "arXiv preprint"
year: 2026
date: 2026-10-11
paper_url: "https://arxiv.org/abs/2610.05816"
project_url: "https://georgenakayama.github.io/lotdiffusion/"
tags:
  - Diffusion Transformer
  - Adaptive Tokenization
  - Flow Matching
  - Image Generation
  - Video Generation
summary: "Level-of-Token Diffusion 用外部语义、纹理、景深或用户规划生成的多分辨率矩形 token 布局，只让 DiT 处理缩短后的序列，再以逐 patch 非对称流恢复 dense velocity；在 FLUX.2-4B 图像生成中达到 1.32–2.04 倍、Wan2.1-14B 视频生成中达到 1.58–3.53 倍加速，并使细节位置成为可控的计算预算。"
permalink: /papers/level-of-token-diffusion/
---

> **阅读依据**：arXiv:2610.05816v1（2026-10-05，39 页，含附录）与[官方项目页](https://georgenakayama.github.io/lotdiffusion/)。论文未在项目页提供代码链接，本文不把网页 demo 视为可复现实装。论文的定量 layout 来自 held-out reference image/video 派生的 VRS 信号；这一点是理解结果边界的关键。

## 一句话总结

Level-of-Token（LoT）Diffusion 把“哪里需要细节”显式编码为覆盖全画面的多分辨率矩形 token layout：粗 token 覆盖平坦/失焦区域，细 token 留给主体与高频纹理；模型以 patch-wise asymmetric flow、extent-specific heads、size embedding 和统一坐标 RoPE 改造预训练 FLUX.2 / Wan2.1，在只处理较短序列的同时每步仍恢复全分辨率 flow，从而同时实现局部细节控制和按布局决定的推理加速。

## 背景与核心问题

标准 Diffusion Transformer（DiT）把 latent grid 切成等大 patch，所有区域在所有去噪步使用相同 token 密度。但自然图像的细节通常高度不均：文字、脸和物体边界需要细粒度，天空、墙面、路面或景深外背景可以更粗。已有 token merging / pruning 可以省算力，却通常不让应用显式指定算力应落在哪里；已有 mixed-resolution 方法又常需从头训练或只支持有限尺度。

本文的切入点来自 level of detail（LoD）与 variable-rate shading（VRS）：如果生成前已知 bounding box、semantic mask、texture variance、depth-of-field，甚至 agent plan，就能先决定空间计算密度，再让生成器服从它。

**核心问题：怎样让一个预训练、固定 patch size 的 DiT 接受任意多尺度 token tiling，同时仍在每个去噪步产生 VAE 所需的 full-resolution latent update？**

难点对应三项设计：

1. 一个粗 token 代表多个原始 patch，却仍要预测这些 patch 的 dense velocity；
2. 不同形状与大小的 token 必须共享 transformer，同时让模型知道其 support extent；
3. layout 必须能由不同先验生成并覆盖图像/视频，而不是只支持固定二级网格。

## 方法详解

### 整体框架

<div class="mermaid">
flowchart LR
    A[full-resolution noisy latent x_t] --> B[LoT layout partition]
    S[mask / box / texture / depth / plan] --> B
    B --> C[extent projection A_e^T]
    C --> D[short multiresolution token sequence]
    D --> E[DiT with size embedding and center-aligned RoPE]
    E --> F[extent-dependent asymmetric velocity heads]
    F --> G[analytic full-rank velocity recovery]
    G --> H[full-resolution latent ODE step]
    H -->|next denoising step| A
</div>

预训练模型的 uniform grid 有 $H\times W$ 个 token。LoT layout 是一组互不重叠且覆盖整个 grid 的轴对齐矩形区域

$$
\mathcal P=\{R_1,\ldots,R_L\}, \qquad
R_i=[u_i,u_i+e_{h,i})\times[v_i,v_i+e_{w,i}),
$$

每个 $R_i$ 只变成一个 token，extent 为 $e_i=(e_{h,i},e_{w,i})$，所以序列长度由 $HW$ 降为 $L$，compression 为 $HW/L$。视频增加时间轴；本文视频 token 不压时间，只采用 $(1,e_h,e_w)$ 的空间 extent。

### 1. Patch-wise asymmetric flow：短序列输入，dense flow 输出

若把区域内 $e_he_w$ 个 dense tokens 直接压成一个 $D$ 维 token，普通 DiT 只能输出 $L$ 个 velocity，无法更新 full-resolution latent。作者为每种 extent $e$ 拟合半正交基 $A_e\in\mathbb R^{D_e\times D}$，其中 $D_e=e_he_wD$。将区域的 dense noisy patch $x_t^{(i)}$ 投影为

$$
\bar x_{t,R_i}=A_{e_i}^{\top}x_t^{(i)}\in\mathbb R^D.
$$

因此 transformer 的 hidden width 保持与预训练模型相同，只有 sequence length 变短。模型预测投影空间的 asymmetric velocity；令 $P_e=A_eA_e^\top$，dense target 为

$$
u_A^{(i)}=P_{e_i}\epsilon^{(i)}-x_0^{(i)}.
$$

预测 $\hat u_A^{(i)}$ 后，解析恢复 full-rank velocity：

$$
\hat u^{(i)}=P_{e_i}\hat u_A^{(i)}+
(I-P_{e_i})\frac{x_t^{(i)}+\hat u_A^{(i)}}{\sigma_t}.
$$

直觉上，低维投影负责由 coarse token 显式预测的部分，正交补空间利用当前 noisy state 与 flow path 的关系补回；当 asymmetric target 完全预测正确且 $t>0$ 时，式子恢复原 dense flow $\epsilon-x_0$。Table 3/5 表明若跳过解析恢复、让粗 token head 直接预测 high-resolution flow，FID 会从 13.800 恶化到 26.487（约 1.52× compression），是最关键的结构之一。

每种 extent 使用独立 output head $h_e$，并以 $W_e=A_eW_{out}$ 从预训练 head 初始化。输入 head 也由预训练 projection 与 $A_e^\top$ 组合初始化，这让不同 extent 尽量继承预训练表示，而非从零学 coarse patch 的含义。

### 2. Token 大小与位置编码

普通 RoPE 只描述 token 在哪里，不描述它覆盖多大范围。LoT 为 extent 构造

$$
s(e_i)=\bigl(\log_2 e_{h,i},\log_2 e_{w,i},
\log_2(e_{h,i}e_{w,i}),\log_2(e_{h,i}/e_{w,i})\bigr),
$$

经零初始化 MLP 加到输入 embedding。它同时表示高度、宽度、面积和 aspect ratio；移除 size embedding 会使主消融的 HPSv3 从 10.3477 降到 10.0284、FID 从 13.800 升至 14.584（Table 3），作用存在但小于 asymmetric recovery。

位置继续沿用预训练 axial RoPE，但索引设为矩形在 finest uniform grid 上的中心：

$$
c_i=\left(u_i+\frac{e_{h,i}-1}{2},
v_i+\frac{e_{w,i}-1}{2}\right).
$$

unit patch 时退化回原位置编码；shape embedding 补充 extent。这样 attention 仍是标准 dense attention，不需 query-dependent position map。

### 3. 从多种空间先验构造 layout

LoT-Flux 支持每轴 $\{1,2,4,8\}$，LoT-Wan 支持每轴 $\{1,2,4\}$。layout 来源包括：

- **Semantic mask / bounding box**：主体赋更高级别，背景赋更粗 level；重叠处取更细者。
- **Texture variance / VRS**：方向性亮度变化低于容差时沿该方向合并，允许 anisotropic rectangle。
- **Depth of field**：由 inverse-depth 与焦平面距离计算 circle-of-confusion；在焦区域用细 token，失焦区用粗 token。
- **Custom / agent detail map**：用户绘制或 agent 生成 detail field，再以 quadtree 从 $8\times8$ 递归拆分到 $1\times1$。

这一 layout 不只是 acceleration mask。Appendix Table 4 把相同 prompt、seed、budget 下的 fine-token box 移位：LoT-Flux2-9B 的 subject target-region coverage 为 66.83%，dense FLUX.2 为 44.94%，说明 token density 会反过来影响主体落点。换言之，计算布局也是一种弱空间控制信号。

### 训练与推理

- 图像模型从 FLUX.2-klein-base-4B 初始化，在 Aesthetic-Train-V2 的 $1024^2$ crop 上训练；text encoder / VAE 冻结。extent heads、size MLP、final adaptive norm 全量训练，transformer 用 rank-256 LoRA；50k steps、batch 32。主定量公平比较则让 LoT、DDiT、Foveated Diffusion 各在 VRS layouts 上训练 20k steps。
- 视频模型从 Wan2.1-T2V-14B 初始化，在 Vchitect-T2V-Dataverse 的 100k 个 81-frame / 720p clips 上微调。
- 图像用 30 sampling steps、CFG 4；视频用 50 steps。每一步都根据固定 supplied layout 压缩 DiT 输入，再恢复 dense velocity；layout construction 和 model loading 不计入表中 runtime。

## 实验关键数据

### 图像生成：FLUX.2-4B

Table 1 在 5,261 个 held-out prompts 上比较 matched token budgets。下表保留最能说明结论的 dense、ToMe-SD 与 LoT 行：

| 方法 | Token compression | 加速 ↑ | ImageReward ↑ | HPSv3 ↑ | FID ↓ | MUSIQ ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FLUX.2-4B | 1.000× | 1.001× | **0.9518** | **10.4299** | 16.748 | 69.783 |
| ToMe-SD | 1.893× | 1.538× | 0.1757 | 5.3569 | 25.757 | 58.991 |
| **LoT** | 1.893× | **1.560×** | 0.8833 | 10.1211 | **13.361** | **69.691** |
| ToMe-SD | 2.935× | 2.003× | -0.6029 | 0.9253 | 43.528 | 49.119 |
| **LoT** | 2.935× | **2.040×** | 0.8326 | 9.5522 | **13.345** | **68.760** |

LoT 在 1.32–2.04× speedup 范围内，大多数指标优于同 budget 的 ToMe-SD、DDiT 和 Foveated Diffusion。尤其 compression 从 1.893× 增至 2.935× 时，LoT FID 几乎不变（13.361→13.345），而 prompt-preference 指标逐步下降；这说明 reference-derived VRS layout 对 held-out reference distribution 有很强帮助，不能简单理解为无条件生成在三倍压缩下完全不损质量。

另一个需谨慎的结果是 LoT FID 优于 dense control。定量 layout 由每个 held-out reference image 的 luminance variance 派生；虽然 reference RGB 不送入 generator，但其空间频率分布构成额外 test-time side information，因此该 FID 不等价于 text-only dense model 的纯压缩比较。

### 视频生成：Wan2.1-14B

Table 2 在 200 held-out prompts 上报告 VBench：

| 方法 | Compression | 加速 ↑ | Aesthetic ↑ | Imaging ↑ | Subject ↑ | Motion ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Wan2.1-14B | 1× | 1.000× | **0.5518** | **0.6465** | **0.9254** | **0.9899** |
| Foveated Diffusion | 3× | 3.276× | 0.5080 | 0.4952 | **0.8761** | **0.9839** |
| **LoT-Wan2.1** | 3× | **3.530×** | **0.5104** | **0.5650** | 0.8687 | 0.9829 |

LoT 在各 budget 都是 compressed methods 中最高 Aesthetic / Imaging，并实现 1.58–3.53× 加速；但 dense model 仍在这些质量指标上领先，且 Foveated Diffusion 在部分 Background / Subject / Motion 指标略高。论文“best”应限定为同压缩 budget 的部分维度，而非全面超过 dense 或所有基线维度。

### 消融与失败案例

Table 3 / Appendix Table 5 支持以下层次：

1. **Asymmetric flow recovery 最关键**：约 1.52× compression 下 direct-HR prediction 的 FID 26.487，对比完整模型 13.800；压缩越激进，差距越大（2.935× 时 68.508 vs. 13.345）。
2. **多级、可各向异性的 extent 有实际价值**：只允许 $1\times1/2\times2$ 的 binary grid，FID 为 20.603，明显差于 13.800。
3. **size embedding 是小而稳定的增益**：移除后多数指标下降，但影响远小于前两者；extent-specific heads 与中心 RoPE 已隐含部分尺度信息。

作者承认的失败案例位于 Appendix A.14：当整幅图都充满细线/纹理（乐谱、近景草地、织物），几乎处处需要细 token，节省空间很小；强制从 3,189 tokens / 1.28× compression 降到 1,466 / 2.79× 时，乐谱 staff lines 更破碎扭曲（Figure 22）。

## 亮点与我的洞察

### 论文亮点

- 将语义布局从“控制画什么、画在哪”推进为“决定哪里花多少 transformer compute”。
- coarse token 不只做平均池化，而是通过 extent basis 与 asymmetric flow 在每一步恢复 dense update，真正保持 full-resolution state trajectory。
- 同一框架覆盖图像和视频，并支持 box、mask、VRS、depth 和人工/agent plan；layout control 实验不仅展示图，还测了 target-region coverage。
- 与多种 token reduction baseline 在 matched budgets 下对比，并将输入/输出 head、size encoding、layout levels 分别消融。

### 我的洞察

- **个人分析：LoT 更像“生成式 variable-rate shading”，不是普通 token pruning。** 它不判断哪些 token 可删除，而是让每个 token 对应一个有几何 extent 的区域，并始终维持 full-resolution latent state。这使它既能省算力，也能把计算密度变成控制通道。
- **个人分析：layout predictor 是系统真正的上限。** 表中 quantitative VRS 使用 reference-derived signal，实际 text-only 部署没有这张未来图。mask、box 或 agent plan 可在生成前获得，但 texture variance 通常要由粗生成、外部预测器或前一帧估计；其误差和构图偏置需要端到端评测。
- **个人分析：细 token 不是被动保真，而会吸引内容。** target-region coverage 说明模型学到“高预算区应放主体”。这对交互控制有用，也意味着错误 layout 可能改变语义构图，而不仅是降低局部清晰度。

## 局限与展望

### 作者承认的局限

- 全局细节密集的场景几乎没有可压缩区域；激进预算会破坏细结构。
- 收益取决于场景 detail requirements，layout adaptation 不保证低预算下的 fidelity。

### 独立分析

- **定量效率排除了 layout construction**。若 mask/depth/VRS 来自额外大模型或先生成 reference，其成本可能改变端到端 speedup。应分别报告 oracle layout、轻量预测 layout、agent layout 的总延迟与质量。
- **主表不是纯粹的同信息比较**。LoT 使用 reference-derived VRS，dense baseline 只有 prompt；更公平的“压缩能力”实验应以同一可部署 predictor 为两者提供相同 side information，或用生成中在线估计的 layout。
- **extent 集合有限且需专用 heads**。所谓 arbitrary layout 是在已支持的 $1/2/4/8$ 尺度矩形 tiling 内任意组合，不是连续任意 shape；更多尺度会增加 heads、basis 拟合与训练覆盖压力。
- **空间布局在时间上如何稳定仍不充分**。视频逐帧 layout 可能因 mask/depth 噪声抖动；VBench 没有直接衡量 token boundary movement 或局部 texture flicker。建议加入 layout perturbation 与 temporal consistency stress test。

最值得追加的实验是两阶段在线系统：低预算 coarse pass 预测 uncertainty/detail map，第二阶段用 LoT refinement；与 oracle VRS、外部 mask/depth、均匀 token budget 比较端到端 latency、FID/HPS、layout cost 和局部失败率。

## 与 Elastic Token Compression / RTI 的对比

| 维度 | LoT Diffusion | Elastic Token Compression（RTI） |
| --- | --- | --- |
| 预算来源 | 外部 mask/box/VRS/depth/用户或 agent plan | 当前去噪状态的中层 features，逐步动态重切分 |
| Token 几何 | 预定义尺度的轴对齐矩形，可各向异性 | Hilbert 序列的连续区间，形状随内容且未必矩形 |
| 压缩位置 | DiT 主干的多分辨率输入 | 只压 middle core；prelude/coda 保持 dense |
| Dense 输出 | asymmetric flow 解析恢复每个 patch velocity | WRITE 把 region update 散回原 patch tokens |
| 主要模型 | latent-space FLUX.2、Wan2.1 | pixel-space MiniT2I / JiT |
| 控制性 | layout 显式指定哪里细、且会影响内容位置 | 主要目标是自适应加速，不提供外部空间控制 |
| 核心局限 | 依赖 layout quality；全局细节场景难压缩 | speedup 受 dense 两端/文本流限制；budget 仍手设 |

两者共同结论是：**视觉 token 的正确单位可以是“有空间 support 的区域”，而不必永远是固定 patch。** 差异在于 LoT 让任务/用户预先规定计算分布，RTI 让模型在生成轨迹中自己发现冗余；前者更可控，后者更接近无外部先验的自动压缩。

## 评分

| 维度 | /10 | 理由 |
| --- | ---: | --- |
| 创新性 | 9.0 | 把多分辨率 layout、预训练 DiT 兼容与 asymmetric dense-flow recovery 组合为明确的新接口。 |
| 技术可靠性 | 8.4 | 解析恢复与初始化设计严谨、消融强；layout 生成的端到端部署链尚未闭环。 |
| 实验充分度 | 8.3 | 图像/视频、多基线、多指标和 controllability 齐全，但主定量依赖 reference-derived VRS。 |
| 写作清晰度 | 8.8 | 公式、extent 定义、训练与限制交代完整；主表的 side-information 边界需读附录才清楚。 |
| 实用 / 研究价值 | 8.7 | 对可控生成、机器人规划和实时视频有直接价值，前提是能低成本获得可靠 layout。 |

**总体推荐：值得细读。** 论文最重要的贡献不是某个速度数字，而是把空间细节先验正式变成 token geometry 与计算预算，并给出兼容预训练 flow model 的 dense-state 更新方式。

## 阅读结论

- **最值得记住的点**：coarse token 仍可通过 patch-wise asymmetric flow 驱动 full-resolution latent trajectory，序列短不必等于状态分辨率低。
- **最需要怀疑的点**：reference-derived VRS 给主定量带来了额外测试时空间信息，实际可部署 layout 的质量–成本尚未同等验证。
- **最值得复现或继续验证的点**：在线预测 layout 的端到端系统，以及 layout 误差/时间抖动对生成语义与局部细节的影响。

## 相关论文

- [Elastic Token Compression for Pixel-Space Diffusion Transformers](https://arxiv.org/abs/2608.29281) — 同期的区域 token 压缩工作；动态发现内容区域，而非接受外部 LoT layout。
- [Asymmetric Flow Matching](https://arxiv.org/search/?query=Asymmetric+Flow+Models&searchtype=title) — LoT 从 compressed token 恢复 full-rank flow 的参数化来源。
- [Token Merging for Fast Stable Diffusion](https://arxiv.org/abs/2303.17604) — 主要 token-merging baseline；按 feature similarity 合并，不显式表示空间 extent。
