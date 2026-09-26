---
title: "DéjàView: Looping Transformers for Multi-View 3D Reconstruction"
subtitle: "以连续时间条件的共享 Transformer block 循环细化多视角 tokens，将模型深度改为推理时可调的计算预算"
authors: "Alessandro Burzio, Tobias Fischer, Sven Elflein, Qunjie Zhou, Riccardo de Lutio, Jiawei Ren, Jiahui Huang, Shengyu Huang, Marc Pollefeys, Laura Leal-Taixé, Zan Gojcic, Haithem Turki"
venue: "arXiv preprint"
year: 2026
date: 2026-09-27
paper_url: "https://arxiv.org/abs/2605.30215"
tags:
  - Multi-View 3D Reconstruction
  - Weight-Tied Transformer
  - Iterative Refinement
  - Camera Pose Estimation
  - Pointmap Prediction
summary: "DéjàView 以一个权重共享、时间条件化的 frame/global-attention Transformer block 循环细化多视角表示，仅在最后一步解码深度、ray 和相机；它以 117M 参数达到五个基准上最高平均重建/位姿表现，并可通过推理步数 K 在质量与算力间交换。"
permalink: /papers/dejavu-looping-transformers-for-multiview-3d-reconstruction/
---

> **阅读依据**：arXiv:2605.30215（官方 PDF）。本文重点讨论 v1 所定义的模型和实验；正文未给出可验证的官方代码链接，因此不补充未核实地址。

## 一句话总结

DéjàView 将多视角重建 Transformer 中原本逐层独立的深层计算，改写为同一个 Transformer block 对隐状态重复执行 $K$ 次；连续时间条件和残差门控让该共享 block 学会“第几步该做多大更新”，从而用约 117M 参数获得与大模型相当或更高的平均重建质量，并让 $K$ 成为推理时可调的质量–算力旋钮。

## 背景与核心问题

近年来端到端多视角重建模型不断堆叠参数与不共享层数，以联合预测 depth、pointmap 与相机。作者认为这种“用不同参数购买迭代”的做法有浪费：已有分析显示训练后的连续 Transformer 层可由少量循环 block 近似；而多视角网络各 decoder 层的预测也会逐步细化。

因此问题不是“能否再加深 Transformer”，而是：

- 能否让一个共享 block 在每次循环中同时做跨视角推理和几何细化？
- 能否在不为每个中间步骤都解码/监督的前提下稳定训练？
- 能否把循环次数变成部署时可选择的计算预算，而非固定网络深度？

## 方法详解

### 整体框架

<div class="mermaid">
flowchart LR
    A[多视角 RGB 图像] --> B[共享 DINOv2 patch encoder]
    B --> C[每视角 patch tokens + register/camera token]
    C --> D[共享 looped Transformer block]
    D --> E{循环 K 次\n连续时间条件}
    E -->|下一步| D
    E -->|最终 z_K| F[Depth decoder]
    E -->|最终 z_K| G[Ray decoder]
    G --> H[Camera MLP 或 rays 解析求相机]
    F --> I[depth / pointmap]
    H --> I
</div>

输入 $V$ 张图像，模型预测以第一视角为参考坐标系的稠密 depth 与 ray map。每像素三维点由

$$
X=R_o+D(u,v)R_d
$$

得到，其中 $R_o,R_d$ 分别为预测 ray origin 和未归一化方向。再从 ray map 解析得到每视角的旋转、平移和内参；论文也提供从 camera token 直接经 MLP 解码的更快变体，但默认推理使用 rays 的解析恢复。

### 1. 共享循环，而非共享后冻结

所有图像先由 DINOv2 ViT-B 编码成 patch tokens；每个视角还加入 learnable register tokens 与 camera token。一个 looped block 重复作用于状态 $z_k$：

$$
z_{k+1}=f_\theta(z_k,t_k,t_{k+1}).
$$

block 先做每视角独立的 frame attention（保留 2D rotary position embedding），再做跨全部视角 token 的 global attention。与 RAFT 一类“固定特征 + 小更新器”不同，DéjàView 每次循环均是完整 Transformer 计算，因此跨视角匹配、全局几何推理和细化在同一状态中交替发生。

连续时间 $(t_k,t_{k+1})$ 而不是离散 step ID 是关键：它使同一组参数可覆盖一段 $K$，在训练范围内以不同 $K_{\mathrm{inf}}$ 推理。训练时从 $[K_{\min},K_{\max}]$ 采样 $K$，推理则在均匀时间网格运行 $K_{\mathrm{inf}}$ 步。

### 2. 时间条件的残差门控

对 attention、MLP 和输出各生成一个随时间区间变化的通道 scale：

$$
\begin{aligned}
z'&=z_k+s_{\mathrm{attn}}\odot\mathrm{LS}_1(\mathrm{Attn}(\mathrm{LN}_1(z_k))),\\
z''&=z'+s_{\mathrm{mlp}}\odot\mathrm{LS}_2(\mathrm{MLP}(\mathrm{LN}_2(z'))),\\
z_{k+1}&=s_{\mathrm{out}}\odot z''.
\end{aligned}
$$

scale 由零初始化 MLP 对两个时刻的正弦嵌入生成，形式为 $s=1+\mathrm{MLP}(\eta(t_k,t_{k+1}))$。直觉是早期循环可进行较大修正，后期学习更小的细化步；论文的 residual-stream 分析显示相对更新幅度约从 0.5 降至 0.1，而状态与终点方向的 cosine similarity 单调逼近 1。状态范数本身并不收敛，但 decoder 前的 LayerNorm 吸收了平行方向的增长，作者称此为 **directional refinement**。

### 3. 只在终点解码与监督

循环中间状态不通过 decoder。最终 $z_K$ 才进入两个浅层 decoder：ray 分支用 linear pixel-shuffle，depth 分支用卷积头以避免 patch boundary artifacts，并输出 depth confidence。训练只监督最终预测，节省 $K-1$ 次 decoder 前/反向传播。

损失包含归一化后的 depth $L_2$ 与多尺度梯度项、ray $L_1$、由 depth/ray 解析得到的 pointmap $L_2$，以及相机平移、旋转、FOV 的损失：

$$
\mathcal L=\mathcal L_D+\mathcal L_{\mathrm{grad}}+\mathcal L_R+\mathcal L_X+\mathcal L_{\mathrm{cam}}.
$$

训练分两阶段：先端到端训练所有项与 linear pixel-shuffle depth head；再换为卷积 depth head，只微调 depth decoder，冻结其他参数并关闭 ray/camera loss。实现使用 128 张 H100、每场景最多 18 视角、最长边 504 px，固定 token budget 约 2.5M。

## 实验关键数据

### 设置

评测 DTU、ETH3D、7-Scenes、ScanNet++、nuScenes 共五个基准，覆盖室内、实验室与户外驾驶。重建采用 Sim(3) 对齐后的 pointmap relative $L_2$（低为好）与 3% inlier ratio（IR，高为好）；位姿以 AUC@3° / AUC@30°（高为好）衡量旋转、平移误差的最大值。

对比包含 VGGT、Pi3、MapAnything、Depth Anything 3、MASt3R 与 MASt3R-SfM。为公平起见，论文用官方 checkpoint 并经统一评测框架运行；不同基线的配对数、检索和后处理成本需与单次 feed-forward 成本分开理解。

### 效率与平均质量

| 方法 | 参数量 | 总 FLOPs（24 views） | 每图 FLOPs | 峰值显存 | 平均 IR | 平均 AUC@30° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VGGT | 1,257M | 190.0 T | 7.9 T | 14.7 GiB | 77.4 | 88.6 |
| Depth Anything 3-L | 356M | 99.8 T | 4.2 T | 7.9 GiB | 74.8 | 90.7 |
| Pi3 | 959M | 153.8 T | 6.4 T | 6.6 GiB | 58.3 | 79.7 |
| **DéjàView** | **117M** | **75.9 T** | **3.2 T** | **4.9 GiB** | **80.3** | **91.8** |

表中为论文 Table 3 在五基准上的平均结果。DéjàView 以最小参数量取得最高平均 IR 和 AUC@30°；相对 VGGT 参数减少约 10.7 倍、每图 compute 约 2.5 倍更低。摘要所称“匹配或超过更大 feed-forward baselines”应理解为跨基准平均，不等于每一个单项都第一。

论文指出 Pi3 是最接近竞争者：其室内 pointmap 精度突出，但约为 DéjàView 的 8 倍参数、2 倍 compute。VGGT 在 DTU 的最严格 AUC@3° 位姿指标上仍有明显优势（96.5 vs. 83.2），但 AUC@30° 差距缩至约 1 点；而在 ScanNet++ 上，DéjàView 相比 concurrent VGGT-1B 报告 AUC@3° 接近 +50 点、AUC@30° 超过 +10 点。

### 循环行为与消融

从 $K=2,4,8,16$ 解码中间状态的实验显示，训练区间内 depth/pointmap 与 pose 指标单调改善。$K_{\mathrm{inf}}$ 因而可作为实际部署中的 compute knob；论文也明确提示，大幅超出训练范围会退化。

架构消融从“16 个互不共享、无时间条件 block”开始，依次加入 weight tying、time-conditioned attention/MLP gates 和 output state gate；每一项均改善五基准的指标。关键结论是：共享权重并非只为压缩，时间条件残差门控使该共享 block 能承担不同细化阶段的不同更新角色。

## 亮点与局限

**论文亮点**：

- 把 Transformer 深度改写为显式循环，直接把“模型深度”转化为可调推理时间预算。
- 不把循环局限于 task-space 输出；完整 frame/global attention 在每一步重新进行跨视角推理，适合多视角几何中匹配与优化耦合的特点。
- 中间状态不解码仅监督终点，训练成本控制得较好；同时以时间条件避免一个固定 step count 把共享权重锁死。
- 参数、FLOPs、显存、重建与位姿指标同时报告，效率主张较完整。

**作者承认的边界**：循环数显著超出训练区间会退化；隐状态表现为方向收敛而不是固定点收敛，因此不能直接套用深度平衡模型（DEQ）的任意步数推断观点。

**我的分析**：

- 每一步仍包含全局 attention，$K$ 增加时延迟线性上升；这不是“免费多想几步”。对实时系统还需报告逐步 latency、不同视角数下的峰值显存和早停策略。
- 所谓方向收敛依赖 decoder LayerNorm 的尺度不变性，若未来任务换成对绝对 feature magnitude 敏感的 decoder 或下游模块，$K$ 的可外推范围可能更窄。
- 多视角模型对输入视角集合、顺序、动态物体和相机内参误差的敏感性未由平均基准完全覆盖；应增加视角 dropout、乱序与动态遮挡条件下的 iteration–quality 曲线。

## 与相关工作对比

| 方法 | 核心表示/计算方式 | 迭代机制 | 与 DéjàView 的区别 |
| --- | --- | --- | --- |
| VGGT | DINOv2 多视角 Transformer | 固定的深层、参数不共享 | DéjàView 将层深压为一个循环 block，参数/显存更低 |
| Pi3 | 大型 feed-forward 多视角重建 | 固定网络深度 | 室内几何强，但参数与 compute 更高 |
| RAFT 类方法 | 固定特征 + 小型 recurrent updater | 输出/任务空间细化 | DéjàView 每步重做完整跨视角 Transformer 推理，细化内部 token state |
| iLRM | 将不共享 Transformer 层视作优化步骤 | 迭代但参数不共享 | DéjàView 的 block 在所有步共享，且状态是 per-view tokens |
| **DéjàView** | 共享 frame/global Transformer | 连续时间条件循环 $K$ 步 | 用 $K$ 显式调节质量–计算 |

## 评分

| 维度 | /10 | 依据 |
| --- | ---: | --- |
| 创新性 | 8.4 | 将 weight tying、连续时间条件和多视角几何 Transformer 结合为真正可控的循环推理。 |
| 技术可靠性 | 8.2 | 方向收敛分析、逐步质量与 block 消融形成闭环；超训练步数稳定性仍有限。 |
| 实验充分度 | 8.4 | 五个异质基准，覆盖重建、位姿、参数、FLOPs 与显存。 |
| 写作清晰度 | 8.3 | 状态、解码和时间条件的分工明确，图示与公式相互对应。 |
| 实用/研究价值 | 8.5 | 对资源受限但可容忍多步推理的多视角重建很有吸引力。 |

**总体推荐：值得细读。** 它最有价值的主张不是“共享权重也能做 3D”，而是把足够深的多视角推理看成一个可控制的迭代过程，并在结构中保留每一步都能跨视角重新推理的能力。

## 阅读结论

- **最值得记住的点**：一个共享 Transformer block 不是重复相同操作；连续时间门控让它在不同循环阶段执行不同尺度的几何修正。
- **最需要怀疑的点**：方向收敛和 decoder LayerNorm 的配合，是否足以支撑比训练区间更长的可靠推理。
- **最值得复现或继续验证的点**：加入基于 update norm/预测不确定度的自适应 early stopping，并测量它在视角数、动态干扰和 OOD 条件下的 compute–quality Pareto 曲线。

## 相关论文

- [VGGT](https://arxiv.org/abs/2503.11651) — DéjàView 的主要 feed-forward 多视角 Transformer 对照与 frame/global attention 来源。
- [DUSt3R](https://arxiv.org/abs/2312.14132) — pointmap、几何监督和 confidence-weighted loss 的重要前身。
- [MASt3R](https://arxiv.org/abs/2409.12939) — pairwise dense matching / reconstruction 相关基线。
- [RAFT](https://arxiv.org/abs/2003.12039) — 任务空间 recurrent refinement 的经典对照；DéjàView 与其细化位置不同。
