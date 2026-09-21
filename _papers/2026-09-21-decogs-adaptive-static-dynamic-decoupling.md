---
title: "DecoGS: Adaptive Static-Dynamic Decoupling of 3D Gaussians for Free-Viewpoint Video Streaming"
subtitle: "用图像差分定位变化区域，只更新相关 3D Gaussians，以减轻在线动态重建的静态漂移与冗余计算"
authors: "Idil Sulo, Alexey Supikov, Ilke Demir, Sainan Liu"
venue: "arXiv preprint"
year: 2026
date: 2026-09-21
paper_url: "https://arxiv.org/abs/2609.17230"
tags:
  - Dynamic 3D Gaussian Splatting
  - Streaming Reconstruction
  - Free-Viewpoint Video
  - Static-Dynamic Decoupling
summary: "DecoGS 在流式动态 3D Gaussian Splatting 中以相邻帧像素差分得到运动 ROI，投影筛选并只优化对应 Gaussians，同时清除未选中高斯的 Adam 状态、限制增密位置并对 ROI 加权监督；它避免静态区域被反复更新，在 N3DV/MeetRoom 上报告 34.55/31.60 dB PSNR、约 260 FPS，以及近零静态区闪烁。"
permalink: /papers/decogs-adaptive-static-dynamic-decoupling/
---

> **阅读依据**：arXiv:2609.17230v1（11 页，用户提供的 PDF）。文中未给出可验证的官方代码或项目链接，故此处不臆测。所有“论文报告”均来自该 PDF；“我的分析”明确标注。

## 一句话总结

DecoGS 为在线动态 3DGS 加上一条低成本的“变化感知”控制链：由帧差找出会变化的图像区域，只对投影至这些区域的 Gaussians 反传和增密，并冻结其余高斯及其优化器状态，从而在不需要大规模预训练的前提下同时提升流式重建质量、时序稳定性与渲染速度。

## 背景与核心问题

自由视点视频（Free-Viewpoint Video, FVV）要求模型随多视角视频流到来持续更新场景。离线动态 NeRF/3DGS 能访问完整序列，但不适用于低延迟直播；在线方法虽逐帧训练，却常把**所有** Gaussians 都更新。即使某个区域本身静止，也会累积来自动态帧的梯度与 Adam 动量，最终出现静态背景漂移、ghosting 和 flicker。

论文的观察是：动态场景中超过一半的 Gaussians 在大多数时刻无需更新；若能用当前帧的廉价证据局部化真正变化的区域，在线优化可从“全场景重训”变为“仅更新变化子集”。

核心挑战有三个：

- 如何不用光流或额外分割网络，在流式条件下找到动态区域？
- 如何将 2D 变化区域可靠、低成本地对应到 3D Gaussians，并阻断未选中高斯的参数和优化器漂移？
- 新物体出现时仍需增密，怎样避免把高斯数量无控制地扩张？

## 方法详解

### 整体框架

<div class="mermaid">
flowchart LR
    A[时刻 t 的多视角帧] --> B[相邻帧像素差分]
    B --> C[阈值化并 max-pooling 膨胀：2D ROI]
    C --> D[相机投影：选出 ROI 内可见 Gaussians S_t]
    D --> E[仅对 S_t 梯度门控优化 NTC/3DGS]
    D --> F[ROI 内选择性 prune/clone 增密]
    E --> G[未选中 Gaussians：参数冻结 + Adam 状态清零]
    E --> H[focus-aware L1 + SSIM 损失]
    F --> I[下一时刻紧凑、时序稳定的 Gaussians]
    G --> I
    H --> I
</div>

初始时刻 $t=0$ 正常优化一套 3D Gaussians；之后每一个时刻先用上一时刻模型初始化，再执行图中选择—优化—增密流程。神经时序校正（Neural Transformation Cache, NTC）每帧训练 250 steps，选择性增密从第 150 step 后启用，最后 100 steps 才联合优化选中的 Gaussians。

### 1. 自适应静动态解耦：帧差而非显式运动网络

对相邻时刻、相机 $v$ 的像素 $(x,y)$，在 RGB 通道 $i_c$ 上取最大绝对差：

$$
D_t^v(x,y)=\max_{i_c}\left|I_t^v(x,y,i_c)-I_{t-1}^v(x,y,i_c)\right|.
$$

再以阈值 $\tau$ 二值化，并以半径 $r$ 的 max-pooling 膨胀：

$$
M_t^v(x,y)=\mathbb{1}[D_t^v(x,y)>\tau],\qquad
\tilde M_t^v=\operatorname{MaxPool}(M_t^v;r).
$$

直觉上，膨胀后的 mask 既容纳物体边界，也缓解了像素级差分对微小定位误差的敏感性。它衡量的是光度变化，不显式估计速度，因此额外成本很低；代价是它也会把光照变化等同于“需要更新”。

### 2. 从 2D ROI 到动态 Gaussian 集合

将每个 Gaussian 中心投影到各相机图像。若其可见且投影点落在至少 $\tau_v$ 个视角的活动像素中，便进入选择集 $S_t$；论文实现取 $\tau_v=2$，并与逐视角 visibility filter 相交以排除遮挡点。

这一步只用现有相机内外参和 Gaussian 中心投影，避免光流 lifting 或额外的 2D-to-3D 对应网络。其作用不只是“少算一些高斯”，更是把后续更新的因果范围限定到观测发生变化的位置。

### 3. 选择性优化：梯度门控与优化器状态重置

所有 Gaussians 仍参与渲染，但仅 $S_t$ 接受反向传播。对第 $i$ 个高斯参数 $\theta_i$，论文的门控可概括为：

$$
\theta_i \leftarrow
\begin{cases}
\theta_i-\eta\nabla_{\theta_i}\mathcal L,& i\in S_t,\\
\theta_i,& i\notin S_t.
\end{cases}
$$

仅将梯度置零还不够：未选中高斯的 Adam 一、二阶动量会保留旧的动态信号，等该高斯以后再被选中时仍可能造成漂移。因此每次更新后，对 $i\notin S_t$ 同时置零其动量 $m_i,v_i$。这是该方法比单纯 mask loss 更关键的工程细节。

### 4. ROI 内选择性增密与聚焦损失

传统 3DGS 的 prune/clone 可能在任何位置生成高斯。DecoGS 只在 ROI 内进行增密，并对新生成但投影落在所有 ROI 外的 Gaussian 回置至父节点位置；文中称被选作增密的高斯从不超过动态场景的 35%，高梯度新生区域采用较低阈值 $\tau_d=0.0001$。

损失由全局 $L_1$、ROI 加权 $L_1$ 和仅在 ROI 上计算的 SSIM 组成：

$$
\mathcal L=(1-\lambda_{\mathrm{ssim}})
\left[\lambda_1\mathcal L_1+\lambda_2\mathcal L^{\mathrm{focus}}_1\right]
+\lambda_{\mathrm{ssim}}\mathcal L^{\mathrm{focus}}_{\mathrm{ssim}}.
$$

全局项防止局部优化破坏整帧，focus 项提高变化区域的梯度权重；而由于静态高斯已被门控，SSIM 只在 ROI 内计算即可集中预算。

## 实验关键数据

### 设置

论文评估 N3DV（21 个多视角相机、$2704\times2028$、30 FPS，报告 300 帧序列）和 MeetRoom（13 个 Azure Kinect、$1280\times720$、30 FPS，快速变化与运动模糊更强）。每个数据集预留一个相机作测试视角，其余视角训练；实验在 RTX 4090 上进行。

指标包括 PSNR（dB，越高越好）、SSIM、渲染 FPS（越高越好）、逐帧训练时间，以及静态区 masked Total Variation（mTV，越低表示闪烁越小）。比较对象同时包括离线方法和在线方法，如 IGS、3DGStream、ComGS、4DGC、HiCoM、ReConGS、QUEEN 与 MoRGS。

### 主结果

| 数据集 | DecoGS | 最接近在线基线 | 论文结论 |
| --- | --- | --- | --- |
| N3DV | **34.55 dB PSNR**，**261 FPS** | 3DGStream：论文报告整体 PSNR 低 **2.01 dB** | 质量更高，速度仍同级 |
| MeetRoom | **31.60 dB PSNR**，约 **260 FPS**，**4.3 s/frame** | ComGS：31.49 dB、98 FPS、28.3 s/frame | +0.11 dB，约 2.65 倍 FPS、约 6.6 倍更快训练 |
| MeetRoom vs. 3DGStream | +**0.81 dB** PSNR | 训练时间量级相近 | 局部更新在更困难场景仍有效 |

N3DV 上，论文还报告 DecoGS 相比 IGS 在两个测试序列平均高 0.40 dB；IGS 虽每帧优化快，但需在 4 个 N3DV 场景上进行 192 GPU-hours 预训练。DecoGS 不需这种大规模预训练，不能把该离线成本忽略不计后再比较。

### 时序稳定性与消融

在 N3DV 的 *coffee martini* 与 *flame steak* 静态区域，DecoGS 的 mTV 分别为 **0.003 / 0.004**，论文称相对 3DGStream 的闪烁最多降低约 **70×**。这与“未选中静态高斯既不更新也不保留 Adam 状态”的设计链条相符。

| 消融（N3DV flame salmon） | PSNR (dB) |
| --- | ---: |
| 无选择、全高斯优化 | 28.61 |
| 仅 Dynamic Gaussian Selection | 30.81 |
| Selection + Focus loss | 30.83 |
| **Selection + Focus + Gradient gate（完整）** | **30.92** |

选择模块单独带来 **+2.20 dB**，是最主要增益；focus loss 与梯度门控继续带来较小但稳定的改善。时间采样消融中，*sear steak* 不采样（30 FPS）为 34.40 dB，采样率 5（6 FPS）为 33.68 dB，采样率 10（3 FPS）为 33.25 dB，说明过度稀疏的时间监督会损失质量。

## 亮点与局限

**论文亮点**：

- 用帧差、投影和 mask 三个已有信号闭合“检测变化→更新相关高斯”的流程，避免昂贵预训练与外部运动模型。
- 不只冻结参数，还清除未选中高斯的 Adam 状态，针对在线优化漂移给出具体且有说服力的机制。
- 质量、在线训练速度、渲染速度和时序闪烁均有对应实验；选择模块的消融尤其清楚。

**作者承认的边界**：方法以光度变化为动态证据；运动模糊、光照变化或低纹理区域会削弱差分 mask 的可靠性。并且它本质上优先保持静态区，极快运动或新物体大幅显现仍依赖 ROI 覆盖与增密机制。

**我的分析**：

- “静态”并不必然意味着不更新。曝光自动变化、阴影、反射和相机标定微误差均会触发帧差，而真正缓慢运动的对象可能低于阈值；应补充按光照变化、运动模糊和跨域场景分组的 mask precision/recall，或至少报告阈值敏感性。
- 按中心点是否落入 ROI 选择 Gaussian，未完整利用其投影椭圆覆盖范围；边界附近的大高斯可能漏选或被过选。可比较 center-only、footprint-overlap 和可微软权重选择。
- 论文以 PSNR/mTV 为主，但“新物体出现”是关键卖点，应额外报告新显现区域的质量曲线、Gaussian 数量增长、峰值显存及最坏帧延迟。

## 与相关工作的对比

| 方法 | 在线性 | 核心机制 | 与 DecoGS 的差别 |
| --- | --- | --- | --- |
| 3DGStream | 在线 | 逐帧更新动态 Gaussian 表示 | 更新范围更广，论文观察到静态区 ghosting/flicker |
| IGS | 在线 | 预训练运动网络直接推断更新 | 帧延迟低，但需 192 GPU-hours 预训练且跨数据泛化受限 |
| ComGS / 4DGC | 在线 | 侧重紧凑表示与存储 | 论文报告其训练/渲染或质量上的取舍更明显 |
| DecoGS | 在线、无预训练 | 帧差 ROI + 2D→3D 筛选 + 梯度/状态门控 | 优先保障动态区适配与静态区稳定的分工 |

## 评分

| 维度 | /10 | 依据 |
| --- | ---: | --- |
| 创新性 | 7.8 | 组件本身朴素，但把 ROI 筛选、优化器状态重置与选择性增密组合成了完整在线机制。 |
| 技术可靠性 | 8.0 | 核心假设有清晰公式、投影选择和消融支持；对光照/模糊鲁棒性仍待更直接验证。 |
| 实验充分度 | 7.6 | 有两套真实多视角数据和时序指标，但场景规模与困难条件覆盖仍有限。 |
| 写作清晰度 | 8.2 | 从问题到模块、损失和消融的对应关系明确。 |
| 实用/研究价值 | 8.3 | 对需要低延迟又在意背景稳定的实时 FVV 系统，易于作为现有 3DGS 在线管线的插件。 |

**总体推荐：值得细读。** 它最值得借鉴的不是单纯“少更新高斯”，而是把选择集合同时作为梯度、优化器状态、增密和损失分配的统一控制变量。

## 阅读结论

- **最值得记住的点**：在线 3DGS 的静态漂移不只来自参数梯度，也来自未清除的优化器历史；冻结两者才真正稳定。
- **最需要怀疑的点**：帧差 ROI 是否能在曝光、反射、低纹理和快速运动下继续正确区分“应更新”与“应冻结”。
- **最值得复现或继续验证的点**：在相同 backbone 上做 ROI 中心点、Gaussian footprint overlap 和软选择的对照，并报告动态/静态区域分开的质量、mTV、显存和尾延迟。

## 相关论文

- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) — DecoGS 所继承的显式 Gaussian 渲染表示。
- [3DGStream](https://arxiv.org/abs/2407.21708) — 流式动态 3D Gaussian 重建的直接在线对照。
- [Instant Gaussian Stream](https://arxiv.org/abs/2503.15282) — 以预训练运动网络降低逐帧更新开销的在线路线。
- [ClipGStream](https://arxiv.org/abs/2608.12787) — 面向更长序列的离线/片段级动态 Gaussian 优化对照。
