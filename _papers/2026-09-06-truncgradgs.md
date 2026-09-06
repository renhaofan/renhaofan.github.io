---
title: "TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates"
subtitle: "用连续线性 surrogate 扩展低 opacity Gaussians 的反向梯度范围"
authors: "Théo Morales, Nhat-Quynh Le-Pham, Robin Atkins, Binh-Son Hua"
venue: "Pacific Graphics 2026 (Short Papers)"
year: 2026
date: 2026-09-06
paper_url: "https://arxiv.org/abs/2609.03534v1"
code_url: ""
project_url: ""
tags:
  - 3D Gaussian Splatting
  - Differentiable Rasterization
  - Optimization
  - Vanishing Gradient
  - Dynamic Scene Reconstruction
  - Dataset
summary: "TruncGradGS 保持 Gaussian splatting 的前向渲染不变，只对低 opacity 的 dead Gaussians 启用救援：当其当前 splat radius 低于阈值时扩大 tile 覆盖，再以连续线性 surrogate 替换反向传播中近零的 Gaussian 导数尾部；它使 N3DV 上 4DGS 从 30.22 提升到 31.70 dB、Gaussian 数减少 38.2%，但训练慢 1–10 倍，且部分 LPIPS/SSIM 与 CEM-4DGS 结果并未改善。"
permalink: /papers/truncgradgs/
---

> **阅读依据**：本笔记基于 [arXiv:2609.03534v1](https://arxiv.org/abs/2609.03534v1)（2026-09-03）的完整官方 HTML 与 12 页 PDF。论文为 **Pacific Graphics 2026 Short Papers**，作者 Théo Morales、Nhat-Quynh Le-Pham、Binh-Son Hua 来自 Trinity College Dublin，Robin Atkins 来自 Dolby Laboratories。正文称代码和新数据集将在 publication 时公开；截至 2026-09-06，arXiv 页面和针对性 Web/GitHub 检索均未发现对应官方仓库、项目页或数据下载链接。
>
> **材料边界**：正文两次提到 supplementary material，其中应包含等值线交点的完整推导与训练开销细节，但当前 12 页 arXiv PDF 在参考文献后结束，未附 supplement。因此，核心公式和 Tables 1–5 可以核对，slope $m$、radius padding、训练阶段切换和 motion-mask processing 等复现细节仍不完整。

## 一句话总结

TruncGradGS 不改变 3DGS 的前向图像与推理，而是给**本应因低 opacity 被剪枝的 Gaussians**一段训练期“救援窗口”：当其当前 projected radius 低于阈值时扩大 rasterization tile 范围，并在真实 Gaussian derivative 已指数衰减的远端用连续线性 surrogate 提供更强的位置梯度，使差初始化和大时空位移下的 primitives 有机会移动到有用区域；代价是 biased gradient、复杂保护条件和最高约 10 倍训练开销。

## 背景与动机

- **任务**：从多视角图像或视频优化一组 3D Gaussians，使其能从未见视角渲染静态或动态场景。
- **3DGS 的高效来源**：renderer 将图像划成 $16\times16$ pixel tiles；只把投影椭圆 bounding box 相交的 Gaussians 分配给一个 tile，并丢弃 contribution $\sigma\alpha$ 低于阈值的 Gaussian–pixel pairs。
- **效率机制的优化副作用**：未被分配到 tile 的 primitive 没有梯度；即使人为扩大 splat radius，Gaussian density 及其导数在远端也会指数趋近零。一个初始化位置很差的 Gaussian 无法从远处目标像素得到足够信号，自然也很难“走”到正确区域。
- **现有补救的局限**：adaptive density control（ADC）通过 clone/split 扩张覆盖，但依赖已有 gradient；RAIN-GS 等改进初始化，MCMC/densification 方法重新分配容量，却没有直接改变 Gaussian derivative 的局部性。
- **动态场景更加严重**：若整个序列从第一帧初始化并按时间优化，远期物体位置与初始点云可能相差很大。标准 rasterizer 的局部梯度使 primitive 很难跨过这段距离。
- **作者切入点**：保留真实 Gaussian gradient 的局部精度，只在 density 低于 cutoff 的尾部接一条连续线性 surrogate，扩大“有效反向支持域”；前向渲染仍使用原 Gaussian，因此不改变推理质量定义和速度。

**核心研究问题：能否不增加推理成本、不放弃 tiled rasterization，只通过训练期的 backward gradient field，让弱初始化或大位移 Gaussians 接收到远距离像素的有效引导？**

## 核心问题

1. **为什么扩大 splat radius 仍不够？** Tile coverage 只决定 Gaussian–pixel pair 会不会进入 kernel；真实 Gaussian derivative 在尾部仍接近零，进入 kernel 不代表能获得有效更新。
2. **如何扩大梯度范围又不破坏局部最优？** 对所有 Gaussians 使用 surrogate 会让已经拟合良好的 primitives 继续被远处 residual 拉动，造成 divergence。
3. **怎样越过 renderer 的两层过滤？** 必须同时处理 tile bounding box culling 和 $\sigma\alpha<1/255$ 的 contribution filtering。
4. **如何防止 surrogate 产生错误的远程排斥？** 只应在某像素能把 Gaussian 拉向更好解释时使用，而不能把所有 loss gradient 方向都传播到远端。
5. **如何控制训练成本？** 扩大 tile coverage 会显著增加 Gaussian–pixel pairs，必须把额外 backward 限制在少量候选 primitives/动态像素上。

## 方法详解

### 整体框架

<div class="mermaid">
flowchart TB
    V[静态多视角图像或动态视频] --> I[Random / COLMAP 初始化]
    I --> G[3D / 2D / Dynamic Gaussians]

    G --> F[标准 tiled Gaussian forward]
    F --> R[渲染图像与 photometric loss]

    G --> D{opacity 低于 pruning threshold?}
    D -->|否| N[标准 backward]
    D -->|是| H{当前 splat radius 低于 padding threshold?}
    H -->|是| P[扩大 splat tile radius]
    H -->|否| M{动态方法且像素静态?}
    P --> M
    M -->|是| S[忽略额外贡献]
    M -->|否| C{Gaussian-pixel contribution}

    C -->|sigma alpha >= 1/255| N
    C -->|低 contribution 且在 cutoff 外| B{低 density 还是低 opacity?}
    C -->|低 contribution 且在 cutoff 内| K[skip]
    B -->|低 density| A{alpha gradient 为 pull?}
    B -->|低 opacity| X[v1 称仅计算 opacity gradient<br/>但与 mean 推导及 Fig. 4 冲突]
    A -->|是| T[mean 的 continuous linear surrogate]
    A -->|否| K
    X --> U

    N --> U[更新参数]
    T --> U
    U --> O[截断梯度阶段与标准 ADC 阶段交替]
    O --> Q[延迟 pruning 到训练末期]
    Q --> G

    G --> E[推理：完全使用标准 renderer]
</div>

方法可以拆成两条路径：

- **Forward / inference**：仍按标准 3DGS 计算 Gaussian density、alpha compositing 和图像，不使用 surrogate。
- **Training backward**：只对低 opacity、原本将被 pruning 的 dead Gaussians 启用救援；仅当其当前 splat radius 低于未披露阈值时扩大 tile coverage。对 cutoff 外且 $\partial\mathcal L/\partial\alpha_i<0$ 的 low-density pairs，正文描述为 mean 的截断梯度；low-opacity 子情形则被写成只更新 opacity，与全文的 mean-gradient 推导和 Figure 4 标注冲突，v1 无法唯一还原。

因此 TruncGradGS 不是把大梯度裁小的 gradient clipping，也不是 truncated backpropagation through time；这里的“truncated”是**截断 Gaussian derivative 的指数尾部，并用更强的线性尾部替代**。

### 1. Vanishing gradient 来自哪里

设投影后的 2D Gaussian 为：

$$
G^{2D}(\Delta)
=\exp\!\left(-\frac12\Delta^T{\Sigma'}^{-1}\Delta\right),
\qquad
\Delta=\boldsymbol\mu^{2D}-\mathbf p,
$$

其中 $\mathbf p$ 是 pixel，$\boldsymbol\mu^{2D}$ 是 projected mean，$\Sigma'$ 是 projected covariance。对 mean 的 loss gradient 可写成链式法则：

$$
\frac{\partial\mathcal L}{\partial\mu_x^{2D}}
=
\frac{\partial\mathcal L}{\partial G^{2D}}
\frac{\partial G^{2D}}{\partial\Delta_x}
\frac{\partial\Delta_x}{\partial\mu_x^{2D}}.
$$

用矩阵形式看，Gaussian 对位移的导数是：

$$
\nabla_{\Delta}G^{2D}
=-G^{2D}(\Delta){\Sigma'}^{-1}\Delta.
$$

决定远端行为的是乘法因子 $G^{2D}(\Delta)$：Mahalanobis distance 墠大时，指数项迅速趋近零，即使 pixel loss 很大，整条链式梯度仍会消失。Rasterizer 的 tile culling 和 $\sigma\alpha$ threshold 又进一步把许多 pair 直接删掉。

论文 Eq. 6–8 尝试把该式展开为 conic coefficients $a,b,c$，但其分量下标/系数标记与 Eq. 8 的标准导数并不完全一致；上面的矩阵表达避免了这一排版歧义，也保留了作者真正需要的结论：**每个 spatial derivative 都被 Gaussian density 本身指数门控**。

### 2. Piecewise truncated gradient

定义 squared Mahalanobis distance：

$$
d(\mathbf p)^2
=(\boldsymbol\mu-\mathbf p)^T
\Sigma^{-1}
(\boldsymbol\mu-\mathbf p)
=-2\log G.
$$

给定 density threshold $\tau$，等值线满足：

$$
G(\mathbf x_b)=\tau
\quad\Longleftrightarrow\quad
 d(\mathbf x_b)^2=-2\log\tau.
$$

$\mathbf x_b$ 是从 query pixel 指向 Gaussian mean 的 ray 与该 isocontour 的交点。论文通过解一个 quadratic equation 得到它；这一步要针对进入额外 backward 的 query pairs 计算。

设某一坐标方向的真实导数为 $g(x)=\partial G/\partial\Delta_x$。截断梯度的核心是：

- 等值线内部仍使用 $g(x)$；
- 等值线外使用穿过 boundary gradient 的线性函数；
- 根据导数正负使用 `min` 或 `max`，避免 surrogate 穿过真实梯度的错误一侧。

连续性由下式保证：

$$
b_{\ell}=g(x_b)-m x_b,
$$

于是 linear surrogate $mx+b_{\ell}$ 在 $x_b$ 处与真实导数相等。这里用 $b_{\ell}$ 表示 line intercept，以免与论文 conic coefficient $b$ 混淆。

- $m$ 较小：尾部接近 constant long-range pull，作用范围大；
- $m$ 较大：离开边界后更快衰减，偏差小但探索范围短；
- $\tau$ 决定真实梯度与 surrogate 的切换等值线。

论文把 $\tau$ 设为 rasterizer 的 alpha-blending discard threshold，但实现段落又用 contribution $\sigma\alpha=1/255$ 决定 normal/truncated path。一个是 density，一个是 opacity-weighted contribution，二者在低 opacity Gaussian 上并不等价；当前正文没有完整解释如何映射。

### 3. 为什么只救 dead Gaussians

对所有 Gaussians 应用 surrogate 会改变局部最优附近的 gradient：即使一个 active Gaussian 已正确拟合局部表面，远处大量 pixels 的线性尾部仍可能累计成更大的错误位移。作者的解决方案是只对：

$$
\sigma_i<\tau_{\mathrm{pruning}}
$$

的 dead Gaussians 启用截断路径。这些 primitives 按 baseline 规则本来就会被 pruning，因此 surrogate 的作用是给它们一次迁移和复活机会，而不是持续扰动成熟结构。

训练策略还包含：

1. 截断梯度 optimization 与正常 optimization/ADC 阶段交替；
2. ADC 只在正常阶段开启，避免 surrogate 污染 accumulated view-space gradient 与 densification decision；
3. pruning 延迟到训练末期，让低 opacity primitives 有足够时间移动并恢复；
4. surrogate 只在 loss 对 alpha 的 gradient 符号对应 long-range pull 时使用，过滤会形成 long-range push 的方向。

**独立判断**：这组 guard 条件非常关键，也说明方法不是某个真实 forward objective 的普通梯度下降，而是**有条件的 biased update rule**。论文没有给出收敛证明；标题为 convergence analysis 的小节实际提供的是风险讨论与经验性保护策略。

### 4. Normal、Truncated 与 Skip 三条路径

论文在 contribution threshold $1/255$ 附近定义三种 backward 行为：

- **Normal path**：$\sigma\alpha\ge1/255$，该 Gaussian 已参与 pixel forward，使用真实 gradient。
- **Truncated path（低 density 子情形）**：$\sigma\alpha<1/255$、$d(\mathbf p)^2>-2\log\tau$（等价于 $G<\tau$），且满足 dead-Gaussian 与 $\partial\mathcal L/\partial\alpha_i<0$ 条件时，正文描述为对 mean 使用线性 surrogate。
- **低 opacity 子情形的歧义**：implementation 又称此时“only compute the gradient for opacity $\sigma$”，与“只研究 mean gradient”及 Figure 4 路径标注冲突；本笔记不替作者将其归入确定的 mean-surrogate path。
- **Skip path**：contribution 虽低，但 query 位于 cutoff 内，或 gradient sign 不适合 long-range pull，则不注入 surrogate。

正文对“低 opacity 情况只计算 opacity gradient”和“本方法只考虑 mean gradient”的描述存在张力；Figure 4 的 B/C/D 字母与 implementation bullet 的引用也不完全一致。没有代码时，无法确定 opacity 与 mean 的实际 CUDA backward 分支。

### 5. Radius padding：让远端 pair 真正进入 kernel

只改 derivative 还不够：若 Gaussian 的投影 bounding box 不与 tile 相交，CUDA kernel 根本不会看到这个 pair。作者因此仅对**当前 projected radius 低于某阈值**的 dead Gaussian 扩大 tile-coverage radius，使更远 pixels 被分配到它的 backward workload。

radius padding 与 truncated gradient 缺一不可：

- padding 解决离散 dispatch/culling；
- surrogate 解决进入 kernel 后的连续导数衰减；
- dead filtering 限制额外 pair 数量和优化偏差。

论文未给出 radius expansion 量、radius threshold、每个 Gaussian 最多覆盖 tiles 数和随训练变化的 schedule。这既影响质量，也直接决定 1–10 倍训练开销。

### 6. 动态场景的 motion-mask 加速

动态方法中，作者用 frame difference 和 image processing 为每个 viewpoint 建立 motion mask，并忽略静态 pixels 的额外 truncated contributions。其目的不是改进前向动态分解，而是减少扩大 tile 后的 backward workload。

该加速依赖 mask quality：光照变化、阴影、噪声可能被当作 motion，缓慢运动或低对比度区域可能漏检。论文没有给出阈值、形态学操作、mask 精度或 `w/o motion mask` 的速度/质量消融，也没有说明 mask preprocessing 是否计入训练时间。

### 7. 训练与推理成本

作者报告：

- 训练根据 scene 与 primitive count，约为 baseline 的 **1–10 倍**；
- 静态 Table 2 的实际倍率约为 1.39–4.78 倍；
- inference 完全不使用 radius expansion 或 surrogate，rendering speed 不变；
- 没有报告 peak VRAM、CUDA pair count、dynamic baselines 的训练时间或 motion-mask preprocessing cost。

因此它是典型的“以训练计算换初始化鲁棒性/最终质量”方法，不是 training acceleration。对于一次训练、多次播放的场景可能划算；对频繁 per-scene reconstruction 或长动态序列，最坏 10 倍成本可能不可接受。

## 实验关键数据

### 实验设置

#### 静态数据

- **Mip-NeRF360**：unbounded indoor/outdoor scenes；每 8 张取 1 张作为 test，其余 training。
- **新数据集静态子集**：从作者动态 synthetic benchmark 取 still frames；论文没有说明具体取哪些 frames、每 scene 多少 views，记为 `Our dataset`。
- **初始化**：random 与 COLMAP 两套；base representations 为 3DGS、2DGS。

#### 动态数据

- **N3DV / Neural 3D Video**：6 个 indoor clips，每段 10 秒、约 20 cameras、1 个 test viewpoint。
- **作者 synthetic benchmark**：6 scenes——`alley`、`windy tree`、`water cup`、`neon city`、`bouncy balls`、`underwater`；每段 300 frames / 10 秒 / 30 FPS / 1600×900，Blender path tracing，1–4 test viewpoints。
- PDF Figure 2 写 camera rigs 为 **25–50** 台，Section 4.1 写 **25–45** 台，两处不一致。
- dynamic sequences 都从第一帧 COLMAP 初始化，再按时间顺序优化，以制造/保留 cumulative temporal displacement。

#### 指标与 baselines

PSNR/SSIM 越高越好，LPIPS 与 Gaussian 数越低越好。动态 baselines 包括 4DGS、CEM-4DGS、4D-Scaffold；TruncGradGS 作为插件分别加在这些系统上。

### 静态重建：PSNR 全部提高，但并非所有质量指标都提高

Table 2 的八组配对结果：

| Init / Base / Dataset | PSNR：Base → +TruncGrad | LPIPS：Base → +TruncGrad | Train | #GS：Base → +TruncGrad |
| --- | ---: | ---: | ---: | ---: |
| Random / 3DGS / Ours | 24.74 → **25.16** | 0.2106 → **0.1974** | 14 → 55 min | 1.126M → **0.799M** |
| Random / 3DGS / Mip360 | 20.92 → **22.18** | 0.3481 → **0.3343** | 27 → 56 min | 1.180M → **0.971M** |
| Random / 2DGS / Ours | 18.80 → **20.70** | 0.4354 → **0.3883** | 23 → 33 min | **1.231M** → 1.446M |
| Random / 2DGS / Mip360 | 19.82 → **20.33** | 0.3943 → **0.3886** | 34 → 55 min | 1.511M → **1.071M** |
| COLMAP / 3DGS / Ours | 30.07 → **30.86** | 0.1270 → **0.1245** | 9 → 43 min | 1.217M → **0.768M** |
| COLMAP / 3DGS / Mip360 | 27.70 → **27.84** | **0.2149** → 0.2197 | 21 → 86 min | 2.496M → **2.035M** |
| COLMAP / 2DGS / Ours | 25.00 → **26.35** | 0.2017 → **0.1962** | 23 → 32 min | 1.801M → **1.048M** |
| COLMAP / 2DGS / Mip360 | 27.17 → **27.33** | **0.2342** → 0.2403 | 44 → 80 min | 3.068M → **2.701M** |

- **随机初始化收益更明显**：3DGS 在 Mip360 上 +1.26 dB，2DGS 在作者数据上 +1.90 dB，支持“扩大探索范围可修复差起点”的主张。
- **COLMAP 仍有 PSNR 增益**：作者数据上 3DGS/2DGS 分别 +0.79/+1.35 dB；Mip360 仅 +0.14/+0.16 dB，边际较小。
- **“所有质量指标一致改善”不成立**：COLMAP + Mip360 的 3DGS/2DGS LPIPS 分别恶化约 2.2%/2.6%，SSIM 也从 0.8212→0.8205、0.8117→0.8104 略降。
- **模型通常更小，但有反例**：8 组中 7 组 Gaussian 数减少；Random 2DGS / Ours 从 1.231M 增到 1.446M，增加约 17.5%。
- **训练代价很高**：最极端的 COLMAP 3DGS / Ours 从 9 到 43 分钟，约 **4.78 倍**；Mip360 3DGS 从 21 到 86 分钟，约 4.10 倍。

### N3DV 动态重建：4DGS 收益强，其他 backbone 混合

| Backbone | LPIPS：Base → +Ours | SSIM：Base → +Ours | PSNR：Base → +Ours | #GS：Base → +Ours |
| --- | ---: | ---: | ---: | ---: |
| 4DGS | 0.1360 → **0.1339** | 0.9443 → **0.9493** | 30.22 → **31.70** | 2.806M → **1.733M** |
| CEM-4DGS | **0.1328** → 0.1359 | 0.9512 → **0.9519** | 32.07 → **32.29** | **0.331M** → 0.352M |
| 4D-Scaffold | **0.1286** → 0.1311 | **0.9502** → 0.9487 | 31.83 → **31.94** | 0.583M → **0.556M** |

- 加到 4DGS 上最有效：PSNR **+1.48 dB**，SSIM +0.005，LPIPS 略降，Gaussian 数减少 **38.2%**。
- CEM-4DGS 只有 PSNR +0.22 dB、SSIM +0.0007；LPIPS 恶化约 2.3%，Gaussian 数增加约 6.3%。
- 4D-Scaffold 只有 +0.11 dB PSNR；LPIPS 和 SSIM 都变差，Gaussian 数减少约 4.6%。
- 因此 TruncGradGS 在 N3DV 上的**PSNR 对三个 backbone 都提高**，但“quality consistent improvement”若指全部指标则不成立；效果高度依赖原方法的 optimization/densification design。

### 新动态 benchmark：强依赖 backbone 与初始化密度

| Method | LPIPS ↓ | SSIM ↑ | PSNR ↑ | #GS ↓ |
| --- | ---: | ---: | ---: | ---: |
| 4DGS sparse $\phi$ | 0.2326 | 0.8329 | 25.47 | 2.409M |
| **4DGS sparse $\phi$ + Ours** | **0.2035** | **0.8574** | **26.08** | **1.715M** |
| 4DGS dense $\theta$ | 0.2288 | 0.8224 | 26.02 | 2.970M |
| **4DGS dense $\theta$ + Ours** | **0.1496** | **0.8923** | **27.56** | **2.333M** |
| CEM-4DGS | **0.2767** | **0.7855** | **22.80** | **0.704M** |
| CEM-4DGS + Ours | 0.2670 | 0.7845 | 22.63 | 0.824M |

- Sparse 4DGS：+0.61 dB，LPIPS 降约 **12.5%**，Gaussian 数减少约 28.8%。
- Dense 4DGS：+1.54 dB，LPIPS 降约 **34.6%**，Gaussian 数减少约 21.4%；它在 Table 4 的 LPIPS/SSIM/PSNR 三个**质量指标最佳**，但 Gaussian 数并非最少。
- CEM-4DGS：LPIPS 改善约 3.5%，但 PSNR **下降 0.17 dB**、SSIM 略降、Gaussian 数增加约 **17.0%**。
- 新 benchmark 没有报告 4D-Scaffold + TruncGradGS，也没有 per-scene 表或方差；无法知道收益集中在哪些 motion/atmospheric effects。

### 消融：surrogate 必要，但完整组合并非所有指标最优

Table 5 同时报告 `Alley` 与 `Tree`，尽管正文称“all ablations on the alley scene”，存在表文不一致。

| 配置 | Alley LPIPS / SSIM / PSNR | Tree LPIPS / SSIM / PSNR |
| --- | ---: | ---: |
| w/o truncated gradient | 0.1350 / 0.8891 / 28.82 | 0.1789 / 0.8681 / 28.03 |
| w/o radius padding | **0.1289** / 0.8913 / 28.83 | 0.1772 / 0.8694 / 28.42 |
| w/o delayed pruning | 0.1309 / 0.8909 / 28.82 | 0.1707 / **0.8753** / **28.79** |
| w/o dead-Gaussian filtering | 0.3716 / 0.6153 / 21.87 | 0.1755 / 0.8728 / 28.67 |
| w/o negative-alpha filtering | 0.1291 / **0.8922** / **29.14** | 0.1701 / 0.8733 / 28.70 |
| **Full method** | **0.1289** / 0.8887 / 29.11 | **0.1685** / 0.8714 / 28.58 |

- 去掉 truncated gradient 后两场景 PSNR 分别降 0.29/0.55 dB，说明 surrogate 有实际贡献。
- `w/o dead-Gaussian filtering` 在 Alley 崩到 21.87 dB，直接证明**不能把 biased tail 用于所有 Gaussians**；保护条件不是工程小优化，而是稳定性的核心。论文文字称“removing truncated gradient causes the largest performance drop”，也被该表自己的 21.87 dB 反例直接否定。
- full method 的 LPIPS 在 Alley 并列最好、Tree 最好，但并非 PSNR/SSIM 最优：Alley 去掉 negative-alpha filtering 反而高 0.03 dB；Tree 去掉 delayed pruning 高 0.21 dB、SSIM 也更高。
- 消融没有扫描最核心的 slope $m$、cutoff $\tau$、padding radius、interleaving ratio 或 dead threshold；也没有报告每个组件的 runtime。
- 单组件 removal 会改变其它 guard 的有效样本集合，存在强 interaction；不能把分数差简单解释为独立贡献。

### 新 benchmark 本身的证据与问题

论文将六个 Blender scenes 作为独立贡献，覆盖 fluid、atmospherics、particles、caustics、complex geometry，并提供 25–50（正文另写 25–45）相机、1–4 test views。它确实比只含简单人类动作的 benchmark 更有视觉复杂度。

但 Table 1 有两个明显问题：

1. 表注定义 `Realistic` 为“photorealistic real-world captures versus synthetic data”，作者数据明确是 Blender synthetic，却标记为 ✓；按自己的定义应是 ×，除非想表达“photorealistic-looking”。
2. `Long duration` 定义为“typically exceeding several hundred frames”，作者数据恰为 300 frames / 10 秒，并未明显超过 several hundred；把它称为 industrial long-duration evidence 较勉强。

此外，当前数据未公开、没有正式名称、license、camera calibration 格式、render settings 或 baseline scripts；在资源发布前，它还不是可复用 benchmark。

### 关键发现

1. Gaussian gradient 的指数尾部与 tile/contribution culling 共同限制 primitive 的探索范围；只扩大 radius 并不能解决 derivative 本身近零。
2. 对 dead Gaussians 使用连续线性 tail，在差初始化和 vanilla 4DGS 上收益最大：随机设置最高 +1.90 dB，N3DV 4DGS +1.48 dB。
3. 方法不是普遍无条件增益：COLMAP Mip360 的 LPIPS/SSIM、N3DV CEM/4D-Scaffold 的部分指标、新 benchmark CEM 的 PSNR/SSIM 都变差。
4. dead filtering、gradient-sign filtering、delayed pruning 和 normal/ADC interleaving 用来约束 biased gradient；直接全局应用会严重不稳定。
5. 推理零额外成本，但训练需处理更多 Gaussian–pixel pairs，代价为 1–10 倍，质量收益必须与 per-scene optimization budget 一起评估。
6. 新 synthetic benchmark 的 motion variety 有价值，但元数据、公开资源和“realistic/long-duration”定义仍需修正。

## 亮点与洞察

### 论文亮点

- **直接修改问题源头**：多数工作围绕 initialization、densification 或 pruning 做补救，TruncGradGS 直接改造导致 primitive 无法移动的 backward field。
- **Forward/backward 解耦清晰**：真实 Gaussian 继续定义图像，surrogate 只负责训练探索；因此不会给 inference 增加参数、网络或计算。
- **兼容多种 representation**：同一思想测试了 3DGS、2DGS、4DGS、CEM-4DGS、4D-Scaffold，说明它不是某个 motion model 的专用模块。
- **对失败风险有明确防护**：dead-only、pull-only、delayed pruning、interleaved ADC 都针对 surrogate bias 的具体故障，而非任意堆叠技巧。
- **同时报告 Gaussian count 与 training time**：结果没有隐藏计算代价，能看到更少 primitives 往往来自更昂贵训练。

### 我的洞察

- **个人分析：它改变的是 optimization geometry，而不是 representation capacity。** Forward model 完全相同，提升来自把原本互相不可达的 basin 用人工 gradient corridor 连接起来。
- **个人分析：dead Gaussian 像一组可迁移的备用粒子。** 标准 3DGS 会删掉它们；TruncGradGS 暂时保留并让远端 residual 牵引，类似在固定表示中回收失败容量，而不是立即 densify 新点。
- **个人分析：surrogate gradient 并不对应论文 forward loss 的精确导数。** 这接近 straight-through estimator 或 synthetic gradient：实用性可由实验验证，但不能从 photometric objective 单调下降自然推出收敛。
- **个人分析：最大收益应出现在 coverage gap，而非所有场景。** Random init、first-frame dynamic init 和大 displacement 收益大；高质量 COLMAP + 成熟 compact backbone 收益小甚至伤害 perceptual metrics，正符合这一判断。
- **个人分析：训练倍率取决于 pair expansion，不只取决于 Gaussian 数。** 最终模型更小不代表训练更省；一个 dead Gaussian 若覆盖大量 tiles，backward workload 可远大于多个局部 Gaussians。
- **个人分析：它与 densification 是互补又竞争的。** Surrogate 让旧 primitive 移动到 error region，densification 在 error region 创建新 primitive。前者可能减少数量，后者可能更快；需要在相同 wall-clock 下比较，而非只固定 iterations。

## 局限与展望

### 作者承认的局限

- 扩大的 optimization support 让更多 Gaussian–pixel pairs 进入 backward，因此训练更慢；方法在不同 scene 上可慢 1–10 倍。
- 除 delayed pruning 外，作者没有为 TruncGradGS 重新调 baseline densification/pruning hyperparameters；与其联合设计可能进一步改善，特别是 SSIM。
- surrogate 若用于 active/good-fit Gaussians会发生 divergence，因此当前只用于 dead Gaussians。

### 独立分析

- **缺少正式收敛保证**：章节名为 convergence analysis，但没有 theorem、bound 或 surrogate objective；保护条件来自经验观察。
- **关键超参数不可复现**：未报告 $m$、radius padding、radius threshold、$\tau_{pruning}$、phase lengths、interleaving schedule 和 motion-mask 参数。
- **正文所引 supplement 不在当前 PDF**：完整 quadratic derivation 与 overhead details 无法核对；代码又尚未发布。
- **符号与公式存在错误/歧义**：Eq. 1 的 transmittance product 使用 $\sigma_i\alpha_j$ 而非通常的 $\sigma_j\alpha_j$；Eq. 6–7 的 conic coefficient 与 Eq. 8 不一致；line intercept 与 conic coefficient 都记作 $b$；Figure 4 的 B/C/D 描述也有重复/错位迹象。
- **threshold 语义混合**：density $\alpha<\tau$、contribution $\sigma\alpha<1/255$ 与 pruning opacity threshold 是三种不同 gate，正文没有给出完整判定优先级。
- **只改 projected mean 是否足够不清楚**：差初始化还可能需要 scale、rotation、opacity 或 color 调整；implementation 对低-opacity case 的 opacity gradient 描述与“只研究 mean gradient”不完全一致。
- **动态加速依赖 motion mask**：frame difference 会受照明、阴影和噪声影响；没有 mask quality 或 runtime 消融，也未说明是否只用 training views。
- **“consistent improvement”措辞过强**：多个 LPIPS/SSIM 变差，新 benchmark 的 CEM PSNR 也下降；更准确的结论是 PSNR 多数/几乎全部设置提高，perceptual/structural metrics mixed。
- **缺少 wall-clock 公平比较**：baseline 训练更快。若允许 baseline 用相同 55/86 分钟继续训练或更频繁 densify，收益是否保留未知。
- **没有统一 Gaussian budget**：TruncGradGS 经常产生更少 Gaussians，但也有数量增加的设置；质量变化与 capacity 变化没有解耦。
- **缺少 random-seed 与 per-scene 结果**：平均增益中的 0.11–0.22 dB 可能接近 run variance；新 benchmark 也无法定位失败 scene。
- **动态 baseline 覆盖不完整**：新数据只报告 4DGS 与 CEM-4DGS，没有 4D-Scaffold、FreeTimeGS 等；N3DV 与新数据的 baseline set 不统一。
- **新 benchmark 自述矛盾**：synthetic 数据被标为 real-world realistic；300 帧被列为 long duration；相机数写作 25–50 与 25–45 两版。
- **资源尚未发布**：无法验证 CUDA backward、复现新数据、检查 motion masks 或测真实显存/速度。

### 建议的后续实验

1. 固定 wall-clock、peak VRAM 和最大 Gaussian–pixel pair budget，对比 baseline、更多 iterations、radius-only、TruncGradGS 与 MCMC/densification 方法。
2. 对 $m$、$\tau$、padding radius、dead threshold 和 truncated/normal phase ratio做联合扫描，报告质量—时间曲线。
3. 记录 dead Gaussians 的 survival rate、迁移距离、复活后 opacity 和最终贡献，直接验证“救援”机制。
4. 分别修改 mean、scale、opacity gradient，确认真正需要 surrogate 的属性与场景。
5. 用 objective value、gradient cosine similarity 和 step acceptance 分析 surrogate bias；尝试 trust region 或逐步退火 surrogate。
6. 对动态 motion mask 做 precision/recall 与 `w/o mask` 消融，计入 preprocessing 和 I/O。
7. 在相同 Gaussian budget 和同一 densification policy 下复测，隔离 gradient field 与 model capacity 的贡献。
8. 给新 benchmark 发布 per-scene 表、calibration、Blender assets、license、render seeds 和 evaluation scripts，并修正 Realistic/Long duration 标签。
9. 增加 sparse-view、blur、lighting change 和 inaccurate-pose stress tests，判断远程 gradient 会不会把 calibration residual 误当成 geometry displacement。

## 与相关工作的对比

| 方法 | 主要干预位置 | 如何改善 coverage / robustness | 推理代价 | 主要风险 |
| --- | --- | --- | --- | --- |
| RAIN-GS | initialization + progressive training | sparse-large-variance points、low-pass 与 bound-expanding split | 无额外网络 | 仍依赖 densification schedule，非直接修改 gradient tail |
| 3DGS-MCMC | primitive optimization / relocation | 将 densification 解释为 sampling，回收低贡献容量 | 与 3DGS 接近 | stochastic schedule 与超参数复杂 |
| PAPR | point radiance field | proximity attention 与 stochastic point propagation 扩展 point influence | representation 不同 | 不是 tiled Gaussian rasterizer 的直接 backward patch |
| Pixel-GS / ConeGS | densification signal | 用 pixel/error geometry 决定在哪里增加 primitives | 最终 GS renderer | 需要新 primitives，不解决已有 primitive 的远端 derivative |
| **TruncGradGS** | **rasterizer backward** | **dead-only radius padding + continuous linear derivative tail** | **无 inference 代价** | **biased gradient，训练 1–10×，依赖多重 guards** |

TruncGradGS 最独特的地方是把“覆盖不足”视为 gradient field 问题，而不只是 point placement 问题。它理论上可以叠加到 RAIN-GS、MCMC 或 error-guided densification 上，但这些方法都在竞争同一件事：如何把有限 primitive capacity 放到 residual 最大且几何合理的位置。联合使用不一定简单相加，需要控制重复探索与训练成本。

## 启发与关联

- **可微渲染器设计**：高效 forward 中的 culling/threshold 往往会切断 optimization connectivity。可以让 forward 保持稀疏、backward 使用更宽但受控的 support。
- **Straight-through 思路的几何版本**：对不易优化的离散/局部 renderer，可以设计连续 surrogate backward，但必须限定到低置信度或即将淘汰的参数。
- **固定预算 Gaussian recycling**：把 dead primitives 当作可移动 reserve，比 clone/split 后持续增大模型更适合显存受限训练。
- **动态场景初始化**：first-frame point cloud 到远期 geometry 的距离很大，局部 gradient 难以跨越；TruncGradGS 可与 FreeTimeGS 的局部时空出生或 ATGS 的 temporal anchors 结合，减少必须跨越的距离后再做 residual rescue。
- **假设：uncertainty-gated surrogate**。不只看 opacity，还结合 multi-view gradient agreement、depth uncertainty 和 reprojection consistency；仅当远端 pixels 在多个视角指向相似方向时移动 Gaussian，可减少错误长程 pull。
- **假设：adaptive radius budget**。给每个 training step 固定 extra pair budget，按 dead age、residual magnitude 和 gradient agreement 分配 radius，避免最坏 10 倍 slowdown。
- **假设：annealed tail**。训练早期使用平缓长程 surrogate，Gaussian 接近目标后逐步增大 slope 或缩小 radius，最终恢复真实 gradient，可能减少 biased optimum。

## 评分

| 维度 | 评分（10 分） | 理由 |
| --- | ---: | --- |
| 创新性 | 8.3 | 直接修改 tiled Gaussian rasterizer 的 derivative tail，而不是再设计 initialization/densification；forward/backward 解耦切入点新颖。 |
| 技术可靠性 | 7.0 | 梯度消失分析和 dead-only guard 有说服力，跨多 backbone 有结果；但 surrogate 无收敛保证、公式/threshold 歧义且代码缺失。 |
| 实验充分度 | 7.3 | 覆盖 static/dynamic、random/COLMAP、3DGS/2DGS/三种动态 backbone，并报告数量和时间；缺 per-scene、seed、wall-clock fairness 和核心超参数扫描。 |
| 写作清晰度 | 6.6 | 问题和主方法容易理解，但多处公式符号、Figure 路径、benchmark 标签、camera 数和“consistent”结论不严谨。 |
| 实用 / 研究价值 | 7.8 | 插件式 backward、推理零成本，对差初始化和动态大位移有价值；1–10 倍训练成本和未公开实现限制当前落地。 |

**总体推荐：值得细读。** 对 3DGS optimization、differentiable rasterization 或 dynamic reconstruction 研究者，最值得学习的是“保持 forward 稀疏，在 backward 为失败 primitives 构造受控长程通道”；但论文当前更像有潜力的 optimization primitive，而非已充分复现和调优的通用方案。

## 阅读结论

- **最值得记住的点**：扩大 tile radius 只解决 dispatch，不能解决 Gaussian derivative 的指数衰减；TruncGradGS 同时修改离散 coverage 与连续 gradient tail。
- **最需要怀疑的点**：surrogate 是 biased update，效果依赖 dead/sign/schedule guards；论文缺代码和关键参数，且“全设置一致改善”被自己的 LPIPS/SSIM/CEM 数据反驳。
- **最值得复现或继续验证的点**：在相同 wall-clock 和 pair budget 下，追踪 dead Gaussians 的迁移/复活过程，并比较 radius-only、linear tail、MCMC relocation 和 error-guided densification。

## 相关论文

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — forward rasterization、tile culling、ADC 与 alpha compositing baseline。
- **PAPR: Proximity Attention Point Rendering** — 论文认定的最近工作；同样处理 point influence/vanishing gradient，但 representation 与 propagation 机制不同。
- **RAIN-GS: Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting** — 从 sparse-large-variance initialization 与 progressive training 提升 random-init robustness。
- **3D Gaussian Splatting as Markov Chain Monte Carlo** — 通过 stochastic relocation/densification 改善 primitive distribution，与“救援 dead Gaussians”目标相近。
- [Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting](https://arxiv.org/abs/2310.10642) — TruncGradGS 动态实验的主要 4DGS backbone。
- [FreeTimeGS: Free Gaussian Primitives at Anytime and Anywhere for Dynamic Scene Reconstruction](https://arxiv.org/abs/2506.05348) — 通过任意时空出生和短程线性 motion 缩短大位移 correspondence，可与长程 gradient rescue 对照。

## 官方资源

- [arXiv v1](https://arxiv.org/abs/2609.03534v1) — 元数据、PDF、HTML 与 TeX source 入口。
- [Official HTML](https://arxiv.org/html/2609.03534v1) — 本笔记的方法公式与 Tables 1–5 来源。
- **代码 / 数据集 / Supplement**：正文承诺 publication 时公开，但截至 2026-09-06 未找到可访问链接；当前 arXiv PDF 也未附正文引用的 supplement。
