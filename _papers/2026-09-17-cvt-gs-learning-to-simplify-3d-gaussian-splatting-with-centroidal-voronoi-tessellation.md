---
title: "CVT-GS: Learning to Simplify 3D Gaussian Splatting with Centroidal Voronoi Tessellation"
subtitle: "以全局 CVT 划分合并单元，并用一次前向预测将每个单元压缩为一颗标准 Gaussian"
authors: "Bingxian Li, Yilong Li, Jingliang Peng, Peng-Shuai Wang, Fei Zhu, Guozheng Li, Chi Harold Liu, Guoping Wang, Bo Pang"
venue: "arXiv preprint"
year: 2026
date: 2026-09-17
paper_url: "https://arxiv.org/abs/2609.08730"
tags:
  - 3D Gaussian Splatting
  - Compression
  - Simplification
  - Centroidal Voronoi Tessellation
  - Post-hoc Optimization
summary: "CVT-GS 是一种面向已训练标准 3DGS 的免场景微调（optimization-free）简化方法：它先以 opacity 加权的 Centroidal Voronoi Tessellation 在全场景建立 M=ceil(ρN) 个空间连贯单元，再以共享的 MergeNet 将每个可变大小单元预测成一颗标准 Gaussian。四个 benchmark、三种压缩率的 immediate-output 对比中，它在质量和端到端简化时间上均优于 LightGS、PUP-3DGS、GHAP、NanoGS；相对最接近的 NanoGS，平均 PSNR 在保留率 0.1/0.01/0.001 时分别提高 1.57/1.30/1.03 dB。"
permalink: /papers/cvt-gs-learning-to-simplify-3d-gaussian-splatting-with-centroidal-voronoi-tessellation/
---

> **阅读依据**：arXiv:2609.08730v1（2026-09-08，12 页，含补充材料）。论文将目标限定为：输入一个已经训练好的、标准格式的 3DGS 模型，直接输出更少的**标准 Gaussian**；它不压缩固定点集的属性比特率，也不做每个场景的恢复训练。PDF 未给出代码或项目链接。

## 一句话总结

CVT-GS 把激进的 Gaussian 删点改写成“全局地分配 $M$ 个空间支持域、每域学习合并为一颗 splat”：CVT 保证合并对象在空间和 opacity 质量上连贯，MergeNet 以局部渲染监督预测残差，从而在 $0.1\%$ 保留率下仍比现有后处理基线保真，并能直接交给原有 3DGS renderer。

## 背景与核心问题

高质量 3DGS 通常有数十万到数百万 primitives，存储、传输、深度排序与渲染都会受其制约。现有压缩途径常见三类：训练时 pruning、改造表示/解码器、或对每个已训练场景再优化。它们不完全适合“手中已有一个标准 3DGS，希望马上得到一个小而仍可用的标准 3DGS”的部署情境。

独立删掉低分 Gaussian 在极低保留率下会留下 coverage holes；局部 pairwise merge 或 k-NN grouping 又不能在全局固定输出预算下协调不同区域的合并尺度。作者的问题是：**给定输出数 $M=\lceil\rho N\rceil$，怎样把原有 Gaussians 划成适合一对多合并的空间单元，并以无需场景微调的方式预测每个单元的代表 splat？**

## 方法详解

### 整体框架

<div class="mermaid">
flowchart LR
    A[预训练标准 3DGS] --> B[过滤低 opacity Gaussian]
    B --> C[opacity 加权 CVT]
    C --> D[M 个空间支持单元]
    D --> E[计算单元参考 Gaussian 与相对特征]
    E --> F[共享 MergeNet 一次前向]
    F --> G[每单元预测一颗标准 Gaussian]
    G --> H[简化后的标准 3DGS]
    I[离线训练：单元局部可微渲染损失] -.监督.-> F
</div>

部署时只走上方实线：网络参数不更新，也没有 per-scene fine-tuning。训练阶段则从原始单元和其一颗预测 Gaussian 渲染同一局部 crop，监督共享 MergeNet。

### 1. 全局 CVT 支持域：决定“哪些 Gaussian 可以一起合并”

先去除 opacity 小于 $\tau_\alpha=0.1$ 的 Gaussian；但目标点数仍按过滤前的 $N$ 计算，即 $M=\lceil\rho N\rceil$。将剩余中心以 opacity 作为质量，构成离散测度：

$$
\nu_S=\sum_{i\in\mathcal I}\alpha_i\delta_{\mu_i}.
$$

对 $M$ 个 sites $C=\{c_k\}_{k=1}^{M}$，CVT 最小化 opacity 加权的单元内平方距离：

$$
E_{\mathrm{CVT}}=
\sum_{k=1}^{M}\sum_{i\in V_k}
\alpha_i\lVert\mu_i-c_k\rVert_2^2.
$$

其中 $V_k$ 是离 $c_k$ 最近的 Gaussian 中心集合。固定归属后，site 有闭式的 opacity 加权质心更新：

$$
c_k=\frac{\sum_{i\in V_k}\alpha_i\mu_i}
{\sum_{i\in V_k}\alpha_i}.
$$

交替执行最近 site 归属和该更新，就是 weighted Lloyd relaxation；本文使用 Geogram 跑 5 次。直觉上，高 opacity 或稠密位置会获得更细的单元，稀疏区获得较大的单元。相比独立 k-NN，这个目标把所有 $M$ 个输出绑定在同一全局量化误差中，因此不容易把一个 cell 跨越不相干表面。

### 2. 参考 primitive：把可变大小单元标准化

一个 CVT cell 内成员数不固定。作者先由成员的归一化 opacity $\omega_i=\alpha_i/\sum_{j\in V_k}\alpha_j$ 构造确定性的参考 primitive $\bar G_k$：位置和 SH appearance 是加权均值，opacity 使用概率合成 $\bar\alpha_k=1-\prod_{i\in V_k}(1-\alpha_i)$。其 covariance 是 mixture 的二阶矩：

$$
\bar\Sigma_k=
\sum_{i\in V_k}\omega_i\Sigma_i+
\sum_{i\in V_k}\omega_i(\mu_i-\bar\mu_k)(\mu_i-\bar\mu_k)^\top.
$$

第一项保留各 splat 自身形状，第二项记录成员中心的离散程度；对它特征分解即可取出参考 scale 和 rotation。随后每个成员编码为相对参考的位置、log-scale 差、相对 quaternion、$(\alpha_i,\omega_i)$ 与 appearance residual。这使同一网络能处理不同场景和不同压缩倍率的单元，而不被绝对尺度绑死。

### 3. MergeNet：不是选择一个成员，而是预测新 Gaussian

MergeNet 是 permutation-invariant set predictor。它先用点级 MLP 嵌入单元成员，再同时做 mean pooling、max pooling 与 opacity-weighted pooling，最后由 decoder 输出相对于 $\bar G_k$ 的残差：

$$
\Delta G_k=\psi(h_k),\qquad G_k^\star=\bar G_k\oplus\Delta G_k.
$$

位置在参考局部坐标预测；scale 与 opacity 分别在 log-scale、logit-opacity 域更新；rotation 以 quaternion 增量组合，appearance 则加性更新。三个 pooling 分别保留平均统计、显著成员与视觉贡献大的成员。重要的是，输出并非输入集合的子集，而是一颗新预测、但仍完全符合原始 3DGS 参数格式的 primitive。

### 4. 离线局部渲染监督

网络只训练一次：对 cell $V_k$，将原 cell 与其一颗预测 $G_k^\star$ 在相同相机/局部 crop 下分别渲染。目标是标准的 $L_1+\mathrm{SSIM}$ 损失再加梯度一致性：

$$
L_{\mathrm{grad}}=
\lVert\nabla_x\hat I_{k,\pi}-\nabla_x I_{k,\pi}\rVert_1+
\lVert\nabla_y\hat I_{k,\pi}-\nabla_y I_{k,\pi}\rVert_1,
$$

$$
L_{\mathrm{train}}=(1-\lambda)L_1+\lambda L_{\mathrm{SSIM}}+\lambda_gL_{\mathrm{grad}}.
$$

实验用 $\lambda=0.2,\lambda_g=0.2$，在 LLFF、ShapeNet、ScanNet 构建的 3DGS cell 上训练 30k steps；同一 checkpoint 用于所有测试场景和 $\rho$，所以“免优化”是指测试场景没有再训练，而不是 MergeNet 从未训练。

## 实验证据

### 设置

在 NeRF-Synthetic、Mip-NeRF360、Tanks & Temples、Deep Blending 上测试 $\rho\in\{0.1,0.01,0.001\}$。所有方法从同一官方 3DGS 输入开始，统一目标点数、renderer、evaluator、硬件，并遵循 immediate-output protocol：删减后立刻评估，不允许方法专属恢复或 fine-tuning。比较 LightGS、PUP-3DGS、GHAP、NanoGS；指标为 PSNR/SSIM（高更好）、LPIPS（低更好）及端到端简化时间。硬件为单张 RTX 4090 24GB，时间从加载模型到写出结果。

### 主实验：极端点数削减仍保住相对质量

下表摘录论文 Table 1 中最相关的 post-hoc 基线 NanoGS，以及 CVT-GS。每格为 `PSNR / SSIM / LPIPS；时间秒`。

| 数据集 | 保留率 | NanoGS | CVT-GS |
| --- | ---: | --- | --- |
| NeRF-Synthetic | $0.1$ | 25.81 / 0.910 / 0.092；11.36 | **27.76 / 0.936 / 0.079；3.58** |
| Mip-NeRF360 | $0.01$ | 19.39 / 0.470 / 0.587；158.05 | **20.57 / 0.523 / 0.532；9.01** |
| Tanks & Temples | $0.001$ | 13.54 / 0.457 / 0.641；62.33 | **14.50 / 0.471 / 0.625；5.71** |
| Deep Blending | $0.001$ | 19.42 / 0.739 / 0.507；99.63 | **20.52 / 0.765 / 0.484；6.97** |

论文报告，跨四数据集相对 NanoGS 的平均 PSNR 增益在 $\rho=0.1/0.01/0.001$ 为 +1.57/+1.30/+1.03 dB，SSIM 为 +0.055/+0.034/+0.025。$\rho=0.01$ 即 100 倍简化时，CVT-GS 平均 PSNR 为 21.32 dB；相对 NanoGS 的 +1.30 dB 与摘要相符。注意这不代表其压缩后绝对质量接近原模型：例如 Mip-NeRF360 原始 3DGS 是 27.43/0.813/0.221，$\rho=0.001$ 的 CVT-GS 已降至 18.11/0.466/0.626。

### 效率

汇总所有场景和倍率，作者称 CVT-GS 比 NanoGS 快 12.3 倍、比 GHAP 快 3.1 倍；补充对比称相对 LightGS/PUP-3DGS 为 7.2/7.0 倍。原因是 CVT 建一次后，MergeNet 仅做一遍前向，而基线存在较重的合并、选择或优化过程。这个结论在该论文的“输出立刻可用”协议内有效；并未报告简化后 render FPS、磁盘字节数或峰值 VRAM。

### 消融：CVT 与 learned merger 缺一不可

四个 benchmark 的平均结果（论文 Table 2）：

| 变体 | $\rho=0.1$ PSNR | $\rho=0.01$ PSNR | $\rho=0.001$ PSNR / LPIPS |
| --- | ---: | ---: | ---: |
| w/o CVT（local k-NN） | 23.69 | 19.67 | 15.80 / 0.677 |
| w/o opacity filtering | 23.73 | 20.56 | 17.47 / 0.530 |
| w/o MergeNet（仅参考统计） | 22.62 | 18.31 | 14.10 / 0.811 |
| **完整 CVT-GS** | **24.57** | **21.32** | **18.33 / 0.481** |

- 去掉 CVT 后，三个倍率平均 PSNR 分别下降 0.88、1.65、2.53 dB；倍率越激进，全局 cell allocation 越关键。
- 去掉 MergeNet 的降幅最大（1.95、3.01、4.23 dB），说明二阶矩构造的 reference 只能给合理初始化，无法独自概括 cell 内的外观和几何残差。
- opacity filtering 的收益较小但稳定，平均提升 0.76–0.86 dB；它也与“用 opacity 表示 spatial mass”的 CVT 设定一致。

## 亮点与局限

**论文亮点**：

- 清楚地区分 primitive-count reduction 与属性编码；输出保留标准 3DGS 格式，兼容性强。
- CVT 以全局、固定预算的量化目标替代独立局部邻域，正中激进 many-to-one merging 的几何难点。
- 参考 Gaussian + 相对描述子让 shared network 可跨场景、跨倍率使用；测试阶段不做场景优化。
- 比较协议对所有方法禁用恢复训练，质量—速度的比较较公平。

**作者明确的后续方向**：更强的 MergeNet，以及把时间维度纳入动态 4D 表示。

**独立分析**：

- “optimization-free”容易被误读：测试免微调成立，但 MergeNet 依赖 LLFF、ShapeNet、ScanNet 上的离线训练。跨训练分布的泛化虽由四 benchmark 间接支持，却没有 dedicated out-of-domain 或 zero-shot failure 分析。
- CVT 只在中心位置与 opacity 上建模，忽略 covariance 朝向、遮挡和 view-dependent SH。两个相邻但属于不同表面的 Gaussians 仍可能落入同一个 cell，后续只能由一颗 Gaussian 勉强表示；论文没有针对这类边界的失败案例或 anisotropic metric 消融。
- 固定 $M=\lceil\rho N\rceil$，且先过滤低 opacity 后仍使用原始 $N$ 计预算：这使不同方法的数量一致，但在极低 $\rho$ 下可能让可用 cell 数与过滤后有效点数不匹配。应报告过滤比例和空/极小 cell 的处理。
- 表中报告的是 primitive 数和时间，而非真正的 rate-distortion：SH 阶数、数值精度、序列化格式与最终文件大小未报告。若部署目标是传输，应结合量化/熵编码，并测量字节数、加载和 render FPS。

## 与相关工作对比

| 方法 | 简化机制 | 测试时场景优化 | 输出 / 取舍 |
| --- | --- | --- | --- |
| LightGaussian | pruning 与压缩 | 依方法而定 | 关注高效 3DGS，非一对多 cell 预测 |
| PUP-3DGS | 不确定性 pruning | 无 | 直接删除不确定/冗余 primitives，极低倍率易有 holes |
| GHAP | optimal-transport 全局 reduction | 无 | 同为全局 reduction，但 CVT-GS 以 CVT cell 与 learned merger 显式生成代表 splat |
| NanoGS | training-free local pairwise merge | 无 | 最近的后处理对照；局部 merge 较快但不协调全局支持域 |
| **CVT-GS** | opacity-weighted CVT + MergeNet many-to-one merge | **无** | 标准格式、一次前向；有离线预训练依赖 |

## 评分

| 维度 | /10 | 依据 |
| --- | ---: | --- |
| 创新性 | 8.2 | 将经典 CVT 全局量化与 learned Gaussian merger 组合为明确的 post-hoc 简化范式。 |
| 技术可靠性 | 7.8 | 目标、参考参数化与消融逻辑自洽；几何 metric 未纳入可见性/方向仍是风险。 |
| 实验充分度 | 7.7 | 四 benchmark、三压缩率、统一 immediate-output、逐项消融齐全；缺 VRAM/FPS/文件大小与 OOD 分析。 |
| 写作清晰度 | 8.0 | CVT 目标、reference construction 和训练/部署边界说明清楚。 |
| 实用/研究价值 | 8.4 | 可直接处理已有标准 3DGS，并保持 renderer 兼容，适合批量部署前的点数缩减。 |

**总体推荐：值得细读。**它最有价值的不是单纯“学一个压缩器”，而是把复杂场景中的 many-to-one merge 分解为全局空间分配与局部外观拟合两个职责明确的步骤。

## 阅读结论

- **最值得记住的点**：极端简化时，先全局决定每个输出 Gaussian 的空间支持域，再预测合并结果，比先局部删/并更稳健。
- **最需要怀疑的点**：仅用位置与 opacity 的 CVT 是否足以尊重表面边界、遮挡和 view-dependent appearance；论文尚未给出直接证据。
- **最值得复现或继续验证的点**：以 covariance/normal/visibility 构造 anisotropic CVT，并补充字节级 rate、FPS、VRAM 与跨域失败分析。

## 相关论文

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 标准表示与 rasterizer 基线。
- [NanoGS: Training-Free Gaussian Splat Simplification](https://arxiv.org/abs/2603.16103) — 本文最接近的 training-free post-hoc local merging 对照。
- [LightGaussian](https://arxiv.org/abs/2311.17245) — 3DGS pruning/压缩的代表性路径。
- [PUP 3D-GS](https://arxiv.org/abs/2408.04137) — 基于不确定性的 pruning 对照。
- [CVT: Applications and Algorithms](https://epubs.siam.org/doi/10.1137/S0036144599352836) — 本文空间支持域构造的几何基础。
