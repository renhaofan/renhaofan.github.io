---
title: "Transforming harmonic coefficients for 3D splat compression"
subtitle: "以渲染误差诱导 Gram 内积正交化球谐系数，再以 KLT 去相关，使量化误差对齐图像误差"
authors: "Tam Thuc Do, Philip A. Chou, Gene Cheung"
venue: "arXiv preprint"
year: 2026
date: 2026-09-16
paper_url: "https://arxiv.org/abs/2609.15735"
tags:
  - 3D Splat Compression
  - 3D Gaussian Splatting
  - Spherical Harmonics
  - Rate-Distortion Optimization
  - KLT
  - RAHT
summary: "本文固定 3D splat 几何，只压缩方向相关颜色的球谐（SH）系数。作者将渲染写成系数向量的线性映射，并用由训练/评价视角诱导的 Gram 矩阵平方根把系数变为正交基，使变换域的 L2 量化误差严格等于图像平方误差；之后接单位正交的 KLT 降低独立熵编码的码率。基于 Sparse Voxel Splats、RAHT、均匀标量量化与 ANS 的无再训练流水线，在 Mip-NeRF 360 与 NeRF Synthetic 的 RD 曲线上报告至少 1 dB 的方向—颜色 KLT 增益，配合能量保持变换总计超过 2 dB。"
permalink: /papers/transforming-harmonic-coefficients-3d-splat-compression/
---

> **阅读依据**：本文基于 [arXiv:2609.15735v1](https://arxiv.org/abs/2609.15735)（2026-09-14 提交）的完整 LaTeX 源码、公式、图和实验表格。尚未见同行评审 venue、官方代码或项目页，因此以下结论仅对应该预印本版本；作者公开的是论文源码而非独立实现仓库。

## 一句话总结

Transforming harmonic coefficients for 3D splat compression 不把每个 SH 系数视为等价的欧氏坐标，而是依据其在给定相机集合中对像素的实际贡献构造 Gram 度量：先以 \((\Phi^\top\Phi)^{1/2}\) 正交化，使量化噪声的系数域能量恰等于渲染图像的平方误差；再以 KLT 装箱能量并降低逐系数熵编码的码率，从而以很小的变换开销提升 RD 性能。

## 背景与问题

一个 splat 常有 16 个 SH basis、RGB 三个颜色通道，即每个 primitive 48 个颜色属性；场景含数百万 splats 时，颜色属性常比几何占更多 bit。现有 VQ、标量量化或熵模型通常按 SH 系数空间的 Euclidean distance 选码字，但同样大小的 \(\ell_2\) 系数误差，对最终图像的影响并不相同：可见性、opacity、空间覆盖范围、观察方向都会改变某一系数的渲染权重。

因此本文不是再设计一个 learned codec，而是提出一个更基础的问题：**怎样改变系数坐标，使普通均匀标量量化所最小化的误差，正好就是真实的图像平方误差？** 其作用范围刻意限定为已确定的 geometry 后的 color attributes，故可与 pruning、空间变换或学习式 entropy coding 叠加。

## 方法详解

### 总体流程

```mermaid
flowchart LR
    A[已训练且几何固定的 3D splats<br/>SH × RGB coefficients] --> B[从评价视角采样 ray directions]
    B --> C[方向 SH Gram matrix<br/>Phi_d^T Phi_d]
    C --> D[能量保持变换 T_ep^d = sqrt(Gram)]
    A --> D
    D --> E[方向—颜色协方差]
    E --> F[单位正交 KLT: T_ec^(d,c)]
    F --> G[RAHT 空间变换]
    G --> H[均匀标量量化 + ANS]
    H --> I[bitstream]
    I --> J[逆变换并重建 splats]
    J --> K[渲染测试视图]
```

实验使用 Sparse Voxel Splats 而非 Gaussian，但理论面向任意固定几何的 splat（Gaussian、voxel、B-spline、local neural function 等）。记 splat 数为 \(N\)、方向 basis 数为 \(M\)（本文取 16）、颜色 basis 数为 \(L\)（RGB 时为 3），颜色系数向量 \(c\) 的长度为 \(NML\)。

### 1. 固定几何后，渲染对颜色 SH 系数线性

沿一条 ray \((o,d)\)，第 \(n\) 个 splat 的终止权重为 \(\phi_n^s(o,d)=T_n\alpha_n\)，它由 geometry/density/遮挡顺序决定。将方向 radiance 展开到 SH \(\phi_m^d(d)\)、颜色展开到 RGB basis \(\phi_l^c(\lambda)\)，则

\[
f(o,d,\lambda)=\sum_{n,m,l}
\phi_n^s(o,d)\phi_m^d(d)\phi_l^c(\lambda)c_{nml}
=\Phi c.
\]

这一步的含义很关键：geometry 参数决定 \(\Phi\)，确实是非线性的；但**只要 geometry 锁定**，颜色系数进入任何可渲染 plenoptic image 的方式是线性的。于是属性编码可以精确讨论为线性变换下的 rate–distortion 问题，而非以系数值大小代替图像质量。

### 2. 用诱导 Gram 矩阵把“系数误差”校准为“图像误差”

令目标与解码图像的平方误差为 \(D(f^*,\hat f)\)。若颜色 basis 正交，论文使用评价 rays 的计数测度定义内积，得到

\[
D(f^*,\hat f)=(c^*-\hat c)^\top \Phi^\top\Phi(c^*-\hat c).
\]

令

\[
T_{ep}=(\Phi^\top\Phi)^{1/2}, \qquad \bar c=T_{ep}c,
\]

则

\[
D(f^*,\hat f)=\|\bar c^*-\hat{\bar c}\|_2^2.
\]

这不是经验 loss，而是代数恒等式：\(T_{ep}\) 把原先彼此不等重要、也非正交的 rendered basis 变为正交 basis。因而在高码率、均匀标量量化和独立最优熵编码的假设下，变换后采用等步长量化对应最优的立方量化单元；直接在原 SH 坐标量化则会把 bit 花在图像里不等权的方向上。

### 3. KLT 只做去相关，不破坏失真对齐

能量保持不等于低码率：若各 transformed coefficients 仍相关，factorized entropy model 会浪费 bit。作者在 \(T_{ep}\) 后对方向—颜色 \(ML\times ML=48\) 维向量估计协方差，取 KLT \(T_{ec}\)：

\[
\tilde c=T_{ec}T_{ep}c.
\]

KLT 是 unitary，故不会改变上式的 \(L_2\) 能量；它的作用是尽量把 variance 集中到少数分量，降低逐分量独立熵编码的交叉熵。论文也计算了不做正交化（\(T_{ep}^d=I\)）直接 KLT 的版本，用来区分“去相关”与“先让 distortion metric 正确”两种收益。

### 4. 可实现的张量近似与 codec

完整 \(NML\times NML\) Gram 矩阵不可存算。论文用空间、方向、颜色三个轴的 Kronecker 近似：

\[
T_{ep}\approx T_{ep}^s\otimes T_{ep}^d\otimes T_{ep}^c.
\]

本实验聚焦方向轴：从所有用于失真测量的 views 收集每个 voxel 的可见方向，计算 \(16\times16\) 的 \(\Phi^{d\top}\Phi^d\) 并开方。空间轴采用 RAHT（而非 learned transform）；编码顺序为 occupancy octree → RAHT（density 与颜色）→ uniform quantization → ANS。KLT、量化步长等 side information 也写入 bitstream header；所有指标均先解码 bitstream、重建 splat model，再渲染测试视图。

## 实验证据

### 设置

作者在 Mip-NeRF 360 与 NeRF Synthetic 上，以两种 Sparse Voxel Splats 模型规模（<10M、<5M splats）评估 RD 曲线。density 的量化步长固定为几乎不损害测试渲染的值，只扫描方向颜色系数的量化步长。评估 PSNR/SSIM（越高越好）和 LPIPS（越低越好），但主图给出 average RGB PSNR 对 bitstream MB。

### 最有力的结果

论文 Fig. 1(a) 比较原 SH 系数、仅方向—颜色 KLT（\(T_{ep}^d=I\)）以及 Gram 正交化后 KLT。作者报告：

- 在两套数据、两种 splat 规模与整个所示 RD 区间，directional-color KLT 相比不变换至少提升约 **1 dB**；
- 加上方向能量保持变换后，整体获得 **超过 2 dB** 的 PSNR 增益，且无需重新训练或 finetune；
- 将模型由 <10M 调为 <5M splats，RD 曲线约再移动 **1 dB**，这属于 splat learning / model-size 选择的收益，不能归因于本文 SH transform。

图中可读的 Mip-NeRF 360 <10M 例子：在约 28 MB 附近，未变换系数约 26.0 dB，仅 KLT 约 26.0 dB，而正交化+KLT 约 26.1 dB；在低码率段（约 13–19 MB）曲线分离更明显，完整变换可相对原系数高约 1–2 dB。图是平均 RD 曲线而非逐场景数字表，故不将读图近似值伪装成精确 benchmark 数字。

作为端到端 pipeline 的横向参照，论文 Table 1 只列 Mip-NeRF 360 单点：

| 方法 | 模型/码流大小 (MB) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| 原始 3DGS | ~700 | 27.45 | 0.815 | 0.237 |
| Compact-3DGS | 15.41 | 24.95 | 0.720 | 0.342 |
| Meson-C3 | 28.33 | 25.96 | 0.769 | 0.266 |
| CompGS | 21.90 | **27.08** | **0.802** | **0.241** |
| SVComp (<10M) | 28.12 | 26.10 | 0.7461 | 0.290 |
| SVComp (<5M) | 16.94 | 25.62 | 0.7187 | 0.323 |

这里不能得出“本文端到端优于 CompGS”：SVComp 是不同的 Sparse Voxel 表示，所列单点 PSNR/感知指标更低。本文要证明的是在**相同 voxel splat 表示**内，SH transform 的 RD 改善；它自己也承认 sparse voxel fidelity 往往以更多存储为代价。

## 亮点、局限与可复现实验

**论文亮点**：

- 给出“固定 geometry 的 splat 渲染对颜色属性线性”的明确形式，因此能把 image-space distortion 精确拉回系数空间；
- Gram 平方根提供可验证的 distortion-aligned transform，而 KLT 保持该度量不变又改善熵编码，是清晰的职责分离；
- 与 pruning、RAHT、VQ、context/latent entropy coding 正交，可作为这些方案的前置线性变换；
- 全链路 bitstream 解码后再渲染，未把未编码 latent 的结果当压缩指标。

**作者限制 / 独立分析**：

- 精确 Gram 的维度是 \(NML\)，实际以三轴张量积近似；论文明确指出该近似源自以 rays 与 directions 的 product measure 替代真实的 joint measure。因而“图像误差严格等价”对完整 Gram 成立，对其可实现近似只近似成立。
- \(T_{ep}^d\) 依赖被纳入 distortion measurement 的相机方向分布。若部署视角分布显著改变，得到的是 training/eval-view 最优权重，不必然是自由浏览视角最优权重。
- 高分辨率均匀量化理论依赖高码率、局部密度近似常数、独立熵编码；极低码率或强非高斯系数下，KLT 的实际收益需要更多 ablation 支撑。
- 目前只有两套 benchmark 的平均曲线、一个 Mip-NeRF 360 横表，尚无逐场景 RD 数字、复杂度/编码时间、header 开销比例、seed 方差或真实 3DGS 的直接 transform ablation；作为 2026-09 的 v1，实验结论仍需复现。

建议复现时固定一个 geometry 与 test camera set，分别跑 `no SH transform`、`KLT only`、`T_ep + KLT`，并额外以未见相机集重算 PSNR；再将同一变换插入 CompGS/NSVQ-GS/entropy-model codec，才能分辨 transform 对 learned codec 的净增益。

## 与相关工作的对比

| 方法类别 | 主要压缩杠杆 | 是否让系数距离对齐 image error | 与本文关系 |
| --- | --- | --- | --- |
| VQ / ECVQ / NSVQ-GS | 码本、量化与熵模型 | 通常否，按系数 Euclidean 距离选码字 | 可在 VQ 前加入 \(T_{ep}\) |
| CompGS / Compact-3DGS / SizeGS | pruning + 属性压缩 | 非核心目标 | 本文固定 geometry，可叠加 |
| RAHT / RAHLE | 空间相关性 | 主要处理 splat index 轴 | 本文补充方向 SH 与颜色轴 |
| HAC / ContextGS / FCGS | context / latent entropy coding | 隐式学习冗余 | 本文是可解释、轻量的线性前端 |
| **本文** | Gram 正交化 + KLT | **完整 Gram 下是** | 固定几何的颜色属性压缩 |

## 评分

| 维度 | /10 | 依据 |
| --- | ---: | --- |
| 创新性 | 8.4 | 以 render-induced Gram 度量推导 SH 压缩坐标，而非只换 entropy model。 |
| 技术可靠性 | 8.0 | 线性与能量等式推导直接；可实现版本依赖张量积近似。 |
| 实验充分度 | 6.4 | 两套数据和 RD 曲线有价值，但 v1 缺逐场景数值、复杂度、真实 Gaussian 专项实验。 |
| 写作清晰度 | 7.8 | 数学主线明确，部分图表与实验叙述仍较粗略。 |
| 实用/研究价值 | 8.2 | 只需小型线性矩阵，可作为多种 codec 的可组合模块。 |

**总体推荐：值得细读并优先做 transform 插件式复现。**

## 阅读结论

- **最值得记住的点**：压缩错误应在渲染后的 image metric 中定义；\((\Phi^\top\Phi)^{1/2}\) 将此 metric 拉回 SH 系数空间。
- **最需要怀疑的点**：实践使用 product-measure tensor 近似，且 transform 绑定评测相机分布，严格等距性与跨视角泛化并未完全验证。
- **最值得继续验证的点**：将此 transform 放到真实 3DGS 的 VQ / context codec 前，量化它在未见相机和低码率下的增益及 header/编码时间。

## 相关论文

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS 与 SH appearance 的基础。
- [HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression](https://arxiv.org/abs/2403.14530) — context/anchor 驱动的 learned compression，对比本文显式线性变换。
- [Compact 3D Gaussian Representation for Radiance Field](https://arxiv.org/abs/2311.18159) — pruning、量化与编码的联合 3DGS 压缩基线。
- [RAHT](https://ieeexplore.ieee.org/document/7566000) — 本文采用的空间属性变换基础。
