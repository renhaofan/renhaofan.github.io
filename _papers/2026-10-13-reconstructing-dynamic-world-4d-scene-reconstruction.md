---
title: "Reconstructing the Dynamic World: A Representation-Centric View of 4D Scene Reconstruction"
subtitle: "从表示、时间建模与评测协议三个轴线统一梳理动态 NeRF 与动态 3D Gaussian Splatting"
authors: "Ziren Gong, Guo Chen, Yongjia Li, Yihua Shao, Fabio Tosi, Stefano Mattoccia, Matteo Poggi, Hao Tang, Fei Ma, Shuyan Li, Ziyang Yan, Nicu Sebe, Ling Shao, Jianfei Cai, Qi Tian, Ming-Hsuan Yang"
venue: "arXiv preprint"
year: 2026
date: 2026-10-13
paper_url: "https://arxiv.org/abs/2609.39960"
code_url: "https://github.com/ZiyangYan/Awesome-4D-Scene-Reconstruction"
tags:
  - 4D Scene Reconstruction
  - Dynamic NeRF
  - 4D Gaussian Splatting
  - Dynamic Novel View Synthesis
  - Survey
summary: "这篇综述用‘场景表示如何承载时间’作为统一坐标系，将 scene-specific 4D 重建归纳为动态 NeRF 的四类方案与动态 3DGS 的三类方案，并用数据集、画质、几何与效率证据揭示三组稳定权衡：隐式表示紧凑但慢，显式 Gaussian 快但容量大，外部运动/语义/深度先验能改善欠约束动态区域却引入依赖。"
permalink: /papers/reconstructing-dynamic-world-4d-scene-reconstruction/
---

> **阅读依据**：[arXiv:2609.39960v1 官方 HTML](https://arxiv.org/html/2609.39960v1)与 PDF（2026-09-30）。本文是综述，不提出需要训练的新模型；其[官方资源页](https://github.com/ZiyangYan/Awesome-4D-Scene-Reconstruction)是持续更新的论文列表。下文的跨方法数字来自作者对原论文结果的整理，并非全部由作者在完全统一环境中复现。

## 一句话总结

作者没有再按“某个模型用了什么模块”罗列文献，而是追问**时间究竟被放进了哪里**：动态 NeRF 可用 canonical deformation、直接 4D 函数、4D feature volume 或 temporal prior，动态 3DGS 则可用显式 4D primitive、deformation field 或逐帧传播；这套 representation-centric taxonomy 把重建质量、时间一致性、速度和容量之间的选择放进了同一张地图。

## 背景、边界与核心问题

### 4D 重建到底重建什么？

任务从视觉观测恢复随时间变化的几何、外观和运动，可写成从 $(\mathbf{x},\mathbf d,t)$ 到颜色、密度或显式 primitive 状态的映射。相比静态场景，多出来的不只是一个时间坐标：非刚性运动、遮挡、拓扑变化和稀疏视角会产生多种同样能解释训练图像的解。

论文将讨论边界限定为 **scene-specific optimization**：每个场景单独优化一个高保真 4D 表示。作者承认 feed-forward 大模型推理更快，但认为其受训练集偏置和域外保真上限制约，因此不把 SLAM、SfM 或 generalizable feed-forward reconstruction 作为主线（Section I）。这个边界很重要：文中结论不能直接外推到一次前向即可处理新场景的模型。

### 为什么需要新的综述坐标系？

- 既有综述多以静态 neural rendering 为主，动态重建只是附带章节；NeRF 与 3DGS 两条线也常被分开讨论。
- “动态”可被编码在坐标变形、4D primitive、特征平面、损失先验或逐帧状态里。只按 backbone 或发表年份分类，很难看清相同假设和共同失败模式。
- 现有论文使用的采集方式、时间跨度、传感器和评价协议高度不一；单帧 PSNR/SSIM/LPIPS 又无法检验长期时间一致性。

**核心问题：能否用场景表示与时间建模的共同语言，解释不同 4D 重建方法为什么在画质、几何、速度、容量和时间一致性上表现不同？**

## 整体框架

<div class="mermaid">
flowchart LR
    A[动态视觉观测<br/>RGB / RGB-D / LiDAR] --> B{基础 radiance-field 表示}
    B --> C[Dynamic NeRF<br/>隐式体渲染]
    B --> D[Dynamic 3DGS<br/>显式 splatting]

    C --> C1[Deformation field<br/>观测空间映射到 canonical space]
    C --> C2[Implicit 4D primitive<br/>直接查询 space-time]
    C --> C3[4D feature volume<br/>平面 / 网格 / 低秩分解]
    C --> C4[Temporal prior<br/>flow / smoothness 等约束]

    D --> D1[Explicit 4D primitive<br/>4D Gaussian 按时间切片]
    D --> D2[Deformation field<br/>canonical Gaussian 加时变偏移]
    D --> D3[Frame-wise training<br/>逐帧传播并增删 Gaussian]

    C1 --> E[按场景优化]
    C2 --> E
    C3 --> E
    C4 --> E
    D1 --> E
    D2 --> E
    D3 --> E
    E --> F[任意时间 / 视角渲染<br/>及几何提取]
    F --> G[画质、几何、时空一致性、效率评测]
</div>

这张图不是单个算法的执行图，而是论文的论证框架：先选择隐式 NeRF 或显式 Gaussian，再决定时间进入表示的方式，最后通过 scene-specific optimization 拟合观测。不同分支的主要差别，是把动态复杂度放进函数、数据结构、外部先验还是跨帧状态。

## 表示基础：NeRF 与 3DGS 的共同点和差异

### NeRF：紧凑的隐式连续函数

静态 NeRF 用参数为 $\theta$ 的网络把三维位置 $\mathbf x$ 和视线方向 $\mathbf d$ 映射为颜色 $\mathbf c$ 与密度 $\sigma$：

$$
F_\theta(\mathbf x,\mathbf d)\rightarrow(\mathbf c,\sigma).
$$

射线上的像素颜色通过体渲染累积：

$$
\hat C(\mathbf r)=\sum_{i=1}^{N}T_i\left(1-e^{-\sigma_i\delta_i}\right)\mathbf c_i,
\qquad
T_i=\exp\left(-\sum_{j=1}^{i-1}\sigma_j\delta_j\right).
$$

$T_i$ 是光到达第 $i$ 个采样点前仍未被遮挡的透射率。这种表示参数紧凑、连续，但每条射线需多次网络查询，渲染慢。

### 3DGS：可光栅化的显式 primitives

3D Gaussian Splatting（3DGS）用中心 $\mu$、旋转 $R$ 与尺度 $S$ 参数化各向异性 Gaussian：

$$
G(x)=\exp\left[-\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right],
\qquad \Sigma=RSS^\top R^\top.
$$

投影到图像后，按深度排序并做 alpha compositing：

$$
C=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j).
$$

它与 NeRF 都是在射线上累积连续密度和颜色，但显式 Gaussian 可用 tile-based rasterizer 高速渲染；代价是 primitive 数量和模型容量容易膨胀（Section II-B、Table X）。

## 动态 NeRF：时间进入隐式场的四种方式

### 1. Deformation field：先把时间变化消掉

维护一个 canonical NeRF，再用时变网络把观测时刻的坐标映回 canonical space：

$$
\mathbf x'=D_\phi(\mathbf x,t),
\qquad
F_\theta(\mathbf x',\mathbf d)\rightarrow(\mathbf c,\sigma).
$$

这样可复用成熟的静态 NeRF，只需额外学习运动。它适合能由一个参考状态平滑变形得到的对象，但 canonical correspondence 本身没有唯一解；大拓扑变化、遮挡中新出现的区域和错误外部深度/光流会破坏映射。D-NeRF、Nerfies、HyperNeRF、NeRF-DS 等位于这一类（Section III-A/B、Table II）。

### 2. Implicit 4D primitive：直接拟合 space-time

不设 canonical space，直接把时间作为网络输入：

$$
F_\theta(\mathbf x,\mathbf d,t)\rightarrow(\mathbf c,\sigma).
$$

这一形式对复杂动态更自由，也不强制所有帧共享一个拓扑；但搜索域由 3D 扩展为 4D，对时间采样、容量和计算的要求更高。NSFF、DyNeRF、DynNeRF 等属于此类（Section III-A/C）。

### 3. 4D feature volume：用结构化分解换效率

典型做法把 4D 场分解成空间与时空特征平面，例如：

$$
\begin{aligned}
\mathbf f(\mathbf x,t)=
&\mathbf f_{xy}(x,y)\odot\mathbf f_{xz}(x,z)\odot\mathbf f_{yz}(y,z)\\
&\odot\mathbf f_{xt}(x,t)\odot\mathbf f_{yt}(y,t)\odot\mathbf f_{zt}(z,t),
\end{aligned}
$$

再由 $F_\theta(\mathbf f)\rightarrow(\mathbf c,\sigma)$ 解码。HexPlane、K-Planes、HyperReel、MixVoxels 等通过 plane、voxel、hash grid 或低秩张量，把昂贵的坐标 MLP 查询变成规则内存访问。核心权衡是：提高平面/网格分辨率可保留细节，却增加内存；过强低秩约束又会抹平复杂运动（Section III-A/D）。

### 4. Temporal prior：时间主要进入监督

这类方法不一定改动 radiance field 主体，而是在重建损失外加入 motion 与 temporal consistency：

$$
\mathcal L=\mathcal L_{\mathrm{recon}}
+\lambda_1\mathcal L_{\mathrm{flow}}(t)
+\lambda_2\mathcal L_{\mathrm{temp}}(t).
$$

光流、scene flow、单目深度、分割或语义特征能在稀疏视角下缩小解空间，但性能上限也被 prior 质量绑定。作者把它视为可与其他表示组合的正交设计，而不只是独立架构类别（Section III-A/E）。

## 动态 3DGS：时间进入显式 primitives 的三种方式

### 1. Explicit 4D primitive：把时间当作几何维度

每个 primitive 的均值与协方差扩展到 $\mathbb R^4$。渲染时在给定 $t$ 处对 4D Gaussian 做条件切片：

$$
\mu'=\mu_x+\Sigma_{xt}\Sigma_{tt}^{-1}(t-\mu_t),
\qquad
\Sigma'=\Sigma_{xx}-\Sigma_{xt}\Sigma_{tt}^{-1}\Sigma_{xt}^{\top}.
$$

直观上，space-time covariance 的交叉项编码位置如何随时间变化，切片后仍得到标准 3D Gaussian，因此能复用 splatting。优势是时间一致性内生于同一 primitive；不足是 4D 参数与 primitive 数量带来容量压力，长时序尤其明显（Section IV-A/B）。

### 2. Deformation field：canonical Gaussians 加时变偏移

保留静态 canonical Gaussian 集合，由网络按位置和时间预测中心、旋转和尺度偏移：

$$
(\Delta\mu,\Delta q,\Delta s)=\mathcal F_\theta(\mathbf p,t),
$$

$$
\mathcal G(t)=\{\mu+\Delta\mu,\ q\otimes\Delta q,\ s+\Delta s,\alpha,c\}.
$$

它把内容和运动解耦，通常比统一 4D primitive 更省参数，也容易表达非刚性动态；但仍继承 canonical deformation 的 correspondence 歧义。4D-GS、Deformable-3DGS、ST-4DGS、MotionGS 等属于这一类（Section IV-A/C、Table III）。

### 3. Frame-wise training：传播上一帧，再局部修补

第 $t-1$ 帧的 Gaussian 通过局部刚体变换传播到第 $t$ 帧：

$$
\mu_{i,t}=R_{i,t}\mu_{i,t-1}+T_{i,t},
\qquad
q_{i,t}=q(R_{i,t})\otimes q_{i,t-1}.
$$

遇到新显露区域或拓扑变化时，再将新 primitives 并入 $\mathcal G_t$。这种顺序优化天然适合流式输入、训练快，却可能积累跨帧漂移，并且长期一致性依赖 propagation、rigidity 或 optical-flow prior（Section IV-A/D）。

## 数据集与评测协议

作者将数据集按合成/真实、单目/多视图、短期/长期、人类中心/通用场景、RGB/多模态组织（Table IV）：

| 场景与代表数据集 | 观测条件 | 主要检验内容 |
|---|---|---|
| D-NeRF | 合成、单相机、8 场景、每段 50–200 帧 | 有可控真值的小尺度非刚性动态 |
| HyperNeRF / DyCheck | 单目或稀疏视图真实场景 | 严重欠约束下的动态新视角合成 |
| Neu3DV | 18–21 相机、30 FPS、6 场景 | 稠密多视角动态渲染 |
| NVIDIA Dynamic | 12 相机、7 个户外场景 | 非结构化真实动态与自由视点一致性 |
| Waymo / nuScenes | 多相机 + LiDAR、长时序驾驶 | 大尺度几何、远处/高速目标与多模态约束 |

指标被分成三组（Section V-B）：

- **画质**：PSNR、SSIM、LPIPS 都主要是逐帧外观指标，不能直接证明时间一致性。
- **几何**：Chamfer Distance（CD）、阈值化 F-Score 与 RMSE 分别受点密度、阈值和离群点影响。文中几何 F1 使用 5 cm 阈值。
- **实践效率**：FPS 只反映渲染速度，必须结合训练时间、参数量、内存和序列尺度解读。

论文最重要的评测批评是：领域尚缺少统一衡量 temporal coherence、topology change 和 long-term consistency 的协议。因此“PSNR 第一”不能等价于“4D 重建最好”。

## 定量证据

### 新视角合成：结构化表示与外部先验普遍有利

| 数据集 / 路线 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 来源 |
|---|---|---:|---:|---:|---|
| Neu3D / NeRF | MSTH | **32.4** | – | **0.056** | Table V |
| Neu3D / NeRF | DaReNeRF | 32.3 | – | 0.084 | Table V |
| Neu3D / 3DGS | FreeTimeGS | **33.19** | – | **0.036** | Table VI |
| Neu3D / 3DGS | TaylorGaussian | 33.02 | 0.970 | 0.053 | Table VI |
| Neu3D / 3DGS | ADC-GS | 31.67 | **0.981** | 0.061 | Table VI |
| D-NeRF / NeRF | SLS4D | **34.84** | 0.98 | **0.025** | Table VII |
| D-NeRF / NeRF | Ced-NeRF | 34.21 | **0.99** | 0.037 | Table VII |
| NeRF-DS / 3DGS | EfficientGS | **24.65** | **0.90** | **0.14**（并列） | Table VIII |

可支持的结论有三点：

1. Neu3D 上，feature-volume NeRF 的 MSTH 达到 32.4 dB，而 3DGS 路线的 FreeTimeGS 达到 33.19 dB；显式路线在这个 benchmark 上同时保持实时级渲染潜力，但这不是同一训练代码下的受控比较。
2. D-NeRF 上 SLS4D 相比原始 D-NeRF 的 30.50 dB 提高 4.34 dB，作者据此认为 semi-explicit / high-dimensional feature volume 更擅长解耦小尺度室内场景的静态结构与动态变化（Section V-C、Table VII）。
3. Neu3D 中 feature matching、optical flow、semantic constraint 等先验经常与更好结果同时出现；论文将其解释为对动态区域、floaters 和多视图不一致的正则化。这里是跨方法观察，不是消融实验，不能单独证明某个 prior 的因果贡献。

### 几何重建：主动深度缓解尺度歧义

| NuScenes 方法 | 监督 | CD ↓ | F-Score ↑ | RMSE ↓ | 来源 |
|---|---|---:|---:|---:|---|
| K-Planes | 视觉 | 0.30 | 0.89 | 6.80 | Table IX |
| LiDAR4D | LiDAR | 0.24 | 0.89 | 6.78 | Table IX |
| STGC-NeRF | LiDAR | **0.22** | **0.91** | **6.54** | Table IX |
| Deformable-3DGS | 独立协议 † | 0.38 | – | 2.97 | Table IX |
| StreetGaussian | 独立协议 † | 0.27 | – | 2.19 | Table IX |
| OmniRE | 独立协议 † | **0.24** | – | **1.89** | Table IX |

STGC-NeRF 相比视觉 K-Planes 将 CD 从 0.30 降到 0.22，并把 F-Score 从 0.89 提到 0.91，支持 LiDAR、时空 flow 与 surface-normal prior 对大尺度几何的帮助。需要注意，表中 3DGS 三行带 †，使用另一文献的协议；其 RMSE 不应与上半表直接排序。

### 效率：3DGS 的速度优势以容量为代价

| 方法 | 表示 | FPS ↑ | 训练时间（h）↓ | 参数量（MB）↓ | 来源 |
|---|---|---:|---:|---:|---|
| D-NeRF | deformation NeRF | <1 | 22.3 | **3** | Table X |
| MSTH | feature-volume NeRF | 15 | 0.3 | 135 | Table X |
| StreamRF | temporal-prior NeRF | 10.9 | 0.3 | 31 | Table X |
| 4DRotorGS | explicit 4D Gaussian | **277** | 1.0 | – | Table X |
| SpacetimeGS | explicit 4D Gaussian | 140 | 0.31 | 200 | Table X |
| HiCoM | deformation 3DGS | 274 | 1.7 | 270 | Table X |
| ED-3DGS | deformation 3DGS | 74.5 | 1.87 | 35 | Table X |
| Ex4DGS | frame-wise 3DGS | 121 | 0.6 | 115 | Table X |

作者报告这些效率值基于 Neu3D，并注明评估使用四张 NVIDIA A100。表中最鲜明的尺度差异是 D-NeRF 少于 1 FPS、3 MB，而 4DRotorGS 达 277 FPS；另一方面，部分 3DGS 模型超过 1 GB，Instant Gaussian Stream 甚至列为 2370 MB。故“显式一定更高效”不准确：它通常是**渲染更快，但存储与长序列扩展更难**。

## 论文贡献与我的洞察

### 论文贡献

- **统一 taxonomy**：动态 NeRF 的四类和动态 3DGS 的三类都由“时间放在哪里”推导，而不是按网络名字硬分组（Figures 3–4、Tables II–III）。
- **把质量与工程代价放在一起**：同时汇总画质、几何和效率，避免只靠 PSNR 描述 4D reconstruction（Section V、Tables V–X）。
- **暴露评测缺口**：明确指出主流指标缺少时间一致性、拓扑变化和长期稳定性维度（Section V-B）。
- **提供研究地图**：官方 Awesome-4D-Scene-Reconstruction 列表适合持续追踪文献。

### 我的洞察

1. **taxonomy 本质上是三种归纳偏置**：canonical deformation 假设跨时刻存在稳定对应；explicit 4D primitive 假设局部运动可由 space-time primitive 表达；frame-wise 方法假设短时间传播可靠。选择模型前，应先判断场景更符合哪种假设。
2. **feature volume 是 NeRF 向显式表示靠拢的中间态**：它保留连续解码，却把大量参数放进可索引网格/平面，因此在质量与速度上接近显式方案，也继承分辨率和内存权衡。
3. **prior 的作用是减少不可辨识性，而非凭空增加表达能力**：稀疏观测无法唯一决定运动、深度和遮挡，flow/LiDAR/semantics 通过排除不合理解来稳定优化；一旦 prior 出错，错误也会成为硬偏置。
4. **下一代 benchmark 应评价轨迹而非帧**：可跟踪 surface point、Gaussian 或语义实体跨时刻的 cycle consistency，并按遮挡、拓扑变化和时间间隔分层报告，而不是只平均逐帧 LPIPS。

## 局限与需要谨慎解读之处

### 作者明确提出的开放问题

- per-scene optimization 计算昂贵、难扩展，未来会向 transformer / large reconstruction model 驱动的 feed-forward 4D 表示迁移。
- 纯显式方法受内存限制，纯隐式方法难实时交互；显式 rasterization 与隐式 latent field 的 hybrid representation 值得探索。
- generative prior 可在稀疏或退化输入下补全不可见区域、预测运动，但会把 reconstruction 从观测恢复推向 plausible generation，需要区分“真实”与“合理”。
- 可交互、可控制的 4D scene 将服务 embodied AI、数字孪生、编辑和人类中心应用（Section VI）。

### 我的批评

- **不是统一复现的 leaderboard**：多数数值来自不同论文，硬件、实现、训练时长和数据预处理未必一致。Table X 即使声明四张 A100，也不能保证原始方法具备同等优化程度。
- **分类边界并不互斥**：temporal prior 可叠加在 deformation、feature volume 或 3DGS 上；一个方法也可能同时使用显式 primitive 和隐式 deformation。taxonomy 适合导航，不宜当作严格集合划分。
- **定量分析仍偏 NVS**：画质表多，统一几何、时间一致性和 failure-case 定量表少；这与论文自己指出的领域评测缺口一致。
- **发表数增长不等于方法成熟**：Figure 1 展示论文数迅速增加，但真实长时序、拓扑变化、未知相机和域外场景仍缺少系统证据。
- **缺少统计不确定性**：汇总表没有跨随机种子方差或显著性检验；0.1–0.2 dB 的相邻排名不应被过度解释。

### 最值得补做的实验

在同一代码基座、相同 primitive/参数预算和训练时长下，对 canonical deformation、explicit 4D primitive 与 frame-wise propagation 做受控比较；同时报告逐帧画质、跨时刻 correspondence drift、遮挡恢复、拓扑变化、峰值显存和随序列长度增长的模型大小。这个实验会比继续增加一个 PSNR 表更能验证论文提出的表示权衡。

## 与相邻路线的关系

| 路线 | 时间建模位置 | 核心优势 | 主要风险 |
|---|---|---|---|
| D-NeRF / Nerfies 类 | canonical deformation | 参数共享强、可复用静态场 | correspondence 歧义，拓扑变化困难 |
| DyNeRF 类 | 统一 space-time radiance field | 表达自由，不强制 canonical topology | 查询慢，4D 采样和容量昂贵 |
| HexPlane / K-Planes 类 | 结构化 4D feature volume | 规则内存访问、质量与效率平衡 | 分辨率、低秩假设和内存权衡 |
| 4DGS / SpacetimeGS 类 | 4D Gaussian 本体 | 时间一致性内生、splatting 快 | primitive 容量与长序列扩展 |
| 4D-GS / Deformable-3DGS 类 | canonical Gaussian + deformation | 运动与内容解耦、非刚性表达强 | 依赖 canonical mapping |
| 3DGStream / 4DGC 类 | 跨帧传播与局部更新 | 流式、训练快、适配新显露区域 | 漂移累积、长期一致性弱 |

## 对研究实践的启发

- 设计动态 Gaussian 方法时，先固定 primitive 数量与显存，再讨论 FPS；否则显式表示可用容量换质量和速度，比较不公平。
- 稀疏视角下若引入 foundation-model feature、flow 或 monocular depth，应单独测试 prior 失效场景，并报告错误是否随时间传播。
- 若研究目标是长期 4D 场景，模型大小关于帧数的增长阶数，比单个 10 秒片段的 FPS 更关键。
- 对 reconstruction 与 generation 的混合系统，建议同时报告 observation consistency 和 hallucinated-region plausibility，避免生成先验提高感知指标却改变真实几何。

## 评分

| 维度 | 评分 | 依据 |
|---|---:|---|
| 创新性 | 8.0/10 | 贡献在统一表示视角和跨 NeRF/3DGS 的组织方式，而非新算法。 |
| 技术可靠性 | 8.0/10 | 数学抽象与代表性方法吻合，但混合方法使分类存在重叠。 |
| 实验充分度 | 7.5/10 | 覆盖画质、几何和效率；仍受跨论文协议不统一、时序指标缺失限制。 |
| 写作清晰度 | 8.5/10 | 从基础表示到 taxonomy、benchmark、展望的结构清楚，表格信息密度高。 |
| 研究价值 | 9.0/10 | 适合作为进入 4D reconstruction 的地图，也能帮助定位新的评测问题。 |

**综合建议：8.3/10，值得细读。** 若目标是快速掌握动态 NeRF 与动态 3DGS 的设计空间，优先读 Sections III-A、IV-A、V 和 Tables II–X；具体实现仍需回到各原论文核验。

## 阅读结论

- **最值得记住的点**：4D 方法的根本差异不是 NeRF 对 3DGS，而是时间被编码成 deformation、4D primitive、结构化 feature、外部 prior 还是逐帧状态。
- **最需要怀疑的点**：跨论文汇总表中的细小排名差异；它们缺少完全统一的实现、预算和时序评测协议。
- **最值得继续验证的点**：在统一预算下，三种时间归纳偏置对 topology change、long-term drift 与遮挡恢复的真实影响。

## 相关论文与资源

- [NeRF](https://arxiv.org/abs/2003.08934)：隐式 radiance field 与体渲染基础。
- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079)：显式 Gaussian 与实时 splatting 基础。
- [D-NeRF](https://arxiv.org/abs/2011.13961)：canonical deformation 动态 NeRF 代表。
- [K-Planes](https://arxiv.org/abs/2301.10241)：用可解释的空间-时间平面分解动态 radiance field。
- [4D Gaussian Splatting](https://arxiv.org/abs/2310.08528)：deformation-based 动态 3DGS 代表。
- [Awesome 4D Scene Reconstruction](https://github.com/ZiyangYan/Awesome-4D-Scene-Reconstruction)：本文作者维护的持续更新资源页。
