---
title: "Inverse Rendering for Modeling with Line Primitives"
subtitle: "用随机可微光栅化和自适应 re-lining，从多视图图像重建显式毛发、纤维与绒毛线图元"
authors: "Kenji Tojo, Ariel Shamir, Nobuyuki Umetani, Bernd Bickel"
venue: "ACM Transactions on Graphics / SIGGRAPH Asia"
year: 2026
date: 2026-09-05
paper_url: "https://kenji-tojo.github.io/sa26-line-primitives/resources/sa26_lines_paper.pdf"
code_url: "https://github.com/kenji-tojo/inverse-line-primitives"
project_url: "https://kenji-tojo.github.io/sa26-line-primitives/"
tags:
  - Inverse Rendering
  - Differentiable Rasterization
  - Line Primitives
  - Fuzzy Geometry
  - Explicit Geometry
summary: "本文用二维子像素网格上的 Bresenham 线段、随机 opacity masking、MSAA 感知梯度和周期性 re-lining，将多视图中的毛发、绒毛与纤维反演为显式 polylines；在 Shelly 上以 0.046 LPIPS 优于 3DGS 的 0.057，并保持 628 FPS 的标准光栅化渲染，但代价是 409 MB 内存和受启发式拓扑更新约束的几何真实性。"
permalink: /papers/inverse-rendering-line-primitives/
---

> **阅读依据**：本笔记基于 ACM TOG / SIGGRAPH Asia 2026 官方论文、arXiv:2609.00625 的完整 LaTeX 源文件、官方项目页、公开训练代码与独立的 FuzzyDR Vulkan 光栅器。论文 DOI 为 [10.1145/3842527](https://doi.org/10.1145/3842527)，共 13 页。代码仓库提供训练、消融、viewer 和 benchmark 脚本，数据与 checkpoints 另行下载；本笔记未在本机复跑需要 RTX 级 GPU、Vulkan SDK 和数 GB 数据的完整实验。

## 一句话总结

论文把模糊外观解释为大量**不透明、子像素宽线段经过屏幕空间滤波后的聚合结果**，再通过 stochastic opacity masking 为遮挡与 primitive 生灭提供梯度、通过 microedge 为位置提供梯度，并用 prune–merge–split 的 re-lining 更新 polyline 拓扑，从而在不依赖半透明体图元的情况下恢复可被标准 z-buffer、曲线着色和物理模拟直接使用的显式 1D 几何。

## 背景与动机

- **传统 mesh 的局部光滑表面假设不适合 fuzzy geometry**：头发、毛皮、织物、植物和果皮绒毛的边界由大量细长结构共同形成，很难用少量表面与 texture/displacement 忠实表达。
- **NeRF/3DGS 能拟合外观，但丢失低维流形结构**：3D Gaussian 是无连接的体图元，适合 novel-view rendering，却不能直接使用基于 curve/surface 的切线着色、Laplacian 编辑、碰撞、测地距离和弹性杆模拟。
- **显式 line primitives 易渲染、难优化**：z-buffer 只选择最近 fragment，遮挡会切断对后方 primitive 的梯度；一像素以下的线宽也很难从图像中稳定反演。
- **已有 hair capture 通常依赖方向场或头皮先验**：这些先验适用于相对整齐的头发，却难覆盖海绵、仙人掌、织物、短毛动物等方向高度杂乱的结构。
- **作者切入点**：不把线段做成可微半透明 tubes，而是保留标准不透明光栅化，通过随机激活概率学习 primitive 是否存在；可见的半透明感只由大量细线和 anti-aliasing 产生。

**核心研究问题：能否仅从多视图 RGB 和相机位姿出发，用可标准光栅化的显式 1D primitives 达到体渲染表示对 fuzzy appearance 的拟合能力，同时保留连通性、切线与物理接口？**

## 核心问题

1. **离散深度测试没有平滑 opacity 梯度**：不透明 fragment 要么赢得 z-test、要么完全不可见，低贡献 primitive 无法逐渐“长出来”或消失。
2. **子像素线段存在强烈采样不稳定**：连续优化 world-space/screen-space width 容易出现虚线、空洞或尺度依赖。
3. **MSAA 后一个像素依赖多个随机 fragments**：原始 stochastic opacity gradient 针对单个可见 fragment，不能直接处理重建滤波覆盖的邻域联合概率。
4. **固定随机线段很难形成有效 polylines**：仅优化位置与颜色不能自动共享端点、清除冗余并把预算移动到高细节区域。
5. **外观相似不代表几何真实**：多视图仍无法观测严重遮挡的内部纤维，也不能唯一确定物理 strand connectivity 和亚像素线宽。

## 方法详解

### 整体流程

<div class="mermaid">
flowchart TB
    A[多视图 RGB + camera poses] --> B[NeuS2 粗表面]
    B --> C[表面宽带内采样 1M 短边]
    C --> D[2M vertices + edge opacity 0.1]

    subgraph Train[50k iterations：训练期]
        D --> E[SH 顶点着色]
        E --> F[2x 子像素 Bresenham 光栅化]
        F --> G[随机 opacity masking + z-buffer]
        G --> H[Gaussian filter / MSAA]
        H --> I[Photometric + opacity score-function + length loss]
        I --> J[更新位置、SH、opacity]
        J --> K[每 100 iter re-lining]
        K -->|prune / merge / split| E
    end

    K --> L[opacity >= 0.5 的显式 polylines]
    L --> M[标准 rasterization / strand shading / simulation]
</div>

输入是真实或合成多视图图像及已知相机。真实采集先用 COLMAP 求相机，再用 NeuS2 得到粗 proxy；线段不是紧贴 proxy，而是在距离带内密集随机初始化。训练同时优化 vertex position、每顶点 SH 和每边 opacity，并周期性修改连接关系。测试时关闭随机性，用固定阈值选择存在的线段。

### 显式线图元表示

场景由 vertices 和 edges 构成：

$$
\mathcal V=\{\mathbf v_v\}_{v=1}^{|\mathcal V|},
\qquad
\mathcal E=\{(v_1^e,v_2^e)\}_{e=1}^{|\mathcal E|}.
$$

每个顶点 $\mathbf v_v\in\mathbb R^3$ 带有可学习外观属性 $\mathbf A_v$，主实验使用最高三阶 spherical harmonics（SH）。每条边有 opacity $\alpha_e\in[0,1]$，表示训练期的存在概率。

方法重点约束为无分支 polylines：vertex valence 为 1 或 2。相比完全独立 segments，共享端点可以减少位置和 SH 存储，也提供可用于切线计算、连通遍历和 simulation 的 1D manifold；但它不恢复 junction 或分叉毛束。

给定视角 $t$，顶点颜色由可替换 shading function 得到：

$$
\mathbf C_v^t=\mathcal S(\mathbf v_v,\mathbf A_v;t).
$$

线段内部 fragment 颜色采用 perspective-correct interpolation：

$$
\mathbf C_f^t=(1-w)\mathbf C_{v_1^e}^t+w\mathbf C_{v_2^e}^t,
\qquad \alpha_f^t=\alpha_e.
$$

因此优化输出不是固定 shader 的烘焙体，而是标准 vertex/edge topology 加可解释属性。

### Stochastic opacity masking：让不透明 primitive 可以生灭

普通 z-buffer 对每个像素选择最近 fragment：

$$
f^*=\arg\min_{f\in\mathcal F}d_f.
$$

这对 primitive 是否存在不可微。继承自 DiffSoup 的 stochastic rasterization 为每个 fragment 采样 $\tau_f\sim U[0,1]$，只在 $\alpha_f>\tau_f$ 时激活：

$$
f^*=\underset{f\in\mathcal F:\alpha_f>\tau_f}{\arg\min}\,d_f.
$$

深度为 $d_f$ 的 fragment 被选中的概率为

$$
p_f=\alpha_f
\prod_{f':d_{f'}<d_f}(1-\alpha_{f'}).
$$

直觉上，它先要求当前 fragment 被激活，再要求所有更靠前的 fragments 都未激活；形式与 alpha compositing 的 transmittance 相似，但每次 forward 最终仍只保留一个不透明 fragment。

把 rasterizer 看成从 $p_f$ 采样，期望损失的梯度写成

$$
\nabla\mathcal L=
\mathbb E_{p_f}[\nabla\mathcal L_1(\mathbf C_f)]
+\mathbb E_{p_f}[\mathcal L_1(\mathbf C_f)\nabla\log p_f].
$$

第一项是颜色与几何的普通可微光栅化梯度；第二项是 likelihood-ratio / score-function estimator，使 opacity 可以根据“这条边出现是否降低误差”而变化。其 log-probability 梯度无需显式排序：被选 fragment 获得 $1/\alpha_f$，更靠前却未激活的 fragment 获得 $-1/(1-\alpha_{f'})$。

训练时 opacity 是连续概率；测试时令全部 $\tau_f=0.5$，等价于保留 $\alpha_e\ge0.5$ 的确定性不透明线段。实现中，同一 primitive 的全部 fragments 共用随机 counter，避免一条线内部出现随机破洞。

### 无宽度 Bresenham 线段与 microedge 梯度

论文不优化线宽。每条线用经典 Bresenham 算法在横纵均 2× 的子像素网格上画成固定 1-sample 宽的 binary coverage，再通过 filtering 得到最终像素。

选择这一离散表示的原因是：

- 每个 primitive 的覆盖稳定，不会因为估计到小于像素的连续 width 而变成断续虚线；
- 无需为不同归一化场景选择 0.2 mm、1 mm 等 world-space radius；
- forward 可映射到标准、高吞吐量的硬件线光栅化。

Bresenham fragment 边界是轴对齐、离散的，不能直接用连续 stroke silhouette 求梯度。作者将 rasterized block 的边界视为 surrogate microedges，采用 Rasterized Edge Gradients 的思想，把线段轻微移动导致的 coverage 跳变转成 vertex position gradient。

这是一种 straight-through 风格的代理：forward 严格使用离散 rasterization，backward 对其局部边界变化构造连续信号。它保证优化可用，但并不意味着 Bresenham 操作本身被解析求导。

### MSAA 与不对称的 forward/backward filter

2× 子像素图在 forward 端使用 Gaussian reconstruction filter：标准差 0.5 pixel、cut-off radius 1.5 pixels，对应 $6\times6$ 子像素支持域。这样大量不透明细线在像素尺度形成连续、半透明的 fuzzy appearance。

困难在于，filter 后像素 $\hat{\mathbf C}$ 依赖邻域 $\mathcal N$ 内多个随机 fragments，opacity gradient 需要联合概率 $p_{\mathcal N}$：

$$
\nabla_{\alpha_f}\mathcal L=
\mathbb E_{p_{\mathcal N}}
\left[\mathcal L_1(\hat{\mathbf C})
\nabla_{\alpha_f}\log p_{\mathcal N}\right].
$$

精确联合概率很难计算，论文先假设 fragments 独立：

$$
\nabla\log p_{\mathcal N}
\approx\sum_{f\in\mathcal N}\nabla\log p_f.
$$

然后进一步只对目标 pixel 内 filter weight 最大的 $2\times2$ fragments 计算 opacity gradient：

$$
\nabla\log p_{\mathcal N}
\approx\sum_{f\in\mathcal N_{\mathrm{nearest}}}\nabla\log p_f.
$$

所以这里是 **forward 6×6 Gaussian、opacity backward 2×2 box** 的有意不对称近似。来自同一 primitive 的 fragment gradients 会先求平均，让 opacity 只负责整条边的存在性，线段长短与轮廓则交给 positional gradients。

这个近似牺牲 estimator 的严格无偏性来降低方差和冲突。作者在局限中也承认，可以进一步研究结合 filter weight 与 primitive orientation 的 control variates。

### Adaptive re-lining：用离散规则改变连通性

连续梯度不能直接增加 edge、共享端点或重分配固定 vertex budget。论文每 100 iterations 执行一次 re-lining：

1. **Prune**：删除 opacity 过低或长度过短的 edges；
2. **Merge / snap**：把距离阈值内、degree 为 1 的 polyline endpoints 贪心合并，形成更长的连接曲线，并删除退化边；
3. **Split**：将最长存活 edges 从中点拆分，使用剪枝释放的 vertex slots，把总 vertex 数恢复到目标预算；
4. **State transfer**：幸存 vertices 保留 optimizer state，新 vertices 的 Adam state 从零开始，位置与 SH 由父 edge 端点插值初始化。

这比 3DGS-MCMC 式把独立 primitive 搬到高贡献区域多利用了一层结构：split 后两个 segments 共享中点，连接信息不再重复存储。

需要准确理解论文“优化 discrete connectivity”的表述：连接关系不是通过一个可微组合优化器直接求得，而是 opacity/position gradients 提供证据，再由 prune–merge–split 启发式离散更新。

### 初始化、损失与训练日程

作者声称可以从完全随机初始化工作，但主实验使用 NeuS2 coarse surface 加速：在 proxy 附近距离带 $0.1$ 内采样 1M seed points，每个 seed 初始化一条极短随机边，形成约 2M vertices；edge opacity 初值为 0.1。真实数据还需手动对齐一个 bounding cube 裁剪 NeuS2 的背景噪声。

论文写出的 photometric loss 为

$$
\mathcal L_{\text{photo}}=
\lambda\mathcal L_{\text{D-SSIM}}+(1-\lambda)\mathcal L_1,
\qquad \lambda=0.8,
$$

另加平均 squared edge length regularization，权重 $10^{-3}$。position 使用 VectorAdam，其他参数使用 Adam。共训练 50k iterations，每次随机渲染一张图；re-lining 到 35k 停止，opacity pruning threshold 在 25k 后逐渐升至测试阈值 0.5。

**源码核对发现一处重要不一致**：公开 `train_shelly_lines.py` 实际计算 `0.8 * L1 + 0.2 * D-SSIM`，即标准 3DGS 常用权重，而不是正文公式所写的 `0.8 * D-SSIM + 0.2 * L1`。源码与论文至少有一处系数或符号笔误，复现时应优先查看作者 release/issue 的后续说明，而不能同时把二者都视为正确。

代码中的默认日程还补充了：re-lining 从 iteration 500 开始；基础 opacity prune threshold 为 0.05；vertex position LR 在 Shelly/Fuzzy 上分别为 $2.5\times10^{-5}$ / $1.6\times10^{-5}$，指数衰减到初值的 1/5；SH DC、SH residual 与 opacity LR 分别为 $2.5\times10^{-3}$、$1.25\times10^{-4}$、$5\times10^{-2}$。

## 实验关键数据

### 数据集与设置

- **Shelly**：6 个 artist-authored furry scenes，有标准 train/test split，用于定量 novel-view comparison。
- **Fuzzy**：作者采集的 8 个真实物体，包括 cactus、dinosaur、flowers、fur、kiwi、tawashi 和 textiles。原图为 $6720\times4480$，每场景 108–173 views，训练时约 4× 下采样。
- **真实采集条件**：暗环境、DSLR、同轴闪光和偏振片，以压低 specular reflection；通过低亮度 threshold、morphological closing 与最大 saturation connected component 构造 foreground mask，并假设纯黑背景。
- **重要评测边界**：公开代码 README 明确说明，Fuzzy 虽提供 train/test split，但论文展示使用了**全部 captured photographs**，目标是展示可达到的最大 reconstruction quality，而非 held-out novel-view 定量泛化。公平定量比较主要依靠 Shelly。
- **硬件**：Intel i9-14900K、64 GB RAM、RTX 4090 24 GB VRAM。默认 2M vertices、三阶 SH，单场景优化约 45 分钟。

### Shelly 主结果

正文 Table 1 对 6 scenes 取平均：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 总内存 ↓ | 每 primitive 几何参数 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| DiffSoup | 30.63 | 0.920 | 0.112 | **68 MB** | 9 |
| VolSurfs | 34.68 | 0.933 | 0.109 | 116 MB | 6.02 |
| AdaptiveShells | 36.02 | 0.954 | 0.079 | 未报告 | 未报告 |
| 3DGS-MCMC（500k） | **37.73** | 0.960 | 0.057 | 118 MB | 10 |
| **Line primitives（2M vertices）** | 37.09 | **0.963** | **0.046** | 409 MB | **3.59** |

- 相比最强 PSNR 基线 3DGS，本文低 0.64 dB，但 SSIM 高 0.003、LPIPS 低 0.011，相对 LPIPS 改善约 19.3%。证据支持“感知质量可比或更好”，不支持“所有 fidelity 指标全面优于 3DGS”。
- 相比 surface methods，Line primitives 对 fuzzy boundary 的优势明显：对 DiffSoup 的 PSNR/LPIPS 改善为 +6.46 dB / -0.066，对 VolSurfs 为 +2.41 dB / -0.063。
- “geometry parameters per primitive” 最少不等于总模型最小。2M vertices 的三阶 SH 占据大量内存，409 MB 是 3DGS 的约 3.47×、DiffSoup 的约 6.0×。

### 渲染速度与质量切换

正文 Figure 2 在 Shelly 六场景上报告平均值：

| 表示 / AA 模式 | FPS ↑ | LPIPS ↓ |
| --- | ---: | ---: |
| 3DGS，500k primitives | 588 | 0.057 |
| 本文默认 Gaussian MSAA，2M vertices | 628 | **0.046** |
| 本文 hardware MSAA | 814 | 0.069 |
| 本文关闭 anti-aliasing | **944** | 0.081 |

默认模式在处理 2×2 子像素、即 4× fragments 的情况下仍略快于 3DGS，同时 LPIPS 更低。硬件 MSAA 或无 AA 能继续提升 FPS，但质量逐步下降，说明显式 line representation 的一个实际优势是可以直接借用图形 API 的质量—速度档位。

这些是 RTX 4090 上的 renderer throughput，不代表低端移动 GPU 性能。项目提供 macOS/laptop viewer 与 Web viewer，但论文未给出跨设备统一 benchmark 表。

### Re-lining 消融

正文 Table 2：

| 离散更新 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| 无离散更新 | 35.48 | 0.949 | 0.065 |
| MCMC relocation | 36.84 | 0.960 | 0.051 |
| **Re-lining** | **37.09** | **0.963** | **0.046** |

MCMC 已经显著优于固定 topology，说明“把预算从低贡献区域搬走”很重要；re-lining 在此基础上再将 LPIPS 从 0.051 降到 0.046，支持 endpoint sharing、merge 和 longest-edge split 对 1D topology 的额外作用。

不过这些模块存在交互：re-lining 同时改变 primitive 数量分布、连接关系和共享属性，不能从 0.005 LPIPS 差值中单独归因 merge 或 split 的贡献。

### 可微渲染与 primitive 类型消融

正文 Table 3：

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| 不优化 opacity | 35.12 | 0.958 | 0.053 |
| 1 mm camera-facing quads | 31.62 | 0.919 | 0.111 |
| 0.2 mm camera-facing quads | 22.52 | 0.801 | 0.178 |
| **Bresenham + stochastic opacity + MSAA** | **37.09** | **0.963** | **0.046** |

- 去掉 opacity optimization 后 LPIPS 从 0.046 退化到 0.053，证明 primitive emergence/extinction 对随机初始化的边界恢复有帮助。
- 固定 world-space quad width 对归一化尺度非常敏感；线越细，undersampling 与 holes 越严重。Bresenham 不需要 width 参数，因此更稳健。
- 同样 2M-vertex budget 下，points 平均 LPIPS 为 0.166，triangles 为 0.051，lines 为 0.046。lines 总体最好，但 Woolly 上 triangles 为 0.066、lines 为 0.084，说明 primitive choice 应与区域维度匹配。
- Gaussian reconstruction filter 的平均 LPIPS 为 0.046，box filter 为 0.059；滤波核不是纯视觉后处理，它也改变训练中的梯度质量。

### 真实数据与应用

Fuzzy 数据集主要提供 qualitative evidence。模型能恢复 kiwi 绒毛、cactus 刺、织物和毛皮的方向结构，并支持：

- Marschner 风格 strand reflectance；
- 增加 $K=6$ KNN edges 后做 mass-spring animation；
- 2M vertices 的 Chebyshev-accelerated Jacobi projective dynamics 约 55 ms/frame；
- 降至一阶 SH 后进行移动端交互预览；
- 把线导出为固定 radius cylinders，在 Mitsuba 3 中与 surface scene 一起做 path tracing。

这些演示证明“显式连通曲线可接入图形工具链”，但重建的 connectivity 不保证等于真实纤维连接；物理动画更接近对视觉代理的 plausible simulation，而非材料参数和真实拓扑的恢复。

### 关键发现

1. **Line primitives 在合适对象上比 triangles 更有效利用 vertex budget**：平均 LPIPS 0.046 对 0.051，并显著优于 points 的 0.166。
2. **随机 opacity 与 re-lining 缺一不可**：前者给 edge existence 连续证据，后者把该证据转为新的离散 topology 和预算分配。
3. **感知指标优于 3DGS，但内存不是优势**：LPIPS 0.046 优于 0.057，409 MB 却远高于 118 MB。
4. **半透明感来自 filtering，不来自半透明 primitives**：测试期 primitive 可完全不透明，仍能以子像素 coverage 聚合出 fuzzy boundary。
5. **没有一种 primitive 适合所有区域**：Woolly 的大面积圆滑 shell 更适合 triangles，论文自己的 failure case 支持 hybrid line–triangle representation。

## 亮点与洞察

### 论文亮点

- **问题定义有价值**：不满足于 view synthesis，而是追问 fuzzy appearance 能否恢复为 geometry-centric pipeline 可用的 1D manifold。
- **forward 与 deployment 对齐**：训练并未依赖测试时不存在的 soft rasterizer；测试仍是标准 Bresenham/z-buffer/MSAA 路径。
- **将 stochastic existence 与 geometric coverage 分工**：opacity 决定 primitive 是否存在，microedge 决定它往哪里移动，避免 opacity gradient 同时承担线段缩短和形变。
- **re-lining 保持固定 vertex budget**：删除无用边后拆分最长边，把 capacity 移向需要细节的位置，同时逐渐形成共享端点 polylines。
- **开放程度较高**：主代码、Vulkan differentiable rasterizer、数据下载工具、checkpoints、viewer 和 benchmark 均已提供。
- **主动展示反例**：Woolly 失败案例直接说明 line/triangle 的维度偏置，而不是只给成功的毛发和刺状物体。

### 我的洞察

- **个人分析：论文真正恢复的是“可操作的视觉几何”，不是 ground-truth fibers**。其价值标准是能否渲染、着色和模拟，而非每条 polyline 是否对应真实毛发。对内容生产很实用，但对科学测量或机器人接触规划需要额外几何验证。
- **个人分析：anti-aliasing 在这里是一种 representation model**。线段的视觉厚度由采样与 filter 定义，因此改变 AA 模式不仅是降低画质，也等价于改变从几何到图像的观测模型。
- **个人分析：stochastic opacity 是离散结构优化的松弛变量**。opacity 最终二值化，却在训练时为“边是否存在”提供可优化概率；这与 pruning mask、Bernoulli architecture search 和 Gaussian densification 中的存在权重有共同结构。
- **个人分析：re-lining 类似 1D adaptive remeshing**。prune 删除不必要 support，merge 恢复连通，split 把分辨率投到长边；它可被看作 curve domain 上的固定预算 refinement，而不只是 3DGS-MCMC 的线段版本。
- **个人分析：与 3DGS 的公平比较应同时看 perceptual quality、memory 和可编辑性**。本方法不是更小的 radiance field，而是用 3.5× 内存换取 explicit connectivity、标准 z-buffer 和 physics hooks。

## 局限与展望

### 作者承认的限制

- 严重遮挡的内部毛发无法恢复完整 topology，需要更强先验或更丰富 capture。
- 接近像素极限的 fiber orientation 和 connectivity 不准确；可考虑学习 directional connection probability。
- 当前 opacity score-function gradient 仍有方差，可用包含 filter weights 和 primitive orientations 的 control variates 改进。
- 子像素 line width 本质上难以从图像唯一恢复，但真实宽度对 close-up 与物理模拟很重要。
- 近距离放大时会看见 1-pixel Bresenham line structure，需要 LOD 或 hybrid triangle–line representation。
- Woolly 这类大、圆滑且包含平坦区域的对象难以被 lines 充分覆盖；triangle 在这些区域更合适。
- 背景去除误差会污染边界，黑色表面还可能被误判为空洞，例如 Textiles 的文字区域。

### 独立分析

- **内存开销很高**：409 MB 的默认模型限制 Web、移动和机器人部署。论文强调每 primitive geometry 参数少，但三阶 SH 按 2M vertices 存储才是总内存主因；需要报告 SH degree、quantization 和 vertex count 的 Pareto curve。
- **Fuzzy 定性结果使用全部图像训练**：这适合展示 inverse modeling 上限，却不能证明对真实 capture 的 novel-view generalization。项目 README 比正文更清楚地说明了这一点。
- **采集并不完全 casual**：暗室、同轴闪光、偏振、黑背景、mask morphology、NeuS2 和手工 bounding cube crop 构成较强 pipeline assumptions。自然光、复杂背景与非偏振高光下的鲁棒性尚未验证。
- **“离散 connectivity 的梯度”容易被过度解读**：连接并没有直接可微；edge opacity 与 vertex gradients 之后仍依赖固定阈值、贪心 endpoint snapping 和 longest-edge split。
- **MSAA opacity gradient 有两层近似**：fragment independence 不成立，同一 primitive fragments 的相关性只通过平均处理；backward 2×2 也忽略 forward Gaussian support 的多数 samples。实验显示它有效，但没有估计 bias/variance。
- **论文与代码损失权重冲突**：正文写 0.8 D-SSIM + 0.2 L1，代码写 0.2 D-SSIM + 0.8 L1。这会显著影响复现与结果解释，应由作者勘误。
- **初始化仍依赖 surface reconstruction**：线方法声称解决 surface 难以捕获 fuzzy geometry，却需要 NeuS2 proxy 确定采样区域。宽 band 缓解遗漏，但 proxy 完全漏掉的长纤维仍可能没有足够 seeds。
- **物理应用缺乏真实性评测**：mass-spring animation 使用额外 KNN connections，并未与真实材料运动或 ground-truth topology 对比。
- **基线预算有多维差异**：3DGS 使用 500k primitives、lines 使用 2M vertices；作者按相近 rendering speed 选择，但内存和参数数目不匹配，因此各表回答的是不同公平性问题。

### 建议后续实验

1. 在 Fuzzy 官方 train/test split 上报告逐场景 PSNR/SSIM/LPIPS，并增加自然背景、室内照明和无偏振采集。
2. 对 0.5M–4M vertices、SH degree 0–3、16/8-bit attributes 绘制质量—内存—FPS 三维 Pareto curve。
3. 用 finite differences 或小场景枚举测 opacity gradient 的 bias/variance，比较 independence、2×2 nearest、完整 filter-weight estimator 和 control variates。
4. 将区域分类为 surface-like 与 fiber-like，自适应分配 triangles 和 lines，并在 Woolly/flat textile 上验证 hybrid representation。
5. 对可见 fibers 有 ground-truth curves 的合成集报告 Chamfer、tangent error、connectivity precision/recall，而不只看 rendering metrics。
6. 对比 NeuS2 band initialization、COLMAP points、纯随机空间 seeds 和 3DGS means 初始化，量化 proxy 依赖与 convergence speed。
7. 在固定 total memory 而非固定 FPS 下重新比较 3DGS、lines 与 triangles，验证 explicit topology 的质量代价。

## 与相关工作的对比

| 方法 | Primitive / 表示 | Visibility 与可微性 | Topology | 主要优缺点 |
| --- | --- | --- | --- | --- |
| DiffSoup | 少量 textured triangles | stochastic opacity + rasterized edge gradients | triangle soup，可离散 remesh | 内存低、surface 区域高效；不擅长显式子像素 fuzzy boundaries |
| VolSurfs | 多层半透明 surfaces | differentiable layered rendering | 2D surfaces | 比单层 mesh 更适合模糊体积，但曲线方向结构仍是间接表示 |
| AdaptiveShells | learned volumetric shells | neural volume rendering | 非显式 1D connectivity | 质量强，需专用 neural rendering，几何工具链兼容性弱 |
| 3DGS-MCMC | 500k volumetric Gaussians | alpha splatting + densification/relocation | 无连接 points | PSNR 与内存更好；不能直接提供 strand connectivity 和标准 depth-tested curves |
| Dr.Hair | camera-facing line quads | differentiable line rendering | scalp-guided hair strands | 对头发有强先验；固定 world-space width 与方向场不适合任意 fuzzy object |
| **本文** | **约 2M vertices 的 opaque polylines** | **Bresenham + MSAA + stochastic opacity + microedges** | **heuristic re-lining** | **LPIPS 与可操作性强、渲染快；内存大，拓扑与物理宽度不保证真实** |

最接近的方法是 DiffSoup：两者都用 stochastic opacity 让不透明 rasterization 可优化。区别在于 DiffSoup 用少量 triangles 和连续 texture 吸收高频外观，本文则用大量 lines 把高频边界直接变成几何，并必须专门处理 subpixel AA 与 1D connectivity。

## 启发与关联

- **3DGS surface reconstruction**：可先从 3DGS/2DGS 提取 coarse surface 或 high-contribution means，再用 line primitives 专门解释高曲率、细杆、毛发和 silhouette residual，形成 surface + line hybrid。
- **显式重建的维度选择**：平坦区域适合 2D triangles，纤维区域适合 1D lines，离散颗粒适合 0D points/Gaussians。primitive dimension 可以成为待优化的变量，而不是整场景固定选择。
- **Gaussian densification 的替代启发**：re-lining 的“删除低存在概率、合并近端点、拆分最长边”可迁移到 Gaussian graph 或 surfel mesh，使 densification 同时维护 connectivity。
- **假设：用 Gaussian covariance 初始化线方向**：将细长 Gaussian 的主轴作为短 line seed 的方向、opacity/SH 初始化为 edge 属性，可能比 NeuS2 宽带随机方向更快收敛，并减少 surface proxy 对长毛的遗漏。
- **假设：语义/实例约束 re-lining**：在 endpoint snapping 时加入 feature、normal、tangent 或 instance compatibility，可能避免仅因空间接近而错误连接不同纤维。
- **假设：多尺度 line splatting**：远距离用预滤波 line clusters 或 anisotropic Gaussian proxies，近距离切换显式 cylinders/curves，可以缓解固定一像素表示的 LOD 问题。

## 评分

| 维度 | 评分（10 分） | 理由 |
| --- | ---: | --- |
| 创新性 | 8.8 | 将 stochastic opaque rasterization 扩展到 MSAA 子像素 lines，并与 1D re-lining 组合，问题与机制都较新颖 |
| 技术可靠性 | 8.0 | 主组件有系统消融、源码和公开 rasterizer；但 opacity gradient 是偏置近似，且论文/代码损失权重不一致 |
| 实验充分度 | 7.9 | Shelly 有强定量对比、完整消融与速度数据，Fuzzy 有丰富对象和应用；真实数据缺 held-out 定量与几何/拓扑 ground truth |
| 写作清晰度 | 8.5 | 从表示动机到 stochastic gradient、MSAA 与 re-lining 的主线清楚；“对 connectivity 求梯度”的措辞比实际实现更强 |
| 实用 / 研究价值 | 8.7 | 显式 lines 可直接接入 shading、rasterization 和 simulation，代码开放；409 MB 内存和受控采集限制当前部署范围 |

**总体推荐：值得细读。** 对 differentiable rasterization、3DGS 几何化、毛发/纤维重建、显式 primitive optimization 和 hybrid representation 研究者尤其有启发。它不是对 3DGS 的全面替代，而是证明了在 fuzzy geometry 上，显式 1D manifold 可以用更高内存换来更强的结构接口和很有竞争力的感知质量。

## 阅读结论

- **最值得记住的点**：通过随机 opacity 学 edge existence、通过 microedge 学位置、再用 re-lining 做离散 topology 更新，三者共同把标准不透明线光栅化变成可工作的 inverse renderer。
- **最需要怀疑的点**：409 MB 默认内存、Fuzzy 全视图训练、强采集预处理，以及论文/代码 D-SSIM 权重冲突，都削弱了“通用、可部署重建”的强版本结论。
- **最值得复现或继续验证的点**：MSAA opacity gradient 的 bias/variance，以及 line–triangle hybrid 在固定内存预算下能否同时解决 Woolly 平坦区域与毛发边界。

## 相关论文与资源

- [官方项目页](https://kenji-tojo.github.io/sa26-line-primitives/) — 完整论文、视频、代码、数据集和结果资源。
- [官方实现](https://github.com/kenji-tojo/inverse-line-primitives) — 训练、re-lining、消融、Vulkan viewer 与 benchmark。
- [FuzzyDR](https://github.com/kenji-tojo/fuzzydr) — 支持 stochastic opacity masking 与 MSAA 的独立可微光栅器。
- [DiffSoup](https://kenji-tojo.github.io/publications/diffsoup/) — stochastic opacity masking 的直接前序工作，使用不透明 triangle soup。
- [Adaptive Shells](https://research.nvidia.com/labs/toronto-ai/adaptive-shells/) — Shelly 数据集与 volumetric baseline。
- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) — 主要 volumetric primitive baseline。
- [Rasterized Edge Gradients](https://doi.org/10.1007/978-3-031-73010-8_20) — 本文 positional microedge gradient 的基础。
