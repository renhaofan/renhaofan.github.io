---
title: "Building Rome from a Single Image"
subtitle: "用距离自适应场景分块、显式可见性条件与自回归 latent outpainting，把对象级 3D 生成器扩展到单图场景 mesh"
authors: "Jiraphon Yenphraphai, Fang Li, Tianshuo Xu, Depu Meng, Quentin Herau, Yihan Hu, Raymond A. Yeh, Wei Zhan"
venue: "arXiv preprint"
year: 2026
date: 2026-10-14
paper_url: "https://arxiv.org/abs/2610.08790"
project_url: "https://build-rome.github.io/"
code_url: "https://github.com/Applied-Intuition-Open-Source/BuildRome"
checkpoint_url: "https://huggingface.co/AppliedIntuitionResearch/BuildRome"
tags:
  - Single-Image 3D Reconstruction
  - Scene Generation
  - Adaptive Chunks
  - TRELLIS 2
  - Flow Matching
  - Generative Completion
summary: "Building Rome 将对象级 TRELLIS 2 改造成单图场景 mesh 生成器：根据相机距离和局部高度动态放大物理 chunk，以显式投影的 DINO 特征与 signed depth residual 区分自由空间、可见表面和遮挡区，再从近到远复用重叠 latents 自回归补全；配合约 4,000 个 agentic 合成户外场景，它在 Tanks and Temples、ScanNet++ 和 121 张野外图像上全面领先所列基线，但仍依赖 MoGe-3 的尺度与相机估计，只输出无纹理且不保证 watertight 的 mesh。"
permalink: /papers/building-rome-from-a-single-image/
---

> **阅读依据**：[arXiv:2610.08790v1 官方 HTML](https://arxiv.org/html/2610.08790v1)与 PDF（2026-10-06，20 页，含附录）、[官方项目页](https://build-rome.github.io/)、[官方推理代码](https://github.com/Applied-Intuition-Open-Source/BuildRome)和[公开 checkpoints](https://huggingface.co/AppliedIntuitionResearch/BuildRome)。论文中的实验结论与代码仓库后来补充的部署信息会分别注明。

## 一句话总结

Building Rome 不把整座场景压进一个对象生成器的固定体积，也不以大量等尺寸局部块暴力铺满空间，而是让物理 chunk 随深度和建筑高度增大；每个 chunk 用 monocular point map 把图像特征准确放到已观测表面，并用 signed depth 告诉生成器哪里必须为空、哪里可自由补全，最后通过近到远的 latent outpainting 生成一个完整场景 mesh。

## 先给结论

这篇论文最值得关注的是它对“reconstruction”和“generation”的空间分工：

- **相机到观测表面之前**是 free space，模型不应生成东西；
- **观测表面附近**有单图证据，应尽量忠实重建；
- **观测表面之后**是 occluded/unknown region，应由 3D generative prior 补全。

显式把这三个区域告诉 3D tokens，比让 cross-attention 自己猜 2D–3D 对应更直接。实验上，方法在 Tanks and Temples 将最强 baseline 的 normalized CD 从 2.63 降到 1.99、F1 从 0.388 提到 0.506；但输出没有纹理、不保证 watertight，而且 geometry、metric scale、chunk placement 和相机内参都依赖 MoGe-3 的单图估计。

## 背景与核心问题

单图场景生成与多视图重建的根本区别是：输入只看到一侧，系统不仅要估计可见深度，还要为墙后、物体背面和视锥遮挡区生成合理几何。现有路线各有缺陷：

- 2D diffusion / video generation 再重建，可能生成漂亮但跨视角不一致的内容，最终出现 floating geometry；
- per-object generation 再组合，依赖检测、分割和布局求解，难覆盖非对象化结构；
- direct scene generator 有完整性先验，但训练集主要是室内场景，面对户外建筑容易施加 room-like prior；
- 对象级 3D generator 拥有强大的 shape prior，却假设对象居中并归一化到固定 canonical volume，不能直接保持真实场景的尺度和布局。

作者基于 TRELLIS 2，需要回答三个问题：

1. 固定分辨率对象生成器怎样覆盖从近处小物体到数百米建筑的空间尺度？
2. 怎样让生成结果服从单图中已经观测到的 3D 位置，同时只在真正未知的遮挡区使用生成先验？
3. 户外完整 mesh 数据稀缺，如何获得足够多的训练监督？

## 整体框架

<div class="mermaid">
flowchart LR
    subgraph Offline[训练数据与训练]
        T1[SAGE-10k / Infinigen<br/>室内完整 meshes] --> T4[canonical 3 m chunks]
        T2[4,000 户外提示与参考图] --> T3[VLM 场景规划 + 对象生成<br/>terrain + layout solver + critic]
        T3 --> T4
        T4 --> T5[GT depth / camera 条件]
        T5 --> T6[TRELLIS 2 + rank-32 LoRA<br/>新 lifting / visibility modules]
    end

    subgraph Inference[单图推理]
        A[一张 RGB 图像] --> B[MoGe-3<br/>相机内参与 metric point map]
        A --> C[DINOv3 image features]
        B --> D[ground-aligned frame<br/>adaptive chunk allocation]
        D --> E[由近到远选择下一个 chunk]
        E --> F[缩放到 canonical 3 m volume]
        B --> G[signed depth residual<br/>free / surface / hidden]
        C --> H[仅向 observed surface<br/>lift image features]
        F --> I[Sparse Structure flow]
        G --> I
        H --> I
        I --> J[Structured Latent flow]
        H --> J
        K[前序 chunk 的 overlap latents] --> I
        K --> J
        J --> L[写回 shared scene latents]
        L --> E
        L --> M[统一解码完整 scene mesh]
    end
</div>

最终输出是带 metric layout 的**无纹理 mesh**。训练时使用真实/合成场景的 ground-truth depth 与 camera；推理时换成 MoGe-3 估计。基础 TRELLIS 2 权重主要通过 rank-32 LoRA 适配，另训练显式条件模块。

## 方法详解

### 1. 基础表示：TRELLIS 2 的两阶段 O-Voxel 生成

TRELLIS 2 是对象级 image-conditioned 3D generator。它以 O-Voxels 表示 surface：只保存 active surface voxels，丢弃 empty voxels，因此既不要求 watertight，也能表达开放和内部表面。

生成分两步（Section 3）：

1. **Sparse-Structure stage**：flow-matching model 先生成粗 latent grid，再解码成 active voxel positions；
2. **Structured-Latent stage**：在 active voxels 上生成 geometry latents，由 pretrained decoder 转成 mesh。

原模型只在 $[-0.5,0.5]^3$ canonical volume 内生成居中对象。Building Rome 保留其对象先验，却重新设计场景的空间划分、几何条件和跨 chunk 生成。

### 2. Adaptive chunks：把分辨率分配给图像真正看得清的地方

透视投影下，相同物理大小的区域离相机越远，在图像中占的 pixels 越少。因此没有必要让 100 m 外的建筑与近处椅子使用相同物理 voxel size。

作者建立 ground-aligned coordinate frame：原点位于相机中心在 ground plane 上的投影，$+z$ 向上、$+y$ 沿相机前方、$+x$ 横向。沿 $+y$ 把场景切成重叠 depth regions，第 $i$ 个 region 使用尺度因子 $a_i\ge1$：

$$
L_i=a_iL,
\qquad
\Delta_i=\frac{a_iL}{N},
\qquad L=3\ \mathrm m.
$$

$L_i$ 是物理 chunk 边长，$N$ 是每轴 voxel 数，$\Delta_i$ 是物理 voxel size。模型始终看到边长 $L=3$ m 的 canonical chunk；实际几何坐标、depth 和 camera translation 同除以 $a_i$，camera intrinsics 不变。因为 geometry 与 camera 一起缩放，图像投影保持不变，所以同一生成器可处理不同物理尺度。

chunk allocation 依据 MoGe-3 point cloud：

- 能容纳局部高度时，从最小 chunks 开始；
- 向远处扩展时，每个新 depth row 至少把物理尺寸翻倍；
- 若一个 depth range 的 99th-percentile height 超过 2.8 m，再提高尺度以容纳高建筑；
- 同一区域相邻 chunks 在 canonical space 重叠 0.5 m，即物理重叠为 $0.5a_i$ m；
- 横向覆盖估计 point cloud，沿深度扩展到 99th-percentile depth。

这解决的是计算–细节分配，而不是凭空提高远处精度：larger distant chunks 覆盖更广，但它们的物理 voxel 也更粗。

### 3. Explicit 2D–3D conditioning：把图像只放到对应表面

对于 canonical chunk 内 voxel center $v$，令 $\pi(v)$ 是图像投影，$z_{\mathrm{cam}}(v)$ 是相机深度，$D$ 是归一化后的 observed depth。定义沿 ray 的深度残差：

$$
r(v)=z_{\mathrm{cam}}(v)-D(\pi(v)).
$$

这个量的符号有明确几何意义：

- $r(v)<0$：voxel 在相机和表面之间，属于 free space；
- $r(v)\approx0$：voxel 位于 observed surface；
- $r(v)>0$：voxel 在表面之后，属于 unobserved/occluded region。

#### 只在 observed surface 提升图像特征

从输入图像提取 DINOv3 feature map $F$，只给靠近表面的 voxel 分配对应 feature：

$$
f_{\mathrm{lift}}(v)=
\begin{cases}
F(\pi(v)), & |r(v)|<\tau,\\
0, & \text{otherwise}.
\end{cases}
$$

$\tau$ 设为 1.5 个 voxel widths；canonical units 下 sparse-structure / structured-latent stages 分别为 0.28 m / 0.07 m，在 region $i$ 中物理容差随 $a_i$ 缩放。这样 DINO content 不会沿整条 ray 复制成“厚墙”或重复表面。

#### 显式区分自由空间与遮挡区

没有 lifted feature 的 voxel 可能是“表面前必须为空”，也可能是“表面后可以生成”。作者把 residual 裁剪后 Fourier encode：

$$
s(v)=\operatorname{clamp}(r(v),-\delta,\delta),
\qquad \delta=1\ \mathrm m,
$$

$$
g(v)=\operatorname{MLP}\left(
[\cos(\boldsymbol\omega s(v)),\sin(\boldsymbol\omega s(v))]
\right).
$$

在每个 sparse-structure transformer block 前加入局部条件：

$$
h_\ell(v)\leftarrow h_\ell(v)
+W_\ell^{\mathrm{img}}f_{\mathrm{lift}}(v)
+W_\ell^{\mathrm{geom}}g(v).
$$

Sparse Structure stage 同时需要“图像是什么”和“这里能否有 surface”；Structured Latent stage 只使用 lifted-image term，因为 occupancy/free-space 已由第一阶段决定。

与普通 cross-attention 的本质区别是：模型无需自己学习每个 3D token 对应哪个 image patch，也无需猜一个无 feature 的位置是 free 还是 hidden。几何投影把这两件事直接编码进条件。

### 4. 近到远的 autoregressive latent outpainting

adaptive allocation 决定所有 chunks 后，系统从靠近相机的 chunks 开始顺序生成。新 chunk 与已有场景重叠时：

1. 把 shared scene frame 中已生成 overlap latents 复制进新 chunk；
2. 在**每个 denoising step** 都重新固定这些 latents，而不是只用一次 initialization；
3. 新 chunk 只生成未覆盖部分，因此相当于 3D latent outpainting；
4. 跨不同尺度的 depth regions 时，把上一 region latents 三线性 resample 到新 cell locations，再同样固定；
5. 所有 chunks 完成后，将 assembled latents 一次解码为完整 scene mesh。

硬约束 overlap 能让邻块几何连续，却带来明显的顺序依赖：早期 chunk 错误会作为条件传到后续 chunks。Appendix A.3 也明确把 autoregressive error propagation 列为局限。

### 5. 用不完美 agentic scenes 补户外训练分布

完整户外 scene mesh 很少，作者构造约 4,000 个场景、覆盖 20 类户外环境（Section 4.3、Appendix A.4）：

1. GPT-5.6 生成场景提示，Ideogram 4 生成 reference image；
2. GPT-5.6/VLM 识别对象、尺寸、空间关系和 ground regions；
3. 裁出可见对象，用 Nano Banana 补全被遮挡视图，再由 TRELLIS 2 转成 object mesh；
4. MoGe-3 point map、top-view layout 与 procedural height field 构造 terrain；
5. search-based layout solver 放置对象、排除碰撞，并通过 physics simulation 落到地面；
6. GPT-5.6 critic 对比 reference 与 top/corner renders，修正漏掉的对象和错误关系，再运行一次布局求解。

assembled scene 未必精确复原 reference image，但最终训练 render、depth、camera 和 target mesh 来自同一个 assembled scene，内部监督保持一致。论文 Figure 4 还展示模型输出可以比生成训练数据的 agentic pipeline 更忠实于新输入；这说明模型学习的是跨样本 prior，而非简单复制合成 artifact。

## 训练与公开实现

- 室内数据：SAGE-10k / Infinigen scene meshes，每场景渲染 16 个 $1024\times1024$ views。
- 训练单位：独立、重叠的 canonical 3 m chunks，不需一次加载完整 scene。
- 适配方式：TRELLIS 2 上训练 rank-32 LoRA 与新 conditioning modules。
- 优化：AdamW，100k steps，batch size 64，16 张 NVIDIA A100；新条件模块学习率 $10^{-4}$，LoRA 为 $3\times10^{-5}$。
- 推理几何来源：MoGe-3 估计 camera intrinsics 与 metric point map。
- 官方 release：包含 inference code；LoRA/conditioning deltas 分为 SS 115M parameters（438 MiB）和 SLat 66M parameters（251 MiB），基础 TRELLIS 2、DINOv3、MoGe-3 权重另行下载。
- 许可：官方代码与 BuildRome checkpoints 为 CC BY-NC 4.0，非商业使用。

官方 README 的可复现示例显示：单张 A100 80GB 上，一个延伸到 98 m、4 个 depth bands、15 chunks 的 street scene 约需 6 分钟，allocated GPU memory 峰值 15.8 GiB。它与论文 Appendix A.1 的 280 m 案例（4 regions、15 chunks、约 5 分钟、26 GB）场景和统计口径不同，不宜直接比较。

## 实验设置

- **Tanks and Temples**：7 scenes，有 ground-truth geometry。
- **ScanNet++**：随机选择 32 scenes，有 ground-truth geometry。
- **In-the-wild**：121 张图像，其中 53 indoor、68 outdoor，来自 Atlas、地标搜索、NeRF/mip-NeRF 与 MoGe examples。
- baselines 覆盖 iterative completion、object composition、generated video→3D 和 direct scene reconstruction，共 7 个方法。

几何评测将 mesh area-weighted sample 成 point cloud，所有 prediction 先用 ICP 对齐 GT。Chamfer Distance（CD，越低越好）再按 median ground-truth depth 归一化；F1 使用 10 cm 阈值（越高越好）。ICP 会减弱初始坐标对齐误差对指标的影响，因此这些数字主要评价 shape，而不是完整的 metric pose recovery。

野外图像无 GT，使用 DreamSim（越低越好）评估输入图像感知一致性，以 Lotus2 估计 normal map 后计算 CLIP-N（越高越好），并做 blind pairwise user study。它们能衡量“看起来是否一致”，不能证明背面补全真实。

## 主实验

| 方法 | T&T CD ↓ | T&T F1 ↑ | ScanNet++ CD ↓ | ScanNet++ F1 ↑ | Wild DreamSim ↓ | Wild CLIP-N ↑ |
|---|---:|---:|---:|---:|---:|---:|
| EvoScene | 4.61 | 0.241 | 5.70 | 0.690 | 0.379 | 0.797 |
| Lyra 2.0 | 3.05 | **0.388** | 8.17 | 0.481 | 0.347 | 0.810 |
| VolFill | **2.63** | 0.376 | – | – | 0.339 | 0.789 |
| GenRecon | 5.57 | 0.215 | 10.06 | 0.411 | 0.294 | 0.847 |
| World Tracing | 3.27 | 0.337 | **5.21** | **0.690** | **0.237** | **0.859** |
| **Building Rome** | **1.99** | **0.506** | **5.01** | **0.702** | **0.221** | **0.870** |

数据来自 Table 1。可支持的结论如下：

1. **Tanks and Temples 增益最大且明确。** 不同指标的最强 baseline 分别是 VolFill CD 2.63 与 Lyra 2.0 F1 0.388；本文达到 1.99 / 0.506，即 CD 约降低 24%、F1 约提高 30%。
2. **ScanNet++ 是小幅领先。** CD 从最强 baseline 的 5.21 降到 5.01，F1 从 0.690 提到 0.702；不能把“所有指标第一”误读成大幅领先。
3. **野外图像的一致性指标全面最好。** DreamSim 0.221、CLIP-N 0.870，均优于 World Tracing 的 0.237/0.859。
4. **用户偏好优势取决于 baseline。** 用户在与 GenRecon 比较时 73.0% 选择本文，这是最低但仍明显过半；对其他 baselines 的胜率为 88.9%–94.4%（Table 1）。论文没有在正文中报告参与者数量或置信区间，因此不应声称统计显著。

Figure 5 与 Appendix Figures A3–A4 的主要定性现象是：GenRecon 在户外施加 room-like geometry，World Tracing 在重遮挡处有 holes，Lyra 2.0 的生成视频虽外观合理却因跨帧不一致产生 floating/noisy geometry，VolFill 户外结果粗糙且有线状 artifact。

## 消融实验

### 1. Adaptive chunks：不是单纯“块越小越好”

| Chunk layout | Indoor CD ↓ | Indoor F1 ↑ | Outdoor CD ↓ | Outdoor F1 ↑ | 时间 ↓ |
|---|---:|---:|---:|---:|---:|
| Single chunk | 2.87 | 0.759 | 49.9 | 0.138 | **1 min** |
| Fixed 3 m | 3.13 | 0.782 | 38.5 | 0.100 | 15 min |
| **Adaptive** | **2.86** | **0.792** | **37.4** | **0.240** | 5 min |

Table 2 表明 adaptive 对室内 CD 的改善极小，但 F1 最好；户外 F1 从 single chunk 的 0.138 提到 0.240。Fixed 3 m 虽分辨率细，户外 F1 反而只有 0.100，作者观察到 duplicated objects 与错误 layout，说明缺少足够 spatial context 的小块会妨碍 completion。

adaptive 相比 fixed-size 快 3 倍。Appendix A.1 的 280 m 场景更直观：fixed 3 m 约需 200 chunks，在 80GB A100 上 OOM；adaptive 只用 4 depth regions、15 chunks，约 5 分钟、26 GB。

### 2. Explicit conditioning：visibility 主要提高完整性

| Conditioning | Indoor CD ↓ | Indoor F1 ↑ | Outdoor CD ↓ | Outdoor F1 ↑ |
|---|---:|---:|---:|---:|
| SAM 3D style | 4.38 | 0.707 | 8.30 | 0.257 |
| Projection only | 4.74 | 0.739 | 8.26 | 0.330 |
| Ours w/o visibility | 4.24 | 0.757 | **6.60** | 0.464 |
| **Full conditioning** | **3.54** | **0.793** | 6.90 | **0.501** |

Table 3 支持显式 surface lifting 优于让 cross-attention 隐式学习对应。加入 visibility 后 indoor 两项和 outdoor F1 均改善；但 outdoor CD 从 6.60 变为 6.90，略有退化。因此论文 caption 所说“全面优于”需要更细读：visibility 更像是减少 holes、提高 precision–recall 平衡，而不是让所有表面距离都更准。

### 3. 约 4,000 个户外场景确实有效

| Training data | Indoor CD ↓ | Indoor F1 ↑ | Outdoor CD ↓ | Outdoor F1 ↑ |
|---|---:|---:|---:|---:|
| Existing data | 3.99 | 0.784 | 10.75 | 0.317 |
| Existing + synthetic outdoor | **3.54** | **0.793** | **6.90** | **0.501** |

Table 4 显示户外数据使 outdoor CD 降低约 36%、F1 提升约 58%，同时室内没有退化。这是论文关于 data contribution 最有力的证据。不过它只比较“有/无整套户外数据”，不能区分 20 个类别、agentic critic、terrain pipeline 或样本数量各自的贡献。

## 亮点与我的洞察

### 论文亮点

- **把透视规律转成计算分配策略**：近处小 chunks 保细节、远处大 chunks 保覆盖，比固定世界分辨率更符合单图信息密度。
- **把 observation boundary 显式交给生成器**：surface lifting 负责忠实重建，signed residual 负责区分 free/unknown，使 completion 的作用区域更可控。
- **latent overlap 是硬一致性而非软提示**：重叠区域在每个 denoising step 被重新固定，邻块不会各自生成互相冲突的边界。
- **公开完整推理栈和权重**：代码、LoRA deltas、配置与实际 A100 示例已发布，优于只有项目视频的系统论文。
- **数据结论有明确消融**：合成户外场景显著提升户外，同时没有牺牲室内结果。

### 我的洞察

1. **这是一种 perspective-adaptive level of detail。** adaptive chunks 并非传统 octree：尺度主要沿 camera depth 增长，目标不是均匀世界误差，而是匹配单图中的可用 pixel evidence。
2. **方法把 epistemic boundary 写进网络输入。** $r(v)$ 不只是 depth feature，而是在告诉模型“这里已知为空、这里被观测、这里未知”。这比仅给 monocular point cloud 更接近生成式重建真正需要的状态表示。
3. **自回归顺序建立了隐式 scene prior 传播方向。** 近处/可见几何先生成，远处 completion 受它约束；适合从相机向外扩张，但也会使早期 scale/layout error 系统性传播。
4. **合成数据无需逐图逼真，但监督必须内部一致。** agentic scene 可能不忠实于 reference，训练 render 与 target mesh 却来自同一个 assembled world；这比使用漂亮但多视图不一致的生成视频更适合作为 3D supervision。

## 局限与风险

### 作者明确承认的局限

- 输出只有 geometry，没有 texture；未来考虑 textured output 或 3D Gaussian Splatting。
- TRELLIS 2 不保证 watertight，偶尔出现 holes，需要 mesh post-processing。
- 依赖 estimated depth 与 camera intrinsics；错误会影响 chunk placement、metric scale 和 feature alignment。
- autoregressive overlap reuse 会把前序 chunks 的错误传给后续 chunks。
- 当前只支持一张输入图，未来计划扩展多图。

### 独立分析

- **不可见区域没有真实性 GT。** T&T/ScanNet++ 指标虽有 3D GT，但单图可见范围、ICP alignment 和 point sampling 会混合 visible reconstruction 与 hidden completion；应分别报告 observed / occluded geometry。
- **MoGe-3 是强依赖而非普通预处理。** 它同时决定 metric scale、相机、point map、adaptive bands 和 2D–3D lifting；没有替换/扰动实验，无法知道主模型对 depth bias 的容忍度。
- **hard overlap constraint 会锁入错误。** 若第一块产生错误墙面，后续块无法在 overlap 中修正；可考虑 uncertainty-weighted soft anchoring 或双向 refinement。
- **adaptive resolution 对远景是主动降分辨率。** Rome/建筑视觉上完整，不代表远处 façade 拥有同等 metric detail；论文应报告随深度变化的 surface error。
- **野外指标仍偏输入视角。** DreamSim 和 CLIP-N 衡量与单张输入及估计 normal 的一致性，不能验证转到背面后的结构合理性。
- **训练数据的生成模型依赖复杂。** GPT-5.6、Ideogram 4、Nano Banana、TRELLIS 2、MoGe-3 和 layout solver 组成很长的数据链，复建同等训练集的成本和许可边界较高；公开 checkpoints 缓解使用门槛，但不等于数据可复现。
- **非商业许可限制部署。** 官方代码与 deltas 为 CC BY-NC 4.0，商业项目不能直接采用。

## 与相关路线的比较

| 路线 | 隐藏区域来源 | 3D 一致性 | 典型问题 |
|---|---|---|---|
| Iterative 2D inpainting + lifting | 2D generative prior | 逐视图累积 | error drift、跨视角不一致、慢 |
| Generated video → 3D（Lyra 2.0） | video/world model | 后续 reconstruction | 视频真实不等于 geometry coherent，可能产生 floaters |
| Object composition（3D-RE-GEN） | 每对象 shape prior | layout solver 组合 | 非对象表面、遮挡关系和整体布局困难 |
| Fixed local chunks（GenRecon） | scene generative prior | overlap / local context | 户外域偏移，远大场景需大量 chunks |
| Direct completion（World Tracing） | learned scene prior | 一次/直接场景预测 | 重遮挡处 holes、细节不足 |
| **Building Rome** | TRELLIS 2 object prior + outdoor scene data | explicit surface/free/hidden conditioning + latent outpainting | 依赖单目几何；无纹理；顺序误差传播 |

## 最值得继续验证的实验

1. 按 observed surface、轻遮挡、完全不可见区域分别报告 CD/F1，区分 reconstruction 与 hallucinated completion。
2. 对 MoGe-3 depth、focal length 和 metric scale 注入可控扰动，画出最终 mesh error 与 chunk-layout failure curve。
3. 将 hard overlap replacement 改为按 uncertainty 退火的 soft constraint，再加一个远到近 refinement pass，检验是否减少错误传播。
4. 在相同 chunk 数/时间预算下比较 fixed、adaptive 与可学习 scale allocation，避免只比较默认布局。
5. 报告 geometry error 随 camera depth 的变化，验证“大 chunk 覆盖远景”究竟牺牲了多少局部细节。

## 评分

| 维度 | 评分 | 依据 |
|---|---:|---|
| 创新性 | 8.5/10 | adaptive physical chunks、free/surface/hidden 条件与 latent outpainting 的组合直指单图场景生成痛点。 |
| 技术可靠性 | 8.3/10 | 公式与系统闭环清楚，核心模块有消融；MoGe-3 敏感性和 hidden geometry accuracy 尚未拆开验证。 |
| 实验充分度 | 8.1/10 | 三类数据、7 个 baselines、用户研究与消融较完整；缺置信区间、深度分层和不可见区域指标。 |
| 写作清晰度 | 8.8/10 | 20 页内将表示、条件、数据和自回归生成讲得紧凑，附录包含明确失败模式。 |
| 实用/研究价值 | 8.6/10 | 代码与权重已发布，单图到 metric mesh 很有价值；无纹理、非 watertight 与非商业许可限制落地。 |

**综合建议：8.5/10，值得细读。** 最值得复用的不是特定 backbone，而是“按 perspective 分配 3D 生成预算”和“显式编码 free / observed / unknown”的问题建模。

## 阅读结论

- **最值得记住的点**：单图 3D 生成不应把整条 camera ray 都当成同一种未知；surface 前必须为空、surface 上需重建、surface 后才应生成。
- **最需要怀疑的点**：野外背面与遮挡区看起来合理，并不等价于真实；现有指标主要证明输入一致性和整体几何，不足以验证 hallucinated geometry。
- **最值得复现的点**：visibility residual 的 outdoor F1 增益、adaptive chunks 的质量–时间优势，以及它们对 MoGe-3 几何误差的敏感性。

## 相关论文与资源

- [官方代码](https://github.com/Applied-Intuition-Open-Source/BuildRome)：完整 inference pipeline、配置与安装说明。
- [官方 checkpoints](https://huggingface.co/AppliedIntuitionResearch/BuildRome)：SS/SLat LoRA 与 conditioning deltas。
- [TRELLIS 2](https://arxiv.org/abs/2512.14692)：本文继承的对象级 O-Voxel 两阶段生成器。
- [GenRecon](https://arxiv.org/abs/2605.23888)：固定大小 overlapping chunks 的生成式场景重建 baseline。
- [MoGe](https://github.com/microsoft/MoGe)：本文推理时 point map、metric scale 和 camera estimation 的关键外部依赖。
