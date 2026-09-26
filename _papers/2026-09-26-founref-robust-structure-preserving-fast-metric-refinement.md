---
title: "FounRef: Robust, Structure-Preserving, and Fast Metric Refinement of Frozen Monocular Foundation Priors with Sparse Anchors"
subtitle: "以稀疏米制锚点校准冻结单目 foundation prior，并用两阶段双边求解在不把 LiDAR 扫描纹理刻进表面的条件下获得稠密米制深度"
authors: "Dan Halperin, Mirko Mühlisch"
venue: "arXiv preprint"
year: 2026
date: 2026-09-26
paper_url: "https://arxiv.org/abs/2609.29224"
tags:
  - Metric Depth
  - Depth Completion
  - Monocular Foundation Models
  - Sparse LiDAR Anchors
  - Bilateral Solver
summary: "FounRef 是训练无关的稠密米制深度细化器：先在 log-depth 中将冻结的单目 foundation prior 对齐到稀疏 LiDAR anchors，再用轻量双边求解生成局部参考面过滤错投影 anchors，最后以保边的双边 correction field 修正 prior。在五个数据集的同锚点协议下，它在 OOD 上兼顾较低深度误差、低法线噪声与 78 ms 延迟，避免深度补全网络常见的 anchor imprinting。"
permalink: /papers/founref-robust-structure-preserving-fast-metric-refinement/
---

> **阅读依据**：arXiv:2609.29224（用户提供链接的官方 PDF）。论文未在正文给出可验证的代码/项目链接，因此不在此补充未核实链接。除明确标注的“我的分析”外，方法与数字均来自论文。

## 一句话总结

FounRef 将单目 foundation model 的稠密几何当作冻结的结构先验、将稀疏 LiDAR 当作米制锚点：通过全局标定、锚点一致性过滤和双边网格上的结构保持校正，把“有几何、没可靠尺度”的单目深度转为可泛化的稠密米制深度，而不需要为新相机或新数据分布重新训练深度补全网络。

## 背景与问题

单目 foundation priors（如 MoGe-2、Metric3D、Depth Anything、DepthPro）通常有锐利边界与合理相对结构，却在陌生相机、长距离和域外场景中存在尺度/偏移误差。学习式 depth completion 虽能从 RGB 加稀疏 LiDAR 输出米制深度，但依赖训练分布和传感器稀疏模式；面对新 rig、投影错位、移动物体或时序累计点云时，往往把错误锚点硬传播到局部表面。

论文的目标不只是减小点误差，而是同时满足：

- **米制准确性**：稀疏观测应矫正尺度与局部深度；
- **结构保真**：连续表面保持平滑，边界不能被锚点扫描纹理打碎；
- **鲁棒与快速**：不用重新训练，也能处理 cross-sensor/out-of-domain anchors。

## 方法详解

### 整体流程

<div class="mermaid">
flowchart LR
    A[RGB 图像] --> B[冻结单目 prior: D_prior 与有效 mask]
    C[稀疏 LiDAR anchors A] --> D[截断深度、划分拟合集/留出集]
    B --> E[全局 log-depth 单调标定 D_cal]
    D --> E
    E --> F[轻量 bilateral solve: 局部参考面 D_1]
    D --> G[以 D_1 验证/过滤 anchors]
    F --> G
    E --> H[全量 bilateral solve]
    G --> H
    H --> I[稠密米制深度 D_2]
</div>

输入是 RGB、冻结 prior 输出的 $D_{\mathrm{prior}}$ 及其有效像素 mask，以及投影到图像的锚点 $A=\{(u_i,z_i)\}$。锚点深度先截断到 $D_{\max}=50$ m，之后按 80/20 划为拟合集与留出集；留出集既避免滤波器只“自证”拟合点，也用于评估。

### 1. 全局单调标定：先修尺度，不重画几何

在 log-depth 中，乘性尺度变成加性偏移。令 $t_i=\log D_{\mathrm{prior}}(u_i)$、$y_i=\log z_i$，论文以 Huber 鲁棒损失拟合 $y\approx f(t)$，并将 anchors 按深度分成 $K=24$ 个 bin，拟合每个 bin 的平均残差、做单调且阻尼的插值：

$$
\log D_{\mathrm{cal}}(x)=f\!\left(\log D_{\mathrm{prior}}(x)\right)
+c\!\left(\log D_{\mathrm{prior}}(x)\right).
$$

这里 $f$ 给出全局尺度/偏置，$c$ 是随 prior 深度变化的一维小修正；锚点过少时只使用 $f$，完全没有锚点则保留原 prior。这个设计刻意只沿 depth value 修正，不直接把锚点局部形状写入整张图，因此首先保住了单目模型学到的表面布局和边缘。

### 2. 锚点验证：以局部参考面筛掉“看似有效”的错点

几何滤波只能处理部分投影遮挡；跨传感器标定误差、移动物体和时间不同步仍会制造错 anchor。FounRef 先在拟合集上运行一次较轻的局部双边求解得到参考深度 $D_1$，再对每个锚点检验：

$$
\left|\log z_i-\log D_1(u_i)\right|<\tau.
$$

通过者组成 $A^+$，未通过者删除。该检验在相对误差空间而非绝对米误差空间进行，避免远处点天然有更大绝对残差就被不公平地拒绝。论文默认 KITTI 使用 $\tau=0.45$，更噪的 nuScenes 使用 $\tau=0.2$。

### 3. 结构保持局部 solver：只传播校正，不覆盖 prior

最终输出采用 log-depth 中的加性 correction field $b(x)$：

$$
\log D_2(x)=\log D_{\mathrm{cal}}(x)+b(x).
$$

在保留锚点上，所需校正为 $t_i=\log z_i-\log D_{\mathrm{cal}}(u_i)$。在双边网格中求解：

$$
\min_b\sum_i w_i\left(b(u_i)-t_i\right)^2+\lambda b^\top Lb,
$$

其中 $L$ 是 bilateral Laplacian。两像素只有在图像坐标接近且 prior depth 相近时才耦合：校正可沿同一预测表面传播，但在深度不连续处停止。最终系统为稀疏正定线性系统，以 Jacobi 预条件共轭梯度求解；半分辨率求解，默认 cell size $\sigma_s=16$。

这与“直接做 depth completion”有本质差别：FounRef 不让锚点决定表面形状，只让它们对已有 prior 施加平滑的米制修正。因此低 RMSE 不以 LiDAR scan pattern 印到深度图为代价。

## 实验关键数据

### 设置与指标

实验覆盖 KITTI、nuScenes、Waymo、GOOSE 与 NYU Depth V2；前两类分别包含带 dense GT 的标准集和来自不同采集/传感器条件的域外道路与非结构化场景。每种方法使用相同 anchors，默认每个验证集 500 帧；KITTI/NYU 以 dense GT 评测，其他数据集留出累计点云的 20% 作目标。

除 RMSE（m，低为好）与 30–50 m 区间 far-RMSE 外，论文特别报告 surface normal dispersion（度，低为好），用于衡量平坦表面是否被局部噪声破坏；并报告 A100 的均值延迟。

### 主结果：精度、结构与效率需要一起看

| 方法 | KITTI RMSE | nuScenes RMSE | Waymo RMSE | GOOSE RMSE | NYU RMSE | 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **FounRef** | 0.726 | 0.687 | 0.745 | 1.687 | 0.174 | **78 ms** |
| OMNI-DC | 1.137 | **0.353** | **0.590** | **0.702** | **0.099** | 1011 ms |
| Prior-Depth-Anything | 1.583 | 0.661 | 0.833 | 0.742 | 0.135 | 1621 ms |
| DMD3C | **0.677** | 0.523 | 0.774 | 0.805 | 1.205 | 1219 ms |

表中是论文 Table 1 的跨五数据集同锚点协议。FounRef 并非所有数据集 RMSE 第一，但在 OOD 的 nuScenes/Waymo/GOOSE 上，论文同时报告其 normal dispersion 为 **3.5° / 1.7° / 8.8°**，明显低于不少“点误差更低”的补全网络。摘要概括为 OOD 数据上最多 24% 更低深度误差、92% 更低表面法线噪声，并比 DMD3C 近 15 倍快。

这一差异并非装饰性指标：论文可视化显示，部分 completion 网络会将 LiDAR 扫描线固化进道路和墙面。比如一个 KITTI 帧中 DMD3C 与 BP-Net 的 point RMSE 可低于 FounRef，却具有后者约 7–8 倍的 normal error；FounRef 的设计目标正是避免该“更准但更破”的解。

### Solver 与鲁棒性证据

在 KITTI 累计稠密深度的 solver 对照中，FounRef 取得 RMSE 1.050、far-RMSE 2.583、法线 dispersion **1.77°**、anchor imprint **0.38**、14 ms；Screened Poisson 的 RMSE 0.880 更低，但 dispersion 4.75° 且 imprint 5.25；TGV$^2$-$L_2$ 需约 300 ms。结果说明 FounRef 选择了明确的 accuracy–structure–speed 折中，而不是单指标最优。

锚点密度从完整累计云降至 1% 时，FounRef 的 RMSE 在五数据集上只增加约 $1.4\times$–$2.1\times$；DMD3C 则增加 $2.9\times$–$12.9\times$。作者将此归因于 prior 提供几何、anchors 只固定米制尺度；学习式回归器更依赖输入点本身。

锚点扰动实验表明：5% 乘性深度噪声使各数据集误差仅上升约 3% 或更少；但投影偏移更危险，KITTI 上 5 px shift 使 RMSE 从 0.832 增至 1.624（约 +95%）。因此 FounRef 能过滤部分不一致 anchor，却不能替代传感器外参和时间同步。

## 亮点与局限

**论文亮点**：

- 将 foundation prior、任意稀疏 metric anchor 源与 refinement solver 解耦；模型或传感器替换不要求重新训练。
- 两阶段求解的目的明确：第一次只构造可靠的局部表面来检测 anchor disagreement，第二次才使用过滤后的点输出结果。
- 以 normal dispersion 与 anchor imprint 补足 RMSE；这比只强调完成精度更贴近 dense depth 用于重建、投影和导航时真正的几何质量。
- 半分辨率双边网格/共轭梯度实现使其在 A100 上 78 ms，远快于表中多种网络基线。

**论文明确的局限**：当 prior 在某处无效时，输出只能填到最大深度 cap，不能凭 sparse anchors 重新生成缺失几何；方法也依赖足以提供几何结构的单目 prior。

**我的分析**：

- 二阶段过滤会把“prior 与真实世界同时错”的锚点当作异常值，故对极端 domain shift 的下界取决于 prior，而不是 solver；应报告不同 prior 的 failure correlation，而不只报告最适配的 MoGe-2。
- Table 1 中 OMNI-DC 在若干数据集的 RMSE 很低，说明 FounRef 的价值主要是跨域结构保真与效率，而非统一的点误差 SOTA。实际部署应按下游任务选择权重，而不应只用摘要的百分比结论。
- anchor filter 对 5 px 错投影十分敏感；给出 online extrinsic refinement、时间偏移估计，或把 anchor 像素位置也作为可优化变量，会是很自然的下一步。

## 与相关工作对比

| 路线 | 如何利用稀疏深度 | 主要风险 | FounRef 的差别 |
| --- | --- | --- | --- |
| DMD3C / OMNI-DC / NLSPN | 学习从 RGB 与稀疏点回归稠密深度 | 训练域、sensor pattern 和错误点传播 | 训练无关，可直接接新 prior/anchor，但不承诺每集最低 RMSE |
| Screened Poisson / TGV$^2$ | 显式正则化的深度/校正传播 | 跨边界泄漏、慢或结构噪声 | 双边网格以 prior depth 为边界线索，兼顾 14 ms solver 时间 |
| RePLA(y) | 用几何规则过滤 LiDAR 投影遮挡 | 不覆盖一般的跨模态与时序不一致 | 以局部参考面检验 log-depth disagreement，覆盖范围更广 |
| **FounRef** | prior 标定 + 过滤 anchors + surface-aware correction | 上限受 prior、标定和同步质量约束 | 将米制信息与表面几何解耦 |

## 评分

| 维度 | /10 | 依据 |
| --- | ---: | --- |
| 创新性 | 8.1 | 组件未必全新，但“冻结 foundation prior + 可信 anchors + 两阶段结构约束”的分工非常清晰。 |
| 技术可靠性 | 8.2 | 公式、运行点、异常锚点及输入稀疏度均有实证；prior 失效时仍受限。 |
| 实验充分度 | 8.5 | 五数据集、多个先验/solver、密度与扰动消融，并额外测量结构指标。 |
| 写作清晰度 | 8.4 | 明确区分 accuracy、structure、runtime；算法流程可复现性强。 |
| 实用/研究价值 | 8.6 | 适合需要快速把任意单目 prior 接到 LiDAR/ToF 等 sparse metric sensor 的系统。 |

**总体推荐：值得细读。** 它给出的通用经验是：对稀疏度量观测，不应让其替代稠密视觉先验的几何；应先判断观测是否可信，再仅传播“米制校正”。

## 阅读结论

- **最值得记住的点**：将 correction field 建在 log-depth 的双边网格上，能沿 prior 表面传播尺度信息、在边界处止步。
- **最需要怀疑的点**：anchor 与 prior 的不一致既可能来自错误 anchor，也可能来自 prior 的系统性错误；过滤会偏向后者。
- **最值得复现或继续验证的点**：在已知外参漂移/时间偏移的数据上，联合比较 anchor 位置校正、软权重而非硬过滤，以及下游 3D reconstruction 指标。

## 相关论文

- [MoGe-2](https://arxiv.org/abs/2504.19154) — FounRef 默认使用的冻结单目几何 prior。
- [DMD3C](https://arxiv.org/abs/2507.10822) — 高精度但需要学习式深度补全的主要效率对照。
- [OMNI-DC](https://arxiv.org/abs/2508.01813) — 面向跨数据集稀疏深度模式的 zero-shot completion 对照。
- [Fast Bilateral Solver](https://arxiv.org/abs/1511.03296) — FounRef 所用双边网格结构保持传播的基础。
