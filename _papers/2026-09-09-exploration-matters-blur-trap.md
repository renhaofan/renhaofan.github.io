---
title: "Exploration Matters for Escaping the Blur Trap in 3D Gaussian Splatting"
subtitle: "诊断 3DGS 的深度梯度缺失与遮挡梯度衰减，并以随机播种和随机分裂补回显式探索"
authors: "Chengbo Wang, Guozheng Ma, Jinhong Wu, Tie Ji, Yizhen Lao"
venue: "arXiv preprint"
year: 2026
date: 2026-09-09
paper_url: "https://arxiv.org/abs/2607.17965v1"
code_url: "https://github.com/Chengbo-Wang/Exploration-for-GS"
project_url: "https://chengbo-wang.github.io/ExploreGS/"
tags:
  - 3D Gaussian Splatting
  - Optimization
  - Exploration
  - Densification
  - Gradient Analysis
  - Novel View Synthesis
summary: "本文将 3DGS 中远景和遮挡区域长期模糊的现象归纳为 Blur Trap：主导位置更新的 2D 投影梯度对当前视线严格正交，难以探索深度；alpha blending 又削弱后排 Gaussian 的梯度，使其无法触发 densification。作者分别用 Random Seeding 在场景包围盒内探测新深度、用 Random Splitting 绕开梯度阈值主动细分大 Gaussian，在 DL3DV 上将 3DGS 从 27.16 提升到 28.43 dB、LPIPS 从 0.167 降到 0.124，但代码默认参数与论文存在差异，且训练成本、随机方差与若干实现混杂因素未充分报告。"
permalink: /papers/exploration-matters-blur-trap/
---

> **阅读依据**：本笔记基于 [arXiv:2607.17965v1](https://arxiv.org/abs/2607.17965v1)（2026-07-20）的完整官方 HTML、PDF、TeX source、[项目页](https://chengbo-wang.github.io/ExploreGS/)与截至 2026-09-09 的[官方代码](https://github.com/Chengbo-Wang/Exploration-for-GS)。论文当前标注为 **arXiv preprint**，作者来自 Hunan University 与 Nanyang Technological University，未给出正式会议或期刊信息。
>
> **资源状态**：官方仓库已发布基于原版 3DGS 的 exploration code；README 中 `gsplat` 版本、论文使用的 Blur Trap dataset 仍标为未发布，4DGS exploration implementation 也未出现在当前仓库。仓库根目录未见 license 文件，复用前应额外确认许可。
>
> **复现边界**：代码审查发现当前 `base.yaml` 与论文 Section 5.1 的默认 seeding 数量不一致，并且 seeding mode 同时修改了标准 3DGS 的 world-space oversized-Gaussian pruning threshold。下文将论文机制、实验结果和当前代码行为分开描述。

## 一句话总结

论文把 3DGS 的模糊停滞重新解释为缺少 exploration 的优化陷阱：2D projection branch 几乎支配位置梯度，却无法沿单视图 viewing ray 移动 Gaussian；遮挡又通过 transmittance 压低后排 primitives 的 densification signal。作者不修改 renderer 或 loss，而是加入两个极简、与梯度幅值解耦的操作——Random Seeding 负责全局深度试探，Random Splitting 负责局部结构细化——再交回原有梯度下降和 opacity pruning 完成 exploitation。

## 背景与动机

- **现象**：标准 3DGS 即使有较多 training views，远处山脉、天空、城市背景和被前景遮挡的近处细节仍可能长期模糊。
- **常见解释不完整**：这些失败通常被归因于相机不足、初始化差、densification threshold 不合适或表示容量不足，但论文认为更深层原因是 gradient geometry 本身存在方向和可见性偏置。
- **3DGS 的 optimization 特征**：训练通常每步采一个 view，直接通过物理投影与 alpha compositing 回传梯度；相比神经网络，没有 dropout、parameter noise 或明显的 minibatch stochasticity 来主动探测当前解以外的空间。
- **作者观点**：标准流程几乎只有 exploitation——已有 Gaussian 依照当前可见的 reprojection error 做局部改进，densification 也必须先等 gradient 超阈值；一旦正确结构位于 gradient 不可达或被遮挡抑制的位置，系统便会自我锁死。

论文将这种持续的局部次优状态命名为 **Blur Trap**，并分成：

- **Far-Side Blur Trap**：远处内容缺少沿视线方向的有效位置探索；
- **Near-Side Blur Trap**：某些近处物体在部分视角被遮挡，累积 2D gradient 太弱，无法触发 split。

**核心研究问题：如果问题确实来自“纯梯度 exploitation”，是否只加入极少量、机制对应的随机 exploration，就足以恢复被困住的结构？**

## 核心问题

1. 3D Gaussian 的位置梯度究竟由哪些 rendering branches 构成，哪一项实际支配更新？
2. 为什么 screen-space supervision 无法稳定校正远景的 depth？多视角什么时候能补救，什么时候仍不够？
3. 为什么被遮挡的 Gaussian 不只是学习慢，而会连 densification 资格都失去？
4. 如何绕过上述限制，又不设计新网络、复杂 uncertainty model 或大幅修改原始 3DGS？
5. 改善是否只是来自“加了更多 Gaussians”，还是来自更有效的 primitive allocation？

## 方法详解

### 整体框架

<div class="mermaid">
flowchart TB
    A[COLMAP 初始化的 3D Gaussians] --> B[标准 differentiable rendering]
    B --> C[Photometric loss 与标准 gradient update]
    C --> D[原生 gradient-based clone / split / prune]

    E{Exploration schedule} --> F[Random Seeding]
    E --> G[Random Splitting]
    F --> H[在当前 Gaussian AABB 内均匀生成候选点]
    H --> I[有效 seeds 被 loss 优化与 densify]
    H --> J[无效 seeds 被 opacity pruning 删除]

    G --> K[按 mean scale 加权随机选择 Gaussians]
    K --> L[不检查 accumulated 2D gradient，直接 split]

    I --> B
    J --> B
    L --> B
    D --> B

    B --> M[推理：仍为标准 3DGS renderer]
</div>

两项 exploration 都只在训练期改变 Gaussian 集合：不引入额外 inference module，不修改前向 alpha blending，也不替代原生 ADC。区别在于：

- Random Seeding 改变**空间覆盖范围**，让优化有机会落到不同 depth basin；
- Random Splitting 改变**容量分配门槛**，让 gradient 被遮挡压低的区域仍有机会变细。

### 1. 位置梯度的三条分支

论文将 Gaussian 三维位置的总梯度写为：

$$
\mathbf g_{\mathrm{all}}
=\mathbf g_{\mathrm{2d}}
+\mathbf g_{\mathrm{cov2d}}
+\mathbf g_{\mathrm{sh}}.
$$

- $\mathbf g_{\mathrm{2d}}$：由 projected 2D mean 对位置的导数产生；
- $\mathbf g_{\mathrm{cov2d}}$：位置影响 projected covariance 后产生；
- $\mathbf g_{\mathrm{sh}}$：位置改变 view direction、进而影响 spherical harmonics color 后产生。

作者的 gradient profiling 显示，$\|\mathbf g_{\mathrm{2d}}\|$ 在训练全程几乎等于 $\|\mathbf g_{\mathrm{all}}\|$，另外两项低约 2–3 个数量级。因此虽然总梯度形式上包含几何和外观的多条路径，实际位置更新近似退化为 screen-space mean reprojection optimization。

Table 1 的分支消融支持这一点：

| 保留的位置梯度分支 | Mip360 PSNR | T\&T PSNR | Deep Blending PSNR |
| --- | ---: | ---: | ---: |
| Cov + 2D + SH（完整） | 27.52 | 23.73 | 29.80 |
| 2D + SH | 27.52 | 23.77 | 29.85 |
| Cov + 2D | 27.53 | 23.73 | 29.78 |
| 仅 2D | **27.54** | **23.84** | **29.80** |
| Cov + SH（去掉 2D） | 24.27 | 21.59 | 24.92 |
| 仅 Cov | 24.18 | 21.54 | 24.88 |
| 仅 SH | 25.23 | 22.45 | 28.12 |

去掉 covariance 或 SH branch 几乎不影响结果，去掉 2D branch 则大幅退化；甚至“仅 2D”在三组 PSNR 上都不低于完整梯度。能够支持的结论是：**原版设置下，2D mean branch 对位置优化占绝对主导**。它并不证明 covariance/SH 对所有 renderer、loss、数据和训练阶段都没有意义。

### 2. 为什么单视图 2D 梯度没有 viewing-ray 分量

设 camera center 为 $\mathbf P_{\mathrm{cam}}$，Gaussian center 为 $\mathbf P_{3D}$。Perspective projection 只取决于这条 ray 的方向；沿同一 ray 前后移动点，不改变其 projected 2D mean。因此 projection Jacobian 的 null space 包含 ray direction，论文证明：

$$
\frac{\partial L^{2D}}{\partial\mathbf P_{3D}}
\cdot
(\mathbf P_{3D}-\mathbf P_{\mathrm{cam}})=0.
$$

也就是说，对某一个训练相机，$\mathbf g_{\mathrm{2d}}$ 严格位于与 camera-to-Gaussian ray 正交的切平面内。优化能修正“投影到图像的横向位置”，但不能仅靠这一分支判断点应在这条 ray 的前端还是后端。

多视角可以通过不同 ray 的交会间接恢复 depth：每个 view 提供一个不同的正交平面，累计方向可能张成 3D。但当远景相对 camera baseline 很大、各视角 ray 近乎平行时，这些平面的组合仍对 depth 病态，Far-Side Blur Trap 因而集中在 mountains、clouds、skyline 等区域。

**独立判断**：论文有时把“缺少 depth gradient”写成 3DGS 的绝对内在缺陷，表述偏强。严格成立的是**单 view 的 2D-mean branch 对该 view ray 无梯度**；在视角分布充分宽时，多视角联合仍可提供 depth constraint。真正的问题是 narrow-baseline / far-field condition number，而非任何场景都完全没有 depth information。

### 3. Alpha blending 如何压低 densification signal

前向颜色为：

$$
C_p=\sum_{i=1}^{N}c_i\alpha_iT_i,
\qquad
T_i=\prod_{j=1}^{i-1}(1-\alpha_j).
$$

按 front-to-back 排序后，越靠后的 Gaussian，其 transmittance $T_i$ 越小。反向传播到其 opacity、颜色及 screen-space position 的信号也被前景累计透明度压低。Supplement 将各项拆开后给出一个由前序 transmittance 控制的 gradient bound；随着 rendering order 增大，上界向零收缩。

这不仅影响连续参数更新，还影响离散 ADC：原版 3DGS 会周期累计 view-space position gradient，只有统计量超过 $\tau_{\mathrm{split}}$ 才 clone/split。被遮挡的 primitive 在多个 views 中反复收到弱梯度，平均值长期过不了门槛，最终保持为覆盖大范围的粗 Gaussian，形成 Near-Side Blur Trap。

需要注意，“near-side”不是指它在所有相机中都在最前面；论文指的是物理上较近或前景结构，在某些 views 中仍会被其他内容遮挡。只要其细节恢复依赖的 views 恰好让它位于 blending sequence 后部，就可能失去 densification signal。

### 4. Random Seeding：用全局覆盖补 depth exploration

论文方法是在 densification period 中随机采样 $N_{\mathrm{seed}}$ 个 3D positions，加入新的 Gaussian candidates。其意义不是改变 gradient 的方向，而是直接把 primitive 放进原有点永远无法沿 ray 抵达的 depth intervals：

- 落到空区域的 seeds 无法稳定降低 photometric loss，作者期望它们被原生 opacity pruning 清除；
- 落到真实结构附近的 seeds 可由标准 2D gradient 做局部 planar refinement，并进一步触发 ADC；
- exploration 负责跳到新 basin，原始 gradient 负责 basin 内 exploitation。

论文 Section 5.1 写的是在“all existing Gaussians 的 axis-aligned minimum bounding box”内 uniform sampling，$N_{\mathrm{seed}}=20$。当前官方代码的关键细节则是：

- `base.yaml`：每 100 iterations 注入 **100** 个 seeds，500 iteration 后开始，15K 后停止；
- seed positions 在当前 Gaussian AABB 内均匀采样，不扩展到 bbox 外；
- SH DC color 随机初始化；其余 SH 为零；
- scale 按 Gaussian 到坐标原点的距离乘以全局 mean scale/distance ratio 初始化；
- opacity 由该 batch 的 position norm 线性设到约 $0.2$–$0.7$，距离原点更远的 seed 初值更高。

因此 Random Seeding 更准确地说是在**当前场景包围范围内重新探测未充分覆盖的 3D 位置**，而不是无限探索未知空间。若 SfM 初始化的 AABB 本身不覆盖真实远景 depth，默认实现也无法到达 bbox 外。

### 5. Random Splitting：绕开 gradient gate

标准 3DGS split 要求 accumulated view-space gradient 超阈值。Random Splitting 取消这一前置条件，直接从当前 Gaussians 中随机选少量大 primitives 做标准 subdivision。

代码并非硬 threshold 后均匀选择，而是令 sampling probability 与 mean scale 成正比：

$$
p_i=\frac{\operatorname{mean}(s_i)}{\sum_k\operatorname{mean}(s_k)}.
$$

每次默认选 $N_{\mathrm{split}}=20$ 个 parent；对每个 parent，从其旋转后的 Gaussian support 随机生成 $N=2$ 个 children，将 child scale 除以 $0.8N$，复制 rotation、SH 与 opacity，然后删除 parent。单次净增加约 $N_{\mathrm{split}}$ 个 primitives。

`base.yaml` 从 iteration 501 到 14999 **每一步**执行一次 split，理论上在 pruning 前最多产生约 29 万净新增 primitives。它不检测 occlusion，也不只选择论文语义上的 near-side region；“targeted”来自 large scale 与 blur failure 的相关性，而非显式 visibility reasoning。

这种设计的优点是简单且直接绕过 suppressed gradient；缺点是仍可能拆到大而正确、未遮挡的 Gaussians。更精细的版本可以结合 multi-view transmittance、visibility variance 或 residual uncertainty 来分配 exploration budget。

### 6. 与原生训练循环的关系

官方实现保留原版 densify-and-prune，顺序为：

1. rendering、loss backward；
2. 原生 ADC statistics 与周期 densify/prune；
3. optimizer step；
4. `explore_step` 执行 seeding/splitting；
5. 新 primitives 在后续 iterations 接受正常 gradient optimization。

Random operations 在 `torch.no_grad()` 环境中运行，不构造新 loss，也不需要修改 CUDA rasterizer。推理只加载最终 Gaussian cloud，速度取决于最终 primitive count，而没有额外 exploration module。

### 7. 论文与当前代码的复现差异

| 项目 | 论文正文 | 当前官方仓库 `base.yaml` / code |
| --- | --- | --- |
| Random Seeding 数量 | $N_{seed}=20$ | `num: 100` |
| Seeding 频率 | each densification iteration（原版通常每 100 步） | `interval: 100` |
| Random Splitting | $N_{split}=20$ per iteration | `num: 20, interval: 1` |
| Exploration 截止 | 正文未在方法段完整列出 | `until: 15000` |
| oversized world-space pruning | 未说明更改 | seed enabled 时从 baseline `0.1×extent` 改为 `10×extent` |
| 不传 `--explore_cfg` | 未说明 | 两项都默认 `num=100, interval=100`，不同于推荐 `base` |

最重要的混杂是 `densify_and_prune()`：只要启用 seeding，所有 Gaussians 的 world-space large-point threshold 就从 `0.1 * extent` 放宽到 `10.0 * extent`，不仅影响新 seeds，也影响原有 primitives 的 survival。论文把效果归因于 stochastic seeding，但当前代码没有提供“Random Seeding + 原阈值”对照，无法完全隔离两个因素。

## 实验关键数据

### 实验设置

- **Mip-NeRF 360**：室内/室外 unbounded scenes；
- **Tanks & Temples（T&T）**；
- **Deep Blending（DB）**；
- **OMMO**：large-scale outdoor NVS；
- **DL3DV Benchmark**：随机选 16 个 scene IDs，使用 $1/4$ resolution；
- **Neu3D**：用于 4DGS 动态实验。

指标为 PSNR/SSIM（越高越好）、LPIPS（越低越好）和 Gaussian count。主要静态 baseline 是作者统一重跑的 3DGS 与训练 50K iterations 的 HoGS；论文未报告多随机种子均值、方差、training time 或 peak VRAM。

### 跨静态数据集结果

| Dataset | 3DGS PSNR / SSIM / LPIPS | Seed only | Split only | Seed + Split |
| --- | ---: | ---: | ---: | ---: |
| Mip360 | 27.52 / .816 / .215 | 27.68 / .819 / .214 | 27.95 / .829 / **.192** | **27.96** / **.829** / .195 |
| T&T | 23.73 / .853 / .169 | 24.30 / .861 / .163 | 24.30 / .868 / **.139** | **24.37** / **.870** / **.139** |
| DB | 29.80 / .907 / **.238** | 29.78 / .906 / **.237** | **30.01** / **.908** / .248 | 29.98 / .907 / .249 |
| OMMO | 30.49 / .920 / .142 | 30.85 / .924 / .131 | **31.29** / .928 / .121 | 31.27 / **.930** / **.119** |
| DL3DV | 27.16 / .860 / .167 | 27.74 / .869 / .155 | **28.47** / .889 / .125 | 28.43 / **.891** / **.124** |

主要结论：

- 联合方法相对 3DGS 在 Mip360、T&T、OMMO、DL3DV 分别提升 **+0.44、+0.64、+0.78、+1.27 dB**；DL3DV LPIPS 从 .167 降到 .124，下降约 **25.7%**。
- Split 是多数数据集的主要贡献者：Mip360、OMMO、DL3DV 的 PSNR 几乎都由 Split 提供；这说明 standard densification gate 的覆盖问题可能比论文标题强调的 depth issue 更普遍。
- Seed 在 T&T、OMMO 与 DL3DV 有稳定收益，但 Mip360 仅 +0.16 dB，DB 反而 -0.02 dB；“consistently improves PSNR”若逐数据集理解并不完全成立。
- 联合并非所有指标最优：Mip360 的最佳 LPIPS 是 Split；DB 的最佳三项组合来自不同方法，Seed/联合 PSNR 不及 Split，联合 LPIPS 比 baseline 恶化 .011；OMMO 与 DL3DV 的最佳 PSNR也都是 Split，而不是联合。
- 与 HoGS 50K 比，联合方法大多数设置更强，但 DB 上 HoGS LPIPS .244 优于联合 .249；论文没有统一 wall-clock，因此不能判断更少 iterations 还是额外 exploration 更划算。

### Gaussian 数量不是统一下降

| Dataset | 3DGS | Seed | Split | Seed + Split | 联合相对 3DGS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mip360 | 2.72M | 2.62M | 2.58M | 2.53M | -7.0% |
| T&T | 1.57M | 1.51M | 2.13M | 2.11M | **+34.4%** |
| DB | 2.48M | 2.33M | .86M | .79M | **-68.1%** |
| OMMO | 1.78M | 1.68M | 1.77M | 1.77M | -0.6% |
| DL3DV | 1.14M | 1.11M | 2.13M | 2.11M | **+85.1%** |

- DB 的 Split 数量减少约 65.3%，联合减少约 68.1%，同时 PSNR 略升；这是最有说服力的“更有效 allocation”证据。
- 但 T&T 与 DL3DV 的联合 primitive count 分别增加 34% 和 85%。因此“maintaining or reducing model complexity”不是跨数据集普遍事实，更准确的是：探索改变了容量分布，在复杂数据上明显用更多点换质量。
- 论文不报告 training time、最终 FPS 或显存；DL3DV 近乎翻倍的 #GS 很可能增加训练、存储与推理成本，不能称为已证明的 negligible overhead。

### 4DGS / Neu3D

| Method | PSNR | SSIM | MS-SSIM | D-SSIM ↓ | LPIPS-VGG ↓ | LPIPS-Alex ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4DGS | 30.575 | .9314 | .9659 | .0171 | .1507 | .0602 |
| + Split | 30.820 | .9369 | .9689 | .0156 | .1380 | .0488 |
| + Split & Seed | **31.085** | **.9379** | **.9702** | **.0149** | **.1377** | **.0483** |

- 联合方法 PSNR +0.51 dB；LPIPS-VGG 下降约 8.6%，LPIPS-Alex 下降约 19.8%。
- Split only 已获得大部分感知质量收益，符合 Neu3D depth range 相对有限、主要问题来自动态遮挡的解释。
- 论文未报告 4D Gaussian 数、训练时间、per-scene breakdown 或运行方差；当前官方仓库也未发布这部分代码。

### Pseudo-depth 与 Random Seeding

作者用 Depth Anything V2 pseudo-depth 加入 $\mathcal L_D$，试图显式补 depth supervision。

- 仅给 3DGS 加 depth loss：T&T 23.73→24.06、OMMO 30.49→30.59、DL3DV 27.16→27.33 dB，说明 depth cue 确实有帮助。
- 不加 depth loss的 Seed：24.30 / 30.85 / 27.74 dB，三组都强于 depth-only baseline。
- 联合 Seed+Split 加 depth 后为 24.38 / 31.06 / 28.39；不加 depth 时为 24.37 / 31.27 / 28.43。T&T 略升，但 OMMO、DL3DV 下降。

论文据此认为不准确的 deterministic depth prior 会限制 exploration。这个解释是合理假设，但表格并不能唯一证明“depth loss 主动 suppress exploration”：也可能是 loss weighting、pseudo-depth scale/shift、边缘误差或 hyperparameter 未重新调优。缺少 exploration survival rate、depth distribution 与梯度冲突统计。

### Random Splitting 强度消融

论文比较 $N_{split}\in\{20,50,100,200\}$ 与降低原生 $\tau_{split}$：简单降低 threshold 会制造大量冗余 primitives，却不产生同等质量收益；scale-weighted random split 在较小 point budget 下表现更好。

该消融支持“不是所有 densification 都等价”，但仍有两点边界：

- lowering threshold 是较弱 baseline，尚未与 error-map、visibility、opacity、Fisher information 或 MCMC relocation 等现代 allocation 策略做同预算比较；
- 图中没有提供可复制的完整数值表和 seed variance，本笔记不从曲线估读精确分数。

### Positional perturbation 附加实验

Appendix 将 3DGS-MCMC 的 position perturbation 限制到 viewing-ray direction，也能改善 unbounded scene。它进一步支持“缺少 depth exploration”这一机制判断。

但正文没有给出该方法的 quantitative table，也没有与 Random Seeding 在相同 perturbation/primitive budget 下比较。它更像机制佐证，而不是证明 Seeding 是最佳 depth exploration operator。

### 关键发现

1. 原版 3DGS 位置更新在实验中几乎由 projected 2D mean branch 支配；单相机下该分支严格没有 viewing-ray component。
2. 远景 depth condition 不良与 alpha-blending 后排 gradient attenuation，是两类可区分的 optimization bottleneck。
3. 极简单的随机容量注入即可显著改善结果，尤其 Random Splitting 在多数 benchmark 上贡献最大。
4. 联合方法在 DL3DV、OMMO 与 T&T 有强收益，但不是所有数据/指标都最好；DB 的 LPIPS 是明确反例。
5. 成本并非总是 negligible：DL3DV #GS +85%，T&T +34%；论文缺 wall-clock、VRAM 与 FPS 数据。
6. 当前代码与论文默认 seeding 数量及 pruning 行为不一致，直接复现前必须固定 config 并补做 threshold 对照。

## 亮点与洞察

### 论文亮点

- **先诊断 gradient geometry，再设计操作**：Seeding 与 Splitting 分别对应 ray-nullspace 与 blending attenuation，不是无目的地加随机噪声。
- **方法极简且 renderer-agnostic**：不改 CUDA forward/backward，不加网络，探索结束后仍是普通 Gaussian cloud。
- **用分支消融支撑分析**：证明实际优化由 2D branch 支配，比只展示 blur case 更有说服力。
- **覆盖多种场景尺度**：从 Mip360/DB 到 OMMO/DL3DV，再到动态 4DGS，说明问题不局限于单一 benchmark。
- **揭示 ADC 的双重依赖**：同一个 screen-space gradient 既负责移动，也负责决定是否增加容量；一旦它偏置，连续优化与离散结构增长会同时失败。

### 我的洞察

- **个人分析：Random Seeding 类似 spatial restart。** 它不是让旧 Gaussian 穿过 depth barrier，而是在新 depth 直接重启一个局部 optimizer；与 multi-start optimization 的关系比与普通 gradient noise 更接近。
- **个人分析：Random Splitting 是对 selection bias 的干预。** 标准 ADC 只观察“已经能产生大 gradient 的点”，这是一种幸存者偏差；随机 split 给低信号区域探索资格。
- **个人分析：Split 的强结果说明 Blur Trap 可能主要是 allocation trap。** 在多个数据集上，Split 远强于 Seed；优化方向不足和容量门控失败并非同等重要，实际系统可能更应优先重新设计 densification statistics。
- **个人分析：Uniform bbox seeding 的效率会随空体积指数恶化。** 大型 unbounded scene 的 AABB 大部分是空空间，100 个随机点命中薄表面的概率可能很低；当前收益说明 pruning 能容忍浪费，但不是可扩展的最终方案。
- **个人分析：探索预算应按 uncertainty 而非全局均匀分配。** 可把 multi-view ray intersection condition、depth variance、accumulated transmittance 和 residual map组合成 proposal distribution，保留本文机制但大幅提高命中率。
- **个人分析：它与 TruncGradGS 是互补的两种修复。** 本文通过创建/拆分 primitives 绕过梯度不可达；TruncGradGS 通过扩大 backward support 让已有 dead Gaussians 获得远程梯度。前者改状态空间覆盖，后者改 gradient field。

## 局限与展望

### 作者文本中可确认的边界

论文没有独立的 Limitations section，但正文承认或暗示：

- multi-view camera trajectories 足够丰富时可以联合解决 depth ambiguity；Far-Side 问题主要出现在远景与窄 angular coverage。
- 当前 operators 刻意追求简单，是为验证 exploration principle，而非声称策略已经最优。
- pseudo-depth 不准确，可能对极远区域形成错误约束。
- 3DGS-MCMC / Opt3DGS 也是可行 exploration 路线，但采用统一 perturbation，本文选择按 failure type 区分操作。

### 独立分析

- **“fundamental limitation”措辞过强**：单 view orthogonality 是数学事实，但多视角可产生 depth constraint；问题严重度取决于 camera-ray angular diversity。
- **Blur Trap 缺操作性定义**：没有给出可自动检测某个 region 已进入 trap 的判据，也没有 blur persistence、depth error 或 visibility threshold。
- **Random Seeding 不真正针对 far side**：代码只在整个 AABB 内 uniform sampling，未利用 cameras、rays、residual 或远景 mask；大量 seeds 会落到已覆盖/空区域。
- **Random Splitting 不真正检测 occlusion**：它按 scale 概率采样，large Gaussian 不等于 occluded Gaussian，因而可能拆分本来正确的 primitives。
- **代码存在关键实验混杂**：启用 seeding 同时把 world-space large-Gaussian pruning threshold 从 $0.1\times$ extent 放宽到 $10\times$；缺少控制变量实验。
- **论文/代码参数不一致**：论文 $N_{seed}=20$，发布 config 为 100；不传 config 时 split schedule 又与推荐 `base` 不同。
- **初始化依赖坐标原点**：seed scale 与 opacity 使用 position norm；若数据没有像标准 3DGS 那样良好中心化/缩放，行为可能变化。
- **高 opacity 随机 seeds 可能扰动训练**：代码把初始 opacity 设约 0.2–0.7，且越远越高；没有与低-opacity warm start 或 gradual activation 比较。
- **没有报告 seed survival statistics**：不知道加入多少、多久被删、多少真正落到新结构、对 loss 造成多大瞬时冲击。
- **随机方法没有多 seed 统计**：所有主表无 mean/std 或 confidence interval，小幅 +0.01–0.05 dB 差异不应被解释为稳定排序。
- **“negligible overhead”证据不足**：未报告训练时间、CUDA memory、最终 FPS；T&T 与 DL3DV #GS 大幅上升。
- **模型规模主张选择性较强**：DB 显著减点很亮眼，但 DL3DV 几乎翻倍；不同数据的 cost-quality tradeoff 应分别讨论。
- **主表不是完整 SOTA comparison**：主要只有 3DGS 与 HoGS，缺 3DGS-MCMC、Opt3DGS、AbsGS、Pixel-GS、MCMC relocation 或 error-guided densification 的统一比较。
- **Depth loss 的因果解释不唯一**：没有扫描 $\lambda_D$、不同 monocular depth models、scale alignment 或 gradient cosine，无法确认是 deterministic exploitation 抑制 exploration。
- **DL3DV subset 是作者随机选择的 16 scenes**：没有 selection seed、难度分层或全 benchmark 结果；“unselected scenarios widespread”是观察陈述，不是完整统计。
- **Blur Trap dataset 尚未发布**：无法复核论文专门展示的 failure cases 或自动化分析。
- **4DGS 代码、点数和耗时缺失**：动态扩展目前可验证性弱于静态实验。
- **仓库无明确 license**：公开可读不自动等于可自由复用。

### 建议的后续实验

1. 固定原始 pruning threshold，分别测试 Seed、threshold change、二者组合，隔离当前代码混杂。
2. 在 5–10 个 random seeds 上报告均值/方差，并跟踪 injected、survived、densified、pruned primitives 数量。
3. 固定训练 wall-clock、最终 #GS 与 VRAM，对比 uniform Seed、ray-based depth Seed、MCMC perturbation、TruncGradGS 与 modern densification。
4. 用 camera-ray angular spread 定量预测 Far-Side Trap，验证 improvement 是否与 depth condition number相关。
5. 记录每个 Gaussian 的 average transmittance/rank 与 split probability，验证 Near-Side failure 是否真的集中在长期后排 primitives。
6. 把 Random Splitting 的 proposal 改为 scale × occlusion × residual uncertainty，并做同 budget 比较。
7. 在 bbox 外设置 bounded expansion，或沿高 residual training rays 分层采样 depth，测试是否比全 AABB uniform 更高效。
8. 报告最终 FPS、training time、peak VRAM 与 disk size，给出完整 quality-cost Pareto curve。
9. 发布 Blur Trap dataset、4DGS code、exact configs、random seeds 与 license。

## 与相关工作的对比

| 方法 | 主要干预位置 | Exploration 方式 | 是否区分 failure type | 主要代价/风险 |
| --- | --- | --- | ---: | --- |
| 3DGS ADC | accumulated 2D gradient | 无显式 exploration | 否 | depth/occlusion bias 会同时影响移动和 densification |
| 3DGS-MCMC | Gaussian position update / relocation | Langevin noise + MCMC interpretation | 否 | 全局统一扰动，schedule 与噪声设计更复杂 |
| Opt3DGS | optimizer phase | adaptive SGLD exploration + curvature exploitation | 部分 | 更复杂的 optimizer state 与 phase design |
| HoGS | position parameterization | homogeneous coordinates 改善 far-field 表达 | 主要针对 far field | 改表示但不直接增加新 depth signal |
| TruncGradGS | rasterizer backward | 扩大 dead Gaussian 的 gradient support | 主要针对 coverage/vanishing gradient | biased gradient，训练 pair cost 高 |
| **本文** | **primitive creation 与 densification gate** | **Random Seeding + scale-weighted Random Splitting** | **是** | **随机命中率、点数膨胀、参数/裁剪混杂** |

本文最有价值的不是证明 uniform random 一定优于所有复杂方法，而是给出一个清晰诊断：3DGS 的探索缺口至少有“位置不可达”和“容量门控失败”两种，应采用不同操作处理。

## 启发与关联

- **Coverage-aware ADC**：densification criterion 不应只看 gradient magnitude，还应考虑长期 visibility、transmittance、ray diversity 和 uncertainty。
- **Training-only exploration、inference-free deployment**：允许训练时临时增加候选、扰动或扩大 backward support，最终仍导出标准 3DGS，是很实用的系统设计范式。
- **把 camera geometry 用于 proposal**：沿 residual ray 采样 depth，比 world-space uniform 更直接对应 Far-Side nullspace。
- **把遮挡统计用于 split proposal**：对“偶尔可见但长期 transmittance 低”的 primitives 主动分裂，比按 scale 随机更贴近 Near-Side 机制。
- **假设：探索退火**。前期广泛 seeding，中期基于 survival/residual 自适应 proposal，后期停止探索只做 exploitation，可减少对收敛末期的扰动。
- **假设：回收而非增点**。把低贡献 Gaussians relocation 到高 uncertainty depth intervals，结合 3DGS-MCMC，可在固定 primitive budget 下实现本文目标。
- **与已有笔记的联系**：TruncGradGS 是“救活旧点”，本文是“创建新候选/强制拆点”；两者联合时需要同预算实验，避免重复解决同一 coverage gap。

## 评分

| 维度 | 评分（10 分） | 理由 |
| --- | ---: | --- |
| 创新性 | 8.0 | viewing-ray nullspace 本身是投影几何常识，但将其与 blending-induced ADC failure 统一为 Blur Trap，并配对两类 exploration，框架有启发性。 |
| 技术可靠性 | 7.1 | 单视图正交证明和 gradient branch 消融扎实；“fundamental”外推、depth-loss 因果解释与代码 threshold 混杂削弱结论。 |
| 实验充分度 | 7.0 | 覆盖五个静态 benchmark 与 Neu3D，报告质量和 #GS；缺主流 exploration baselines、随机方差、wall-clock、VRAM 和完整动态复现。 |
| 写作清晰度 | 7.7 | 两类 trap 与两项 operator 对应直观；部分结论使用“consistently / negligible / state-of-the-art”比表格证据更强。 |
| 实用 / 研究价值 | 8.2 | 实现很小、无需改 renderer，适合快速验证；推荐 config 与论文不一致、点数可大增，工程使用前需重新标定。 |

**总体推荐：值得细读并做控制变量复现。** 论文最强的贡献是把 3DGS densification 重新放进 exploration–exploitation 框架，而不是 Random 本身有多复杂。对 ADC、初始化、动态 3DGS 或 gradient pathology 研究非常有启发，但当前代码与论文的差异意味着不能只跑 `--explore_cfg base` 就把结果视为论文机制的纯复现。

## 阅读结论

- **最值得记住的点**：3DGS 用同一条 screen-space gradient 同时决定“往哪里移动”和“是否增加容量”；它一旦被投影 nullspace 或遮挡削弱，优化会双重停滞。
- **最需要怀疑的点**：Random Seeding 的提升是否完全来自 exploration，当前代码同时放宽了 oversized-Gaussian pruning threshold，且论文默认 seed 数与发布配置不一致。
- **最值得复现或继续验证的点**：固定 wall-clock/#GS/pruning policy，对比 uniform seeding、ray-depth seeding、random splitting、visibility-aware splitting、MCMC relocation 与 truncated-gradient rescue。

## 相关论文

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 原始 renderer、ADC 与 densify/prune baseline。
- [3D Gaussian Splatting as Markov Chain Monte Carlo](https://arxiv.org/abs/2404.09591) — 最近的显式 exploration 前身，以 SGLD noise 与 relocation 重写 Gaussian optimization。
- **Opt3DGS: Optimization of 3D Gaussian Splatting** — 同期 exploration–exploitation 视角，使用 adaptive SGLD 与 curvature-aware exploitation。
- **HoGS: Unified Near-to-Far Gaussian Splatting via Homogeneous Coordinates** — far-field parameterization baseline；改善表达但不直接补 depth exploration。
- [TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates](https://arxiv.org/abs/2609.03534) — 通过 backward surrogate 扩大 dead Gaussians 的有效梯度范围，与本文的随机容量注入形成对照。
- [AbsGS: Recovering Fine Details for 3D Gaussian Splatting](https://arxiv.org/abs/2404.10484) — 修改 densification gradient aggregation，相关于“gradient cancellation/selection statistics 决定细节容量”。

## 官方资源

- [arXiv v1](https://arxiv.org/abs/2607.17965v1) — 完整论文、PDF、HTML 与 TeX source。
- [Project Page](https://chengbo-wang.github.io/ExploreGS/) — 方法概览与视觉案例。
- [Official Code](https://github.com/Chengbo-Wang/Exploration-for-GS) — 当前已发布原版 3DGS integration；`gsplat`、Blur Trap dataset 与 4DGS implementation 尚未发布。
