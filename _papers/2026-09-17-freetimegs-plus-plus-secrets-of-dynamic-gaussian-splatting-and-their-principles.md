---
title: "FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles"
subtitle: "系统拆解 4DGS 的隐性经验，并以 gated temporal opacity、UFM 初始化与训练期颜色校正提升可复现性"
authors: "Lucas Yunkyu Lee, Soonho Kim, Youngwook Kim, Sangmin Kim, Jaesik Park"
venue: "arXiv preprint"
year: 2026
date: 2026-09-17
paper_url: "https://arxiv.org/abs/2605.03337"
code_url: "https://yklcs.com/ftgspp"
tags:
  - Dynamic Gaussian Splatting
  - 4D Gaussian Splatting
  - Dynamic Scene Reconstruction
  - Motion Consistency
  - Reproducibility
summary: "FreeTimeGS++ 先将 FreeTimeGS 中未充分说明的初始化、relocation 与评估细节显式化，并据此提出三项改动：以 gated marginalization 区分持续/瞬态 Gaussian，以 UFM flow 初始化速度，以仅训练期使用的 affine color correction 降低优化不稳定性。在 DyNeRF 与 SelfCap 上，它将复现的 FreeTimeGS 基线从 32.62/26.43 PSNR 提高到 33.40/27.12，并在固定初始化、10 次重复优化下显著缩小 PSNR 标准差。"
permalink: /papers/freetimegs-plus-plus-secrets-of-dynamic-gaussian-splatting-and-their-principles/
---

> **阅读依据**：arXiv:2605.03337v3（2026-07-01，13 页，含补充材料）。代码链接由论文提供：[ftgspp](https://yklcs.com/ftgspp)。本文以 FreeTimeGS 为代表 4D Gaussian Splatting（4DGS）系统做受控分析；对原 FreeTimeGS 的结果称为 `FreeTimeGS Wang`，而文中自行实现、正式实验的基线称为 `FreeTimeGS ours`，二者不能混为同一复现结果。

## 一句话总结

FreeTimeGS++ 的主张不是凭一个更复杂的时空网络取胜，而是先暴露动态 3DGS 中常被忽略的五个机制：Gaussian lifetime 的时间分工、图像质量与运动质量脱钩、与场景匹配的时空初始化、relocation 的参数继承、以及 run-to-run 方差；再以连续 persistence gate、UFM motion prior 和训练期 color correction 让一个 4D primitive pipeline 更稳定、更可复现。

## 背景与核心问题

4DGS 通常在每颗 Gaussian 中建模位置、时间、运动和 opacity。不同工作可能采用 deformation MLP、时空特征或原生 4D primitive，却都能取得相近的 photometric scores。作者认为这掩盖了一个问题：高分究竟来自何种表示能力，哪些来自未写清的工程 heuristic、数据集结构或评估约定？

在 FreeTimeGS 的 duration-based primitive 中，空间中心按线性速度移动：

$$
\mu_x(t)=\mu_x+v(t-\mu_t),
$$

并以高斯形的时间 activation 调制 base opacity：

$$
a(t)=\exp\left[-\frac12\left(\frac{t-\mu_t}{s}\right)^2\right].
$$

但公式本身还不足以复现一个系统：点云如何初始化、dead Gaussian 如何 relocation、参数如何继承、LPIPS 用什么值域、是否报告多次训练的方差，都会改变结论。本文的核心研究问题是：**如何把这些隐性选择变成可检验原则，并用它们构造更可靠的 4DGS？**

## 方法与五个发现

### 整体框架

<div class="mermaid">
flowchart LR
    A[FreeTimeGS 透明复现基线] --> B[表示、初始化、训练、评估的受控分析]
    B --> C1[生命周期与运动行为]
    B --> C2[时空初始化]
    B --> C3[relocation 与重复运行]
    C1 --> D1[Gated marginalization]
    C2 --> D2[UFM 引导速度初始化]
    C3 --> D3[训练期 color correction]
    D1 --> E[FreeTimeGS++]
    D2 --> E
    D3 --> E
    E --> F[标准动态 Gaussian 渲染]
</div>

论文先用 RoMa correspondence、显式 parameterization、MCMC-inspired relocation 等选择复现 FreeTimeGS，再分别干预各部分。最终的 FreeTimeGS++ 保留线性速度和 relocation 主框架，只加入后三个实线模块。

### Secret 1：duration 会自发形成时间角色分工

若以渲染 opacity 超过阈值 $\theta$ 定义“可见”，补充材料给出的 effective lifetime 为：

$$
\Delta t(\theta)=2s\sqrt{2\log(o/\theta)}
=\frac{d}{3}\sqrt{2\log(o/\theta)},
$$

其中 nominal duration $d=6s$，$o$ 是 base opacity。因此寿命并不只由 duration 决定，opacity 同样影响可见窗口。收敛后，长寿命 Gaussian 主要重建静态背景，短寿命 Gaussian 则突出动态主体：模型即使没有显式 static/dynamic branch，也会出现 persistent/transient 的隐式时间划分。

### Secret 2：干净 RGB 不保证运动正确

在图像重建损失下，模型可以通过不同 Gaussian 在相邻时刻轮流解释同一物体（role switching）获得很高 PSNR；但 velocity map 仍可在应静止区域出现高频噪声与 motion leakage。故 photometric fidelity 没有约束 primitive identity 或轨迹连续性，论文特别提醒不要以 RGB 指标代替运动一致性评估。

### Secret 3：初始化密度必须匹配场景与动作

空间端，RoMa matching 的 indoor/outdoor depth prior 应按实际 geometry 而非数据集标签选择：室内序列若有窗外或深走廊，outdoor-like prior 反而更好。时间端，keyframe stride 既控制点云数量也控制初始 velocity correspondence 的时间分辨率；它不是越密越好。

| 固定 500k primitive 的 temporal stride | DyNeRF PSNR / LPIPS-Alex | SelfCap PSNR / LPIPS-VGG |
| --- | --- | --- |
| 较稀疏 | stride 50：32.55 / 0.033 | stride 10：26.22 / 0.143 |
| 默认 | **stride 10：32.62 / 0.033** | **stride 1：26.43 / 0.137** |
| 更密 | stride 5：32.57 / 0.033 | 不适用 |

DyNeRF 动作较慢或局部，stride 10 已足够；SelfCap 的快速人体动作需要每帧初始化。结论是 temporal sampling 应由序列长度、帧率与动作复杂度决定。

### Secret 4：relocation 继承 opacity/scale 会改变伪影，而非只改变分数

训练中，低 opacity 的 dead primitives 被移到高 sampling score 区。作者比较三种继承：`Exact Copy` 复制 target 的全部参数；`Partial Copy` 不复制 opacity/scale；`MCMC` 按 MCMC density control 重算这两项。

| relocation | DyNeRF PSNR | SelfCap PSNR | 观察 |
| --- | ---: | ---: | --- |
| MCMC | 32.62 | 26.43 | 时空边界更稳定 |
| Exact Copy | **32.64** | **26.58** | 易在背景堆积冗余贡献，产生 haze/wobble |
| Partial Copy | 32.55 | 26.09 | 定量和视觉均较弱 |

Exact Copy 的 PSNR 最高，却可能造成静态背景雾状抖动；MCMC 的 opacity/scale 重分配略牺牲分数但抑制过度贡献。这个实验支持“仅看 PSNR 会错过 density-control artifact”。

### Secret 5：单次训练分数掩盖随机性

SelfCap 上以相同配置跑 10 次，部分场景 PSNR 分布明显分散。4DGS 同时优化 geometry、motion、temporal activation、appearance 与 density，多个不同内部解都能拟合逐帧 RGB；因此细小随机性就可能导致不同收敛路径。作者主张在比较微小改动时报告 repeated-run mean 和 standard deviation。

## FreeTimeGS++：三项分析驱动的改动

### 1. Gated marginalization：显式但连续的 persistent/transient 分配

为每颗 Gaussian 学一个 gate：

$$
g_i=\sigma(\gamma\tilde g_i),
\qquad
\phi_i(t)=g_i+(1-g_i)
\exp\left[-\frac12\left(\frac{t-\mu_{t,i}}{s_i}\right)^2\right],
$$

并令 $\alpha_i(t)=o_i\phi_i(t)$。$g_i\to1$ 时 primitive 始终存在，$g_i\to0$ 时恢复局部 temporal activation；这不是硬分类，而是连续选择。实现中 $\gamma=20$、gate logit 初始化为 $-1$，另有 gate regularization。

### 2. UFM-guided initialization：可靠时用 flow，失效时回退 KNN

原 baseline 以相邻 keyframe 点云的 KNN correspondence 估 velocity。FreeTimeGS++ 从预训练 Unified Flow & Matching（UFM）投影/采样多视角 optical flow 并反投影为 $v_i^{\mathrm{ufm}}$：

$$
v_i^0=m_i v_i^{\mathrm{ufm}}+(1-m_i)v_i^{\mathrm{knn}},
$$

$m_i\in\{0,1\}$ 由 confidence 与 multi-view consistency 决定。也就是说，它不会盲信 flow：没有可靠 UFM estimate 时保持 KNN 初始化。代价也须计入：有 flow cache 时 DyNeRF/SelfCap 初始化为 44.1/80.8 s（baseline 为 17.5/20.7 s）；无 cache 的 flow 预计算则为 741.3/1707.8 s。

### 3. 只在训练期使用的 affine color correction（CC）

对相机 $c$ 的 render，训练时施加：

$$
\hat I_c^{\mathrm{cc}}=\hat I(I_3+\Delta M_c)+b_c.
$$

每相机的 $\Delta M_c\in\mathbb R^{3\times3}$ 与 $b_c\in\mathbb R^3$ 被正则到 identity/zero，最终渲染时完全移除 CC。作者解释它吸收 illumination 或 camera color 差异，使 Gaussian 不必用几何与运动去补偿 photometric variation，因而更稳定；这是一种训练 optimizer 的缓冲层，不是部署时的颜色后处理。

总体损失为：

$$
L=\lambda_1L_1+\lambda_sL_{\mathrm{ssim}}+L_{\mathrm{reg}}+L_{\mathrm{gate}}+L_{\mathrm{cc}}.
$$

## 实验证据

### 主结果与消融

论文在 DyNeRF（6 场景、19–21 同步相机）、SelfCap（6 个快速人体场景）、ENeRF-Outdoor、Google Immersive 上测试；一般设置为 30k iterations、最多 500k Gaussians。Table 5 的平均消融如下（SSIM、LPIPS 使用的 Alex/VGG backbone 随数据集而异）：

| 方法 | DyNeRF：PSNR / SSIM / LPIPS | SelfCap：PSNR / SSIM / LPIPS |
| --- | --- | --- |
| FreeTimeGS ours（BASE） | 32.62 / 0.978 / 0.033 | 26.43 / 0.946 / 0.137 |
| BASE + UFM | 32.73 / 0.979 / 0.031 | 26.25 / 0.945 / 0.138 |
| BASE + UFM + GATE | 32.78 / 0.978 / 0.034 | 26.41 / 0.945 / 0.139 |
| **FreeTimeGS++（+ CC）** | **33.40 / 0.978 / 0.033** | **27.12 / 0.947 / 0.137** |

- CC 是该表中最大的平均 PSNR 增益来源：相对 BASE，DyNeRF +0.78 dB、SelfCap +0.69 dB；单独 UFM 对 SelfCap 的平均指标没有正增益，说明它更像初始化 prior，而非稳定的单一质量提升器。
- 表 6 在**固定初始化**后重复优化 10 次，CC 将 DyNeRF 从 $32.47\pm0.22$ 提升至 $32.81\pm0.08$，SelfCap 从 $26.43\pm0.36$ 提升至 $27.32\pm0.09$。这是本文最直接的 reproducibility 证据。
- 相对原论文报告、500k 点上限下的 FreeTimeGS Wang，FreeTimeGS++ 的 DyNeRF PSNR 为 33.40 对 32.97；SelfCap 为 27.12 对 27.27。后者并未超过原报告，但作者自己的 six-run 基线与最终方法之间有明确提升，不能把不同实现的数值直接归因于某一模块。

### 泛化与评估约定

| 户外数据集 | FreeTimeGS ours | FreeTimeGS++ |
| --- | --- | --- |
| ENeRF-Outdoor | 25.20 PSNR / 0.256 LPIPS | **25.26 / 0.246** |
| Google Immersive | 22.29 / 0.094 | **22.68 / 0.091** |

户外收益存在但很小，尤其 ENeRF-Outdoor 仅 +0.06 dB。论文还明确 LPIPS 的输入范围会显著改变数值；全文采用同一 3DGS 习惯的 $[0,1]$ convention，避免把不同 normalization 下的 LPIPS 混在一起。这个提醒很有用，但也意味着与采用数学标准 $[-1,1]$ normalization 的论文不能直接横比 LPIPS。

## 亮点与局限

**论文亮点**：

- 不只推出一个变体，还将以往含糊的 4DGS 工程选择转化成可复现实验和明确公式。
- 明确证明 RGB quality 与 motion behavior 脱钩，并以 velocity map、space-time slice 补充传统 view synthesis 指标。
- 把 repeated-run variance 当成一等评估对象；CC 在固定初始化的 10-run 实验中同时改善均值与方差。
- UFM 始终有 KNN fallback，避免外部 motion prior 在无效区域强行覆盖。

**作者承认的限制**：分析以 FreeTimeGS 为代表；五个秘密是否跨所有 4DGS 架构普遍成立，仍需系统性 cross-method evaluation。补充材料只对 STGS 初步验证 CC（DyNeRF 六次运行 PSNR 从 $31.77\pm0.33$ 到 $33.06\pm0.19$）。

**独立分析**：

- “Secrets”中前五项大多是基于一个复现基线的诊断，而非每种 4DGS 的因果定律；deformation-field 方法是否同样有 duration induced partition，本文没有直接检验。
- motion inconsistency 的证据主要是可视化 velocity map 和伪影，不含 3D trajectory ground truth、flow EPE 或 temporal geometry metric。因此“运动更好”仍是强但未完全量化的判断。
- CC 只在训练时使用而能大幅提升 PSNR，说明 camera/illumination nuisance 对 benchmark 很敏感；但也可能让模型将跨相机外观差异吸收到相机参数，弱化真实 appearance consistency。需要在未见相机、跨曝光或 color-calibrated 数据上进一步测试。
- UFM 的无缓存预计算在 SelfCap 需约 28 分钟，远高于 20.7 s baseline initialization；论文应同时报告训练总墙钟、GPU 能耗与不同 flow cache 可用性下的收益。

## 与相关工作对比

| 方法 | 动态表示 | 核心控制 | 本文关系 |
| --- | --- | --- | --- |
| 4DGS | 原生 4D primitive | 4D rotation、时空 ADC | 强 photometric 对照，设计不同 |
| STGS | spacetime Gaussian features | polynomial / guided sampling | 补充实验显示 CC 也能降低其方差 |
| FreeTimeGS | duration-based 4D primitive + linear velocity | temporal activation、relocation | 本文复现并分析的直接基线 |
| **FreeTimeGS++** | gated duration-based primitive | gate、UFM velocity、训练期 CC | 将经验变成可控模块，强调稳定性 |

## 评分

| 维度 | /10 | 依据 |
| --- | ---: | --- |
| 创新性 | 8.1 | 系统分析与可复现化比单纯增加模块更有研究价值；gate/CC 本身较轻量。 |
| 技术可靠性 | 7.8 | 公式、受控消融和 repeated-run 设计较扎实；跨架构普适性尚未充分验证。 |
| 实验充分度 | 7.9 | 四数据集、户外检查、速度初始化成本与 STGS-CC 初测齐全；缺运动 GT 指标和完整 cross-method audit。 |
| 写作清晰度 | 8.5 | 明确区分原方法报告与自行复现，并交代 LPIPS convention。 |
| 实用/研究价值 | 8.6 | 对复现/比较任何 4DGS 都有直接的 checklist 价值。 |

**总体推荐：值得细读。**特别适合在复现动态 Gaussian 方法前阅读，因为它指出了论文代码里最容易被当作“小细节”、却足以改变结论的环节。

## 阅读结论

- **最值得记住的点**：duration-based Gaussian 已会自发分出背景与瞬态角色；显式 gate 是把这个现象变成可控变量。
- **最需要怀疑的点**：高 PSNR 不足以证明 3D motion 正确，而本文也尚未用 ground-truth motion 指标完整证实其内部轨迹更好。
- **最值得复现或继续验证的点**：将 velocity consistency、space-time artifact、PSNR 方差一同报告，并在 deformation-based 4DGS 上逐项复现五个因素。

## 相关论文

- [FreeTimeGS](https://arxiv.org/abs/2503.16359) — 本文直接复现并改进的 duration-based 4D Gaussian 基线。
- [4D Gaussian Splatting](https://arxiv.org/abs/2310.08528) — 原生 4D primitive 的代表性动态渲染方法。
- [STGS](https://arxiv.org/abs/2403.11447) — spacetime Gaussian features；本文在补充材料检验 CC 的另一基线。
- [MCMC-based 3D Gaussian Splatting](https://arxiv.org/abs/2404.09591) — relocation 中 opacity/scale 重算的来源。
- [UFM](https://arxiv.org/abs/2501.03729) — 本文用于 multi-view flow 初始化的预训练 correspondence 模型。
