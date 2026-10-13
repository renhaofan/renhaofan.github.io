---
title: "Efficient Confidence-Aware Densification for 3DGS"
subtitle: "用置信度加权的图像误差决定 Gaussian 增密位置，以最大混合权重释放低贡献 primitives，在固定预算下修正前景偏置"
authors: "Manuel Hofer, Thomas Köhler, Felix Windisch, Bernhard Kerbl, Markus Steinberger, Lukas Radl"
venue: "arXiv preprint"
year: 2026
date: 2026-10-13
paper_url: "https://arxiv.org/abs/2610.13541"
tags:
  - 3D Gaussian Splatting
  - Densification
  - Fixed Primitive Budget
  - Confidence Estimation
  - Novel View Synthesis
summary: "该工作保留 3DGS-MCMC 的固定预算与渐进增长，却以置信度加权的绝对屏幕空间梯度和结构误差选择增密来源，以最大 blending weight 淘汰低贡献 Gaussian，并用一次性位置采样和图像颜色初始化完成迁移；在 24 个场景、100k–2M primitives 的所有预算上均超过 3DGS-MCMC，500k 起取得固定预算方法中最高的平均 PSNR 与 SSIM。"
permalink: /papers/efficient-confidence-aware-densification-3dgs/
---

> **阅读依据**：[arXiv:2610.13541v1 官方 HTML](https://arxiv.org/html/2610.13541v1)与 PDF（2026-10-09，20 页，正文及补充材料）。截至阅读时，论文和 arXiv HTML 未提供可核实的官方代码/项目页链接；速度对比中本文使用 Faster-GS rasterizer 与 fused Adam，而部分基线运行官方代码，因而不能把全部时间差都归因于 densification 算法。

## 一句话总结

Efficient Confidence-Aware Densification 诊断出 3DGS-MCMC 的 opacity sampling 会反复在已重建良好的前景增加 Gaussian，于是把固定预算下的两个关键决策改写为“图像误差决定新 primitives 去哪里、实际渲染贡献决定谁让出名额”：80% 来源按置信度加权的绝对屏幕空间梯度抽样、20% 按结构误差抽样，再把最大 blending weight 最低的 primitives 迁移过去，从而在无需 radiance-field/dense-match prior 的情况下显著改善低预算背景与总体 NVS 质量。

## 背景与核心问题

3D Gaussian Splatting（3DGS）的 Adaptive Density Control（ADC）会在大 image-space gradient 处 split / clone，并按低 opacity 剪枝；它效果可靠，但依赖 SfM 点，且无法直接保证最终 Gaussian 数量。3DGS-MCMC 改用固定预算的渐进增长和 relocation：按 opacity 抽取 source Gaussian，把低 opacity primitives 搬到这些来源附近，因此既能从随机点启动，又能精确控制 primitive count。

作者指出 opacity 不是 reconstruction error：

- 被许多训练视图看到的前景 primitives 容易保持高 opacity，即使该区域已经重建良好；
- opacity-proportional source sampling 会继续在这些地方复制 primitives；
- 低 opacity relocation 又把空出的 primitives 搬回同样的高-opacity 区域；
- 这种正反馈在预算填满期间就形成，导致远处背景 primitive 不足（Figures 2–3、5）。

低预算 prior-based 方法 ConeGS / EDGS 因更好的初始化而领先，但需预训练 iNGP 或 dense matches 和独立 initialization stage；当预算增至 2M 时，它们反而落到 3DGS-MCMC 后面（Figure 1）。这说明 prior 提供低预算 head start，却不必然提供更好的 densification。

**核心研究问题：能否保留 3DGS-MCMC 的固定预算、渐进增长与随机初始化鲁棒性，同时让每个有限的 Gaussian 真正流向尚未解释好的图像区域？**

## 方法详解

### 整体框架

<div class="mermaid">
flowchart LR
    A[当前 3D Gaussians] --> B[训练视图渲染与 RGB loss]
    B --> C[更新 Gaussian 参数和 learned confidence]
    C --> D[累计 absolute screen-space gradients]
    B --> E[累计 per-primitive SSIM error]
    C --> F[每 100 iter 的 densification]
    D --> G[80%: gradient plus confidence source sampling]
    E --> H[20%: SSIM source sampling]
    G --> I[source Gaussians]
    H --> I
    F --> J[每 400 iter: 选择低 max blending-weight primitives]
    I --> K[从 source Gaussian 一次性采样新位置]
    J --> K
    K --> L[训练图像初始化 DC color]
    L --> A
</div>

训练共 30k iterations；从 500 到 25k 每 100 iterations 做一次 densification，每步增长 5% 直到达到 budget，同时进行 relocation。这些外层规则沿用 3DGS-MCMC。本文改变的是 source distribution、被搬走的 primitives、迁移后的新位置与颜色。

### 1. Learned confidence：把“尚未解释好”变成 per-primitive 信号

每个 primitive 学一个 log-confidence $\gamma_i$，正值形式为 $\bar\gamma_i=\exp(\gamma_i)$。像颜色一样 alpha blend 得到像素 confidence $\hat C(\mathbf p)$，使用 CoMe 的自监督损失：

$$
\mathcal L_{\mathrm{conf}}(\mathbf p)
=\mathcal L_{\mathrm{rgb}}(\mathbf p)\hat C(\mathbf p)
-\beta_{\mathrm{conf}}\log\hat C(\mathbf p).
$$

对固定 photometric error 求最优值得

$$
\hat C^*(\mathbf p)=
\frac{\beta_{\mathrm{conf}}}{\mathcal L_{\mathrm{rgb}}(\mathbf p)}.
$$

因此 error 高于 $\beta_{\mathrm{conf}}$ 时 confidence 低于 1；覆盖这些像素的 primitives 会学成低置信度。实验统一用 $\beta_{\mathrm{conf}}=0.075$，并把 confidence map clamp 到 $[10^{-3},5]$。Table 4 中换成 0.05 或 0.1 时，100k/500k PSNR 变化均不足 0.1 dB，说明该参数不敏感。

confidence 的局限是冷启动：所有 $\bar\gamma_i$ 初始为 1，前几百步几乎没有区分度（Figure 4）。因此作者不用 confidence 单独采样，而让它逐渐调制一个从训练开始就可用的 error signal。

### 2. Confidence-aware error sampling：决定新 Gaussian 去哪里

作者采用 GOF / AbsGS 的绝对屏幕空间梯度 $\nabla_{\mathrm{abs}}$：对 primitive 覆盖像素的 projected-position gradient 逐项取 magnitude 后求和，避免相反方向梯度互相抵消。大 Gaussian 横跨多个结构时，普通 signed sum 可能很小，但 absolute gradient 仍能暴露其模糊细节的问题。

记 $r_i(x)\in[0,1]$ 为量 $x_i$ 在所有 primitives 中的归一化 rank。80% sources 按以下未归一化概率抽取：

$$
P_i=r_i(\nabla_{\mathrm{abs}})
\left((1-\eta)+\eta\bigl(1.2-r_i(\bar\gamma)\bigr)\right),
\qquad
\eta=\min(t/5000,1).
$$

使用 rank 有两点好处：梯度与 confidence 的量纲不需人工配平，极端大梯度也不会独占分布。前 5k iterations，$\eta$ 从 0 线性增至 1：早期只按 absolute gradient，confidence 收敛后再把低置信度 primitive 权重提高。最终 confidence factor 从最低 confidence 的 1.2 到最高 confidence 的 0.2，但不会彻底排除已重建区域。

剩余 20% sources 从 per-primitive structural error 抽取：先计算像素 $1-\mathrm{SSIM}$，再按 blending weight 分配给贡献 primitives。它不乘 confidence，专门补足“$L_1$ 不高但结构明显错误”的区域。Figure 5 显示 gradient+confidence 与 SSIM 两个分布都会覆盖 opacity sampling 漏掉的高误差背景。

### 3. Maximum blending weight：决定谁让出预算

3DGS-MCMC 搬走 opacity $\le 0.005$ 的 primitives；但 opacity 没有考虑前方遮挡。本文以自上次 relocation 以来渲染过的训练视图 $\mathcal V$ 为范围，记录

$$
w_i=\max_{k\in\mathcal V,\mathbf p}b_i^k(\mathbf p),
$$

其中 $b_i^k(\mathbf p)=\alpha_i(\mathbf p)T_{i-1}(\mathbf p)$ 是 primitive $i$ 对像素的实际 alpha-blending weight。若 primitive 被遮挡或从未可见，$w_i$ 会很低甚至为 0；因此它比 opacity 更直接衡量“这个 Gaussian 是否真的影响过任何训练图像”。

relocation 每 400 iterations 执行一次，给新 primitives 时间适应：

- 0–5k：只搬 $w_i=0$ 的 primitives，最多占总数 10%；
- 5k–10k：搬 $w_i$ 最低的 10%；
- 10k–25k：比例乘 $\exp[-\tfrac12((t-10k)/5k)^2]$，在 15k/20k/25k 降到初始的 61%/14%/约 1%；
- 刚 relocation 的 primitive 免于下一轮再次搬动。

这项设计必须和新的 source distribution 配合。Table 4 中把本文 relocation 换回 MCMC threshold，PSNR 在 100k/500k 分别从 26.32/27.84 降至 25.75/27.23，约损失 0.6 dB；反过来，仅把本文 relocation 或 sampling 单独装进 3DGS-MCMC，都只带来不到 0.1 dB。直觉是：释放谁与搬到哪里必须同时正确。

### 4. 一次性位置采样与颜色初始化

每个新增或被搬迁 primitive 复制 source 的 rotation、SH coefficients 和 confidence，并沿用 3DGS-MCMC 的 opacity/scale adjustment。区别在于：

- 禁用每 iteration 的连续 position noise；
- 只在创建时从 source 的调整后 Gaussian $\mathcal N(\boldsymbol\mu_i,\boldsymbol\Sigma'_i)$ 采样一次新中心；
- 找到放置后第一个能看到该 primitive、且 projected radius 小于 16 pixels 的训练视图，把新中心投影进去并双线性采样 GT color；保留 higher-order SH，只调整 DC coefficient 使该视角颜色等于 GT；重置 optimizer moments；
- 400 iterations 内找不到合适视图时，保留 source color。

一次性采样提供 clone 分离所需的探索，却避免连续噪声反复扰动少视角背景。Table 4 中完全去掉 position randomness 使 500k PSNR 从 27.84 降到 26.96；在 position sampling 上再加 MCMC noise 也降到 27.62。color initialization 是较小但稳定的贡献（27.84→27.68）。

## 实验关键数据

### 实验设置

评测覆盖 24 个真实场景：Mip-NeRF 360 的 9 个、Tanks & Temples 和 Deep Blending 各 2 个、DL3DV 的 11 个；每 8 张图留 1 张测试。固定预算为 $\{100\mathrm{k},500\mathrm{k},1\mathrm{M},2\mathrm{M}\}$，报告 PSNR / SSIM（高为好）和 VGG LPIPS（低为好）。固定预算基线包括 3DGS-MCMC、Taming 3DGS、ConeGS 与 EDGS；另在相同最终 primitive count 下对比 3DGS、3DGS+GOF、DashGaussian、Mini-Splatting2 和 GaussianSpa。

论文把低于 0.05 dB PSNR 或 0.002 SSIM/LPIPS 的差异视为 on par。本文和 3DGS-MCMC 在同一 Faster-GS 代码库运行；其他方法使用官方代码。

### 固定预算主结果

下表将 Table 2 的四个 dataset means 按场景数加权为 24-scene mean；这些数也与 Table 7 的 SfM 行一致：

| Budget | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | ---: | ---: | ---: |
| 100k | 3DGS-MCMC | 25.70 | 0.783 | 0.310 |
| 100k | **本文（SfM）** | **26.54** | **0.809** | **0.282** |
| 100k | ConeGS | **26.75** | **0.813** | **0.273** |
| 500k | 3DGS-MCMC | 27.37 | 0.843 | 0.233 |
| 500k | ConeGS | 27.88 | 0.859 | **0.199** |
| 500k | **本文（SfM）** | **28.05** | **0.863** | 0.202 |

- 在 100k，本文相对 prior-free 3DGS-MCMC 提高 0.84 dB，但仍落后 ConeGS 0.21 dB；不能说最低预算全面最佳。
- 从 500k 起，本文在全部固定预算方法中取得最高 PSNR 和 SSIM；领先第二名的 PSNR 从 500k 的 0.18 dB 增长到 2M 的 0.34 dB（Section 5.1、Figure 7）。
- 本文在 100k–2M 的每个 budget 上均以 PSNR、SSIM、LPIPS 超过 3DGS-MCMC；相对增益随预算增加仍存在，但 PSNR 从 +0.84 dB 收窄到 +0.34 dB。
- LPIPS 是弱项：SfM 初始化下，本文在 24-scene average 从未拿到固定预算最佳 LPIPS；500k 后只落后 ConeGS / EDGS 至多 0.003，差距主要来自 outdoor high-frequency detail。

### 前景/背景诊断

Table 1 把 outdoor Mip-NeRF 360 中按 aligned Depth Anything V2 depth 最远的 25% 测试 pixels 定义为 background。本文相对 3DGS-MCMC 的 PSNR 增益为：

| Budget | Background PSNR 增益 | Foreground PSNR 增益 |
| ---: | ---: | ---: |
| 100k | +0.69 dB | +0.41 dB |
| 500k | +1.20 dB | +0.23 dB |
| 1M | +1.03 dB | +0.15 dB |
| 2M | +0.94 dB | -0.03 dB |

500k 以后几乎所有 PSNR 收益都来自背景；2M 时 foreground 甚至轻微下降。这直接支持“修正 opacity foreground bias”的机制主张，也说明方法不是均匀改善所有区域。

### 消融：模块必须协同

Table 4（9 个 Mip-NeRF 360 scenes）最关键的 500k 结果：

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| 3DGS-MCMC | 27.30 | 0.797 | 0.260 |
| MCMC + 本文 relocation | 27.35 | 0.800 | 0.258 |
| MCMC + 本文 sampling | 27.31 | 0.797 | 0.263 |
| 本文但用 MCMC sampling | 27.74 | 0.814 | 0.239 |
| 本文但用 MCMC relocation | 27.23 | 0.804 | 0.249 |
| **完整方法** | **27.84** | **0.819** | **0.231** |

两项新规则并非即插即用地各自贡献一半：放进旧 pipeline 时几乎没有 PSNR 收益；完整系统才超过 MCMC 0.54 dB。Table 5 进一步显示 absolute gradient、confidence weighting、20% SSIM sources 的 PSNR 增益会叠加：gradient-only 27.61，+SSIM 为 27.74，+confidence 为 27.70，三者合用为 27.84。结构误差还明显改善背景细节（Figure 9）。

### Prior、随机初始化和超预算增长

- **Matched initialization（Table 7）**：从 ConeGS 的 iNGP 初始化开始，100k 时本文 26.79 / 0.813 / 0.278，对比 ConeGS 26.75 / 0.813 / 0.273；PSNR/SSIM 持平，但 ConeGS LPIPS 更好。500k 时本文 28.22 / 0.866 / 0.197，三项均优于 ConeGS 27.87 / 0.859 / 0.199。用 EDGS dense matches 也得到几乎相同结论，说明 densification 能利用而不绑定某一种 prior。
- **Random initialization（Section 5.2 / Table S.5）**：100k 时比随机初始化的 3DGS-MCMC 高 1.9 dB；在原 3DGS primitive counts 下，两者 PSNR 相同，但本文 LPIPS 0.188 优于 0.210。连续 noise 并非 random-start exploration 的必要条件。
- **Growth beyond budget（Figure 10）**：允许训练中增长到最终预算的 1.5–2 倍再按 $w_i$ 压回，可追平 Mini-Splatting2 / GaussianSpa 的 LPIPS，同时 PSNR 仍领先 0.5–0.6 dB；增至 3 倍反而使 LPIPS 变差，说明 $\max$ blending weight 会偏爱“只主导少量 pixels”的小 primitives。

### 内存与训练时间

Table 6 在单张 RTX 5090、Mip-NeRF 360 的 9 scenes 上：

| 方法 | 设置 | Peak memory | Init | Train |
| --- | --- | ---: | ---: | ---: |
| EDGS | fixed 500k | 7.3 GB | 1.8 min | 7.2 min |
| ConeGS | fixed 500k | 4.9 GB | 2.2 min | 11.8 min |
| **本文** | fixed 500k | **4.6 GB** | **无** | **2.8 min** |
| Mini-Splatting2 | variable budget | 8.2 GB | 无 | **1.9 min** |
| GaussianSpa | variable budget | 11.3 GB | 无 | 24.6 min |
| **本文（matched count）** | variable-budget comparison | **4.8 GB** | **无** | 3.0 min |

本文在 fixed-budget block 的端到端时间和显存最好；ConeGS 仅 initialization 的 2.2 分钟就接近本文完整训练。但作者明确承认实现差异：本文使用 Faster-GS / fused Adam，prior baselines 用官方代码。因此“方法本身快 3–5 倍”并未由同实现 ablation 证明；更稳妥的结论是本文不需要额外 initialization，并在其优化代码中 500k 可低于 3 分钟完成。

## 亮点与我的洞察

### 论文亮点

- 先以 source locations、pixel coverage 和 foreground/background metrics 证明 opacity bias，再针对“去哪里”和“谁让位”分别设计规则，问题—机制—证据闭环清楚。
- learned confidence 不替代 gradient，而作为 warm-up 后的排序权重；这避开 confidence 冷启动，且无需额外网络或离线 prior。
- 最大 blending weight 把 opacity、visibility、occlusion 和实际 image contribution 合成一个直接的 relocation criterion。
- 同时评测 fixed-budget、matched final count、prior initialization、random initialization 和 temporary overgrowth，区分了初始化、增密与剪枝的收益来源。

### 我的洞察

- **个人分析：这是一种固定容量的 online resource reallocation。** source distribution 是“需求端”，低 $w_i$ relocation 是“供给端”；只优化任何一边都无效，正如 Table 4 所示。这个框架比单纯设计更好的 split score 更一般。
- **个人分析：confidence 在这里不是不确定性估计终点，而是 error routing 的慢变量。** absolute gradient 提供早期、瞬时信号，confidence 汇总持续难以解释的区域，SSIM 捕捉 $L_1$ 不敏感的结构；三者分别覆盖不同时间尺度和误差类型。
- **个人分析：$\max$ contribution 适合保住细小显著 primitive，却不等于全局重要性。** Figure 10 在 3× overgrowth 时退化已暴露这一点：单个 pixel 上权重很高的小 Gaussian 可能被保留，而覆盖更大区域、累计贡献更高的 primitive 被淘汰。max、sum 与跨视图 coverage 的多目标 score 值得比较。

## 局限与展望

### 作者承认的局限

- 100k 极低预算下，SfM 初始化仍无法追上 ConeGS；即使用 ConeGS 初始化，其 LPIPS 仍更好。
- strict budget + sparse start 难以恢复 outdoor high-frequency detail；先 overgrow 到 1.5–2 倍可补救，但会提高 peak memory。
- growth / relocation schedules 手工设定，并对所有 budget 共用；高预算可能适合更大的 relocation share。
- primitive budget 也由用户指定；confidence 或可用于判断继续加 primitives 是否仍有收益。

### 独立分析

- **背景评测依赖一个定义选择**：最远 25% pixels 来自 COLMAP-aligned Depth Anything V2。该分析有说服力，但它衡量的是“远景”而非所有语义背景；透明物体、反射和近处背景的结论仍未知。
- **速度公平性有限**：Faster-GS 与 fused optimizer 同时改变底层执行效率，应该在统一代码库中替换 densification rule，单独报告 densification overhead、raster/optimizer speedup 与总时间。
- **额外状态虽小但非零**：每 primitive confidence、gradient/SSIM statistics、跨 relocation window 的 maximum blending weight 都需维护。论文报告 peak memory 很好，但没有单列这些 buffer 的增量与 densification step latency。
- **最大值对噪声和偶发可见性敏感**：一个 primitive 只需在一个 pixel/view 有过大权重就能存活。建议对 $\max$、top-$k$ mean、跨视图分位数、累积贡献做 controlled ablation，并测 geometry / floaters，而不只看 NVS metrics。

最值得追加的实验是把 source demand 和 primitive utility 合成自适应 budget controller：根据 confidence/error mass 的下降速度决定是否继续增长，根据跨视图 contribution distribution 决定 relocation share，并与固定 100k/500k/1M/2M schedule 比较质量–显存 Pareto。

## 与相关工作的对比

| 方法 | 初始化 | 增密/压缩依据 | 预算约束 | 主要优势与代价 |
| --- | --- | --- | --- | --- |
| 3DGS ADC | Sparse SfM | signed screen-space gradient；低 opacity prune | 最终数量不直接受控 | 简单可靠，但不能远离已有 points 探索 |
| 3DGS-MCMC | SfM 或随机 | 高 opacity source；低 opacity relocation；连续 noise | 训练全程固定上限 | 鲁棒、预算明确，但产生 foreground bias |
| ConeGS / EDGS | iNGP / dense matches | prior 提供 dense start；ConeGS 另按 error 放置 | 固定预算 | 100k head start 强，但有额外初始化和成本 |
| Mini-Splatting2 / GaussianSpa | SfM | 先训练大模型，再按 importance simplification | 中间超过最终预算 | outdoor LPIPS 强，peak memory 较高 |
| **本文** | SfM、随机或 prior 均可 | gradient × confidence + SSIM source；低 max-blending-weight relocation | 默认训练全程固定上限 | prior-free、背景与 PSNR 强；低预算 LPIPS 仍受 sparse start 限制 |

## 启发与关联

- **对 Gaussian compression 的启发**：与其训练后一次性 prune，可在固定容量训练中持续把低贡献 primitives 迁移到高残差区域，使表示从一开始就围绕部署预算形成。
- **对动态/在线 3DGS 的假设**：confidence 可作为长期未解释区域的状态，absolute gradient 作为当前帧突发信号；二者结合或可区分真实新几何与短时 photometric noise。但动态场景需防止低 confidence 只是在标记运动/遮挡。
- **对一般稀疏表示学习的启发**：固定 token/primitive budget 下，新增位置和删除位置是耦合的 transport problem；需求 score 与 utility score 应成对设计，而非分别贪心。

## 评分

| 维度 | /10 | 理由 |
| --- | ---: | --- |
| 创新性 | 8.4 | 组件多来自已有 gradient、confidence 与 pruning ideas，但把它们重组为固定预算的 demand–utility relocation 很有洞察。 |
| 技术可靠性 | 8.8 | 公式、warm-up、schedule 与初始化细节完整；核心双向协同由强消融支持。 |
| 实验充分度 | 9.0 | 24 scenes、四个 budgets、fixed/variable/prior/random/overgrowth 多种轴线，并主动报告 LPIPS 弱项。 |
| 写作清晰度 | 8.8 | foreground-bias 动机和每项设计角色清楚，主文与补充材料证据对应紧密。 |
| 实用 / 研究价值 | 9.0 | 无额外 prior、固定显存预算、500k 下低于 3 分钟，对受限 3DGS 训练很实用；代码尚未核实公开。 |

**总体推荐：值得细读。** 它不是简单给 3DGS-MCMC 加一个 confidence weight，而是把 fixed-budget densification 明确拆成“误差在哪里”和“谁没有贡献”两个耦合问题，并用实验说明只有同时解决二者，Gaussian 资源才会从冗余前景流向欠重建背景。

## 阅读结论

- **最值得记住的点**：固定预算 densification 的 source sampling 与 relocation 必须配套；单独换一个更聪明的 score 几乎不起作用。
- **最需要怀疑的点**：训练速度优势混有 Faster-GS / fused optimizer 的实现收益，且 100k 与 outdoor LPIPS 仍显示 sparse initialization 的硬上限。
- **最值得复现或继续验证的点**：统一代码库下的 densification-only 时间/显存消融，以及以跨视图贡献分位数替代单一 maximum blending weight。

## 相关论文

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 原始 3DGS 与 ADC 基线。
- [3D Gaussian Splatting as Markov Chain Monte Carlo](https://arxiv.org/abs/2404.09591) — 本文直接改造的 fixed-budget、prior-free densification 框架。
- [Gaussian Opacity Fields](https://arxiv.org/abs/2404.10772) — absolute-gradient / position-sampling 思路的重要来源之一。
- [AbsGS: Recovering Fine Details for 3D Gaussian Splatting](https://arxiv.org/abs/2404.10484) — 绝对屏幕空间梯度，避免 signed gradient cancellation。
