---
title: "PCGS: Deblurring 3D Gaussian Splatting with Patch Comparison"
subtitle: "用自适应 patch error 找出未充分重建区域，再以增长预算和边缘重要性控制 Gaussian densification"
authors: "Anonymous Authors"
venue: "ICML 2026 under review"
year: 2026
date: 2026-09-16
paper_url: "https://openreview.net/forum?id=4bLAhR3KK7"
tags:
  - 3D Gaussian Splatting
  - Densification
  - Patch Comparison
  - Novel View Synthesis
  - Gaussian Budget
summary: "PCGS 针对 3DGS 在模糊/伪影区域中 view-space positional gradient 过小、无法触发 split 的问题，将 rendered-vs-GT pixel loss 先用 Otsu 阈值化、再按 patch 聚合并二次 Otsu 筛选；它从每个错误 patch 中选取 alpha-compositing 贡献最大的 Gaussian 加入候选集，并与传统 gradient 候选合并。随后用对数增长的 per-step Gaussian budget 和 edge-map importance score 做受限采样。在统一重跑协议下，PCGS 在 Mip-NeRF360、Tanks&Temples、Deep Blending 分别取得 28.01/0.826/0.189、24.40/0.860/0.150、30.04/0.907/0.243（PSNR/SSIM/LPIPS），且训练时间与点数低于原始 3DGS。"
permalink: /papers/pcgs-deblurring-3d-gaussian-splatting-with-patch-comparison/
---

> **阅读依据**：本文依据用户提供的原始 PDF `17480_PCGS_Deblurring_3D_Gauss_Originally Submitted PDF.pdf`（14 页）。稿件标注为 **ICML 2026 under review / Do not distribute**，作者及机构匿名，未提供公开代码或项目页。论文的“Deblurring”指消除 3DGS 重建造成的 blur/artifacts，**不是**从模糊输入图像恢复清晰场景的相机运动/散焦去模糊任务。

## 一句话总结

Patch Comparison Gaussian Splatting（PCGS）不再只依赖 per-Gaussian 的 view-space positional gradient 触发 densification，而是从渲染误差中定位连续的 error patches，直接挑出其中贡献最大的 Gaussian 做 clone/split；再用动态增长预算和边缘重要性采样限制点数，从而以近似或更少的 Gaussian 获得更清晰的 novel views。

## 背景与核心问题

3DGS 每隔若干 iteration 累积各 Gaussian 的 screen/view-space positional gradient；超过阈值才 clone 或 split。该规则在高纹理、低覆盖或大 Gaussian 主导的区域会失效：一个大 splat 覆盖许多 pixel，却可能因 photometric loss 对局部 blur 不敏感而梯度不足，始终达不到 split threshold。直接下调阈值也不是答案：Mip-NeRF360 上从 $2\times10^{-4}$ 降至 $1\times10^{-4}$，PSNR 仅从 27.21 到 27.34 dB、SSIM 从 0.815 到 0.821，Gaussian 数却由 3.18M 升至 8.94M；$5\times10^{-5}$ 直接 OOM。

作者的切入点是：blur/artifact 往往是空间连续区域，不应只由孤立 pixel 的梯度决定。问题变为：**如何把 image-level 的“哪里重建坏了”可靠地映射到少量应 densify 的 3D Gaussians，同时不让点数失控？**

## 方法详解

### 整体框架

<div class="mermaid">
flowchart LR
    A[SfM 初始化 Gaussians] --> B[标准 3DGS 可微渲染]
    B --> C[rendered image 与 GT]
    C --> D[像素 loss map + Otsu 得到 error pixels]
    D --> E[16×16 patches；error-pixel ratio + Otsu]
    E --> F[error patches]
    F --> G[每个 pixel 选最大 alpha-weight contributor]
    B --> H[传统 view-space gradient candidates]
    G --> I[合并候选集]
    H --> I
    I --> J[edge-map importance score]
    J --> K[按对数增长 budget 采样]
    K --> L[clone / split densification]
    L --> B
</div>

### 1. Patch comparison：从误差区域反查主导 Gaussian

对 pixel $p$ 的颜色合成，3DGS 使用深度排序 alpha compositing：

$$
C(p)=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j).
$$

其中第 $i$ 个 Gaussian 的实际贡献权重为

$$
\omega_i(p)=\alpha_i(p)\prod_{j<i}(1-\alpha_j(p)).
$$

PCGS 先计算 rendered image 与 GT 的 pixel loss map，以 Otsu 自动阈值 $\tau$ 将显著误差标为 error pixels；随后将图像切成 patches，计算每个 patch 的 error-pixel ratio，并对这些 ratio 再跑一次 Otsu 得到 $\epsilon$，选出 error patches。使用两层自适应阈值避免按场景手调绝对 loss 或 patch sensitivity。

对于每个 error patch 内的 pixel，不取固定 Top-K，而只加入其最大贡献者：

$$
G_{\mathrm{select}}(p)=\arg\max_i\omega_i(p).
$$

它与原始 gradient 选中的 Gaussians 合并为候选集。论文的关键工程细节是：clone 后按比例降低 opacity，使子 Gaussian 的总贡献与原 primitive 对齐；最主导者被拆开、单个权重下降后，原先被遮蔽的次要贡献者会在后续 densification step 逐渐成为最大贡献者。作者以此将一次性的 Top-K 选择改成自演化的逐个 refinement，免去 K 的手调。

### 2. Growth control：预算限制与重要性采样

error patch 会带来额外 densification，硬性限定最终点数又会过早耗尽额度。PCGS 在 densification 开始/结束 iteration $I_s,I_f$ 之间采用对数增长预算：

$$
A(t)=S+(B-S)\frac{\log(1+kt)}{\log(1+k)},
\quad t=\frac{I-I_s}{I_f-I_s},\quad k=9.
$$

$S$ 是 SfM 初始点数，$B$ 是最终 budget。该曲线前期增长快，给 geometry 足够的 clone/split 探索；后期变慢，将优化重心转到既有 Gaussian 的属性细化。

当候选多于当前 step 能增加的配额时，PCGS 用 edge map 来表示结构显著性。对 Gaussian $i$，将它影响的 pixel 集合 $P_i$ 的 edge 值累加：

$$
e_i=\sum_{p\in P_i}e_{\mathrm{pix}}(p).
$$

高 $e_i$ 的候选优先 densify。它不是额外的监督 loss，而是预算下的选择分数；训练损失仍为原始 3DGS 的 $(1-\lambda)L_1+\lambda L_{SSIM}$。

### 实现与协议

patch size 为 $16\times16$，最终 budget $B$ 取原始 3DGS 的最终点数，代码基于 Taming-GS 并扩展 CUDA rasterizer 以输出每 pixel 最大贡献 Gaussian index。实验遵循 3DGS：Mip-NeRF360 9 scenes、Tanks&Temples 2 scenes、Deep Blending 2 scenes；每第 8 张图为 test，A6000 GPU，Adam。论文还重新运行多个 baseline，统一 Mip-NeRF360 的 9 scenes 与 ImageMagick 预下采样协议；因此不要把其数字与原论文的 PIL resize 结果直接混比。

## 实验证据

PSNR、SSIM 越高越好，LPIPS 越低越好（PDF 的部分表头将 LPIPS 误写成 ↑，此处按指标定义标为 ↓）。

| 数据集 | 3DGS | 最强/相近基线 | PCGS |
| --- | --- | --- | --- |
| Mip-NeRF360 | 27.21 / 0.815 / 0.214 | Perceptual-GS 27.68 / 0.825 / **0.189** | **28.01 / 0.826 / 0.189** |
| Tanks&Temples | 23.14 / 0.841 / 0.183 | Taming-GS 24.04 / 0.851 / 0.170 | **24.40 / 0.860 / 0.150** |
| Deep Blending | 29.41 / 0.903 / 0.243 | Scaffold-GS 30.21 / 0.906 / 0.255 | 30.04 / **0.907** / 0.243 |

表中每格是 `PSNR / SSIM / LPIPS`。作者证据支持的结论：

- Mip-NeRF360 相对 3DGS：+0.80 dB PSNR、+0.011 SSIM、LPIPS 下降 0.025；相对 Perceptual-GS 又高 0.33 dB，LPIPS 持平。
- Tanks&Temples 是最完整的提升：相对 Taming-GS +0.36 dB、+0.009 SSIM、LPIPS 下降 0.020。
- Deep Blending 并非所有 metric 最优：PCGS 的 SSIM 最好，但 PSNR 低于 Scaffold-GS 0.17 dB，LPIPS 与原始 3DGS 持平。因此只能称“competitive”，不能称三个数据集全面 SOTA。

在成本上，PCGS / 3DGS 的训练时间与最终 primitives 分别为：Mip-NeRF360 **21m / 2.50M** vs 42m / 3.178M，Tanks&Temples **13m / 1.50M** vs 27m / 1.831M，Deep Blending **18m / 1.75M** vs 36m / 2.805M。由于实现建在已经加速的 Taming-GS 上，论文只将其视为与 Taming-GS 的 runtime 可比证据，不能把减半完全归功于 patch comparison。

### 消融

| Mip-NeRF360 设计 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| 完整 PCGS | **28.01** | **0.826** | **0.189** |
| w/o Patch Comparison | 27.79 | 0.819 | 0.211 |
| w/o Importance Score（均匀随机采样） | 27.84 | 0.822 | 0.203 |
| 对数 budget | **28.01** | **0.826** | **0.189** |
| 线性 budget | 27.87 | 0.824 | 0.194 |
| 指数 budget | 27.84 | 0.821 | 0.198 |

Patch comparison 是最大单项贡献（相对移除后 +0.22 dB、LPIPS -0.022）；边缘重要性也有效（+0.17 dB、LPIPS -0.014）。对数 budget 的优势与其“前快后慢”的 geometry-first training 直觉一致，但该 ablation 未显示不同 budget 是否都严格匹配同一最终点数。

## 亮点与局限

**论文亮点**：

- 将 2D 连续 error region 与 3D densification 相连，补上纯 positional gradient 的盲区；
- Otsu 在 pixel 和 patch 两层自适应阈值化，减少场景特定超参；
- 最大贡献者的逐步“unmasking”避免固定 Top-K，并让 refine 更聚焦；
- budget、importance、patch candidate 三者分工明确：在哪里补、补多少、优先补谁。

**作者承认的限制**：artifact identification 主要依赖单视图 loss；后续可加入 multi-view consistency。

**独立分析**：

- error patch 仍由 GT training image 定义，可能把曝光变化、反射或不一致标定当作 geometry 欠拟合；单视图最大 $\omega_i$ 也不保证该 Gaussian 在其他视角同样是问题源。
- 选择“最大贡献者”会偏向大且高 alpha 的 splat；opacity proportional reduction 虽有助于 unmasking，但论文未消融该操作，尚不能区分收益来自 patch mapping 还是 opacity 调度。
- edge score 有利于高频细节，却可能在平坦但有低频 blur 的区域低估重要性；可尝试结合 patch loss magnitude、uncertainty 或多视图一致性。
- runtime/点数比较混入了 Taming-GS 的底座和不同实现；论文未报告相同 Taming-GS codebase 下仅开关 PCGS 的 wall-clock profiling、render FPS、峰值 VRAM 与多 seed 方差。

## 与相关工作对比

| 方法 | densification 信号 | 点数控制 | 与 PCGS 的差别 |
| --- | --- | --- | --- |
| 原始 3DGS | view-space positional gradient | prune / 固定规则 | 易漏掉梯度小的 blur region |
| Pixel-GS / Abs-GS | 改写或放大 pixel-aware gradient | 各自规则 | 仍主要从 gradient 侧修复 |
| Perceptual-GS | 感知驱动 densification | 自适应 | PCGS 直接从 error patches 反查主导 Gaussian |
| Taming-GS | resource-budgeted 3DGS | budget | PCGS 在其基础上加入 patch candidate 与 edge sampling |
| **PCGS** | Otsu error pixels → error patches → max contributor | 对数 budget + edge importance | 显式解决“哪里坏、谁主导、何时增长” |

## 评分

| 维度 | /10 | 依据 |
| --- | ---: | --- |
| 创新性 | 8.0 | patch-to-dominant-Gaussian 的 densification 映射简洁且有针对性。 |
| 技术可靠性 | 7.4 | 机制与消融相符，但 contributor/opacity unmasking 缺独立拆解。 |
| 实验充分度 | 7.2 | 三类 benchmark、重跑协议和消融较完整；无多 seed、VRAM/FPS。 |
| 写作清晰度 | 7.8 | 流程图、Otsu 两阶段和预算公式清楚；表头 LPIPS 箭头有错误。 |
| 实用/研究价值 | 8.1 | 仅需 rasterizer 输出最大 contributor，易嫁接到现有 densification pipeline。 |

**总体推荐：值得细读和复现。**

## 阅读结论

- **最值得记住的点**：不要仅问某 Gaussian 的 gradient 大不大；先问哪些连续 patch 没重建好，再定位每个 patch 中真正主导像素的 Gaussian。
- **最需要怀疑的点**：error patch 与 single-view maximal contributor 是否稳定对应真实的 3D 欠采样，而非视角/外观噪声。
- **最值得继续验证的点**：加入 multi-view consistency，消融 opacity unmasking，并在同一个 Taming-GS 底座上报告 PCGS 的训练时间、峰值 VRAM、render FPS 和多 seed 方差。

## 相关论文

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 基础 densification 与 rasterizer。
- [Pixel-GS](https://arxiv.org/abs/2406.10219) — pixel-aware gradient density control。
- [AbsGS](https://arxiv.org/abs/2404.10484) — 通过绝对梯度恢复细节。
- [Taming 3DGS](https://arxiv.org/abs/2406.15676) — 资源预算控制，与本文 growth-control 最接近。
- [Steep-GS](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.html) — 学习 split 决策/位置的对照路径。
