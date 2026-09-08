---
title: "Fix False Transparency by Noise Guided Splatting"
subtitle: "在物体内部注入随机变色的高不透明 Gaussian，打破前后表面共同拟合 RGB 的退化解，并用内部填充测量 3DGS 的虚假透明"
authors: "Aly El Hakie, Yiren Lu, Yu Yin, Michael Jenkins, Yehe Liu"
venue: "NeurIPS"
year: 2025
date: 2026-09-09
paper_url: "https://arxiv.org/abs/2510.15736"
code_url: "https://github.com/OpsiClear/noise_guided_splatting"
project_url: "https://opsiclear.github.io/ngs/"
tags:
  - 3D Gaussian Splatting
  - Object-Centric Reconstruction
  - False Transparency
  - Opacity Regularization
  - Novel View Synthesis
  - Evaluation Metric
summary: "本文指出 object-centric 3DGS 中存在标准静态图像指标难以发现的 false transparency：只用 photometric loss 时，低 opacity 前表面与后方 Gaussians 可以共同合成正确 RGB，却在相机移动时暴露内部视差。Noise Guided Splatting 在训练中以多尺度 convex-hull voxelization 构造内部 noise Gaussians，经多视角 depth pruning 和 erosion 保留真正内点，再持续随机化 RGBCMY 颜色，迫使表面学成不透明屏障；同时将这些 infill recolor 后提出 Surface Opacity Score。NGS 在 DTU、Stone 和 OmniObject3D 上将 SOS 分别提高到 0.749、0.922 和 0.736，且保持接近的 NVS 质量，但依赖高质量前景 mask、闭合体积和人为构造的 infill，不适用于真实透明物体或薄结构。"
permalink: /papers/noise-guided-splatting/
---

> **阅读依据**：本笔记基于 [arXiv:2510.15736v1](https://arxiv.org/abs/2510.15736) 的完整正文与 appendix、[官方项目页](https://opsiclear.github.io/ngs/)、[官方代码](https://github.com/OpsiClear/noise_guided_splatting)和 README。论文发表于 **NeurIPS 2025**，代码仓库包含基于 `gsplat` 的训练、infill 生成、可视化和 SOS 评测实现，并采用 CC BY 4.0 license。
>
> **资源状态**：作者公开了 Stone Dataset 与为 DTU 等数据准备的 noise infills，README 指向 Hugging Face 数据集。代码固定了一个 `gsplat` commit；默认配置仍带有作者机器上的绝对数据路径和 `gpu_indices: '7'`，复现前需要覆盖。
>
> **评价边界**：NGS 主要面向有高质量 foreground masks 的 360° object-centric reconstruction。论文没有证明同一机制能直接扩展到无实例分割的大场景，也不应该用于真实玻璃、半透明塑料等本来就需要透射的材质。

## 一句话总结

Noise Guided Splatting（NGS）不是直接把 opacity 正则调大，而是在 3DGS 物体内部放入每步随机变色的噪声 Gaussians，令半透明前表面再也无法与固定后表面“合谋”解释 RGB，于是优化被迫学出能够遮挡内部噪声的高 opacity surface；训练得到的 infill 还可插入任意 3DGS，用 Surface Opacity Score（SOS）在静态视角中量化原本只在相机运动时明显的 false transparency。

## 背景与动机

- **3DGS 的静态帧可以正确但运动观感错误**：某个物体在单张测试图上 PSNR 很高，旋转相机时却像磨砂玻璃一样看到背面或内部结构发生视差漂移。
- **问题不是传统 popping**：depth sorting 或近似 rasterization 导致的 popping 主要表现为视角切换时 primitive 排序突变；false transparency 则来自表面 opacity 本身没有被唯一确定。
- **纯 RGB supervision 没有表面/内部概念**：只要前表面与后方 Gaussians 的 alpha-composited color 等于 GT，photometric loss 就不会区分“一个不透明表面”和“多层半透明结构”。
- **object-centric 360° 更严重**：每个方向的前表面往往都对应另一个视角可见的背面，两者空间距离小、颜色又可能相似，因此 optimizer 很容易共同使用它们解释同一条 ray。
- **标准 IQA 指标失明**：PSNR、SSIM、LPIPS 比较静态图像；如果内部颜色刚好补齐前表面，这些指标不会说明光线实际上穿过了错误的表面。

False transparency 常出现在低纹理、重复图案、specular highlight 或几何复杂区域。它不只影响观看，还会破坏 surface extraction、体积分析、碰撞和 3D printing 等依赖明确物体边界的任务。

**核心研究问题：怎样在不改变 3DGS rasterizer 的前提下打破 opacity 的前后层退化解，并为这种动态伪影建立可在静态渲染中计算的指标？**

## 核心问题

1. 为什么 photometric loss 允许半透明表面与内部 Gaussian 得到与不透明表面相同的训练损失？
2. 单纯让前景 alpha 接近 mask 为什么仍不能稳定解决 360° object reconstruction？
3. 如何构造只位于物体内部、不会污染真实表面和背景的 noise barrier？
4. 噪声若有固定颜色，表面是否又会学会与它互补，重新形成新的退化解？
5. 如何量化 false transparency，而不是依赖研究者反复旋转模型肉眼判断？

## 方法详解

### 整体框架

<div class="mermaid">
flowchart TB
    A[COLMAP images / cameras<br/>foreground masks] --> B[标准 gsplat 训练]
    B --> C[约 6000 steps<br/>初步 surface Gaussians]
    C --> D[Quickhull 构造 convex hull]
    D --> E[多尺度 voxelization<br/>生成 noise Gaussians]
    E --> F[逐视图 depth pruning<br/>删除表面前方 noise]
    F --> G[occupancy erosion<br/>与表面留出 buffer]
    G --> H[冻结 surface 1000 steps<br/>只优化 noise opacity]
    H --> I[低 opacity noise 被 prune]
    I --> J[冻结内部 noise<br/>解冻 surface + reset mean LR]
    J --> K[持续随机 RGBCMY noise color]
    K --> L[训练到 30000 steps]
    L --> M[输出 surface PLY]
    L --> N[单独输出 inside infill PLY]

    N --> O[红色 surface + 绿色 infill]
    M --> O
    O --> P[transmittance map + SOS]
</div>

默认配置在 step 6000 注入 noise，冻结表面并训练 noise opacity 1000 steps，之后 surface 继续训练到 30k。内部 Gaussians 在导出时与 surface 分开保存，因此部署只使用 surface PLY 时不需要承担内部点的推理开销；infill 可留作诊断或继续研究。

### 1. False transparency 的数学来源

对像素 $\mathbf p$，3DGS 按深度从前到后混合覆盖该像素的 splats：

$$
\mathbf C(\mathbf p)=
\sum_{i\in\mathcal N(\mathbf p)}
\alpha_i\mathbf c_i
\prod_{j<i}(1-\alpha_j).
$$

$\alpha_i$ 是第 $i$ 个 splat 的 opacity，$\mathbf c_i$ 是颜色，乘积是到达它之前剩余的 transmittance。训练主要优化：

$$
\mathcal L_{\mathrm{photo}}
=(1-\lambda)\frac{1}{|\mathcal P|}
\sum_{\mathbf p\in\mathcal P}
\|\mathbf C(\mathbf p;\Theta)-\mathbf I_{GT}(\mathbf p)\|_1
+\lambda\mathcal L_{\mathrm{D-SSIM}}.
$$

设 ray 首次命中的表面 splat 为 $s$，后方集合为 $\mathcal B$，则像素可拆成：

$$
\mathbf C(\mathbf p)=
\underbrace{\alpha_s\mathbf c_s}_{\text{front surface}}+
\underbrace{(1-\alpha_s)
\sum_{i\in\mathcal B}\alpha_i\mathbf c_i
\prod_{s<j<i}(1-\alpha_j)}_{\text{leaking background}}.
$$

只要 $\alpha_s<1$，就可以调整后方 opacity/color，让两项之和仍等于真实的不透明表面颜色。因此 image loss 对沿 ray 的颜色/opacity 分配并不可辨识：它约束最终 RGB，却不约束“颜色应该在哪一层终止”。

相机不动时这个解可以完美；相机移动后，前后层的 parallax 不同，内部图案便会相对表面滑动，false transparency 才显现。

### 2. Alpha-consistency loss：必要但不充分

给定 rendered alpha $A_i\in[0,1]$ 与 binary foreground mask $M_i\in\{0,1\}$，背景抑制项为：

$$
\mathcal L_b=\frac{1}{|\mathcal P|}
\sum_{i\in\mathcal P}A_i(1-M_i),
$$

让 mask 外 alpha 接近 0。互补的 foreground opacity loss 为：

$$
\mathcal L_f=\frac{1}{|\mathcal P|}
\sum_{i\in\mathcal P}(1-A_i)M_i,
$$

让 mask 内累计 alpha 接近 1。二者合并得到：

$$
\mathcal L_a=\mathcal L_f+\mathcal L_b
=\frac{1}{|\mathcal P|}\sum_{i\in\mathcal P}|A_i-M_i|.
$$

这相当于让 rendered alpha silhouette 对齐前景 mask。对只有部分视角的场景，它能很好地区分对象与背景；但 360° 情况下，ray 上多个表面都属于不同视角中的 foreground，累计 alpha 达到 1 并不保证 opacity 集中在最前表面。后层仍可以分担颜色和透明度。

实验中的 `GSplat+α` 即加入 $\mathcal L_a$ 的强基线。它已经显著改善 SOS，但 NGS 在三个数据集上仍进一步提高，说明 silhouette alpha consistency 不能完全替代表面优先约束。

### 3. 用 convex hull 初始化内部 noise

NGS 在已有 surface Gaussian means 上运行 Quickhull，得到一个粗 convex hull，再将 hull voxelize。每个 occupied voxel 映射成一个 noise Gaussian。

直接填 convex hull 会把凹形物体的外部空区也包含进来。例如杯把内洞、石头凹槽或腿之间空间都可能被误填。作者因此不把 convex hull 当作最终内部，而只把它当作高 recall proposal，后面再根据真实视图 carve。

初始 noise 使用较低 opacity，位置和尺度由 voxel 决定。代码默认起始 `voxel_resolution: 32`，并做 coarse-to-fine multi-resolution injection：高分辨率阶段只在尚未被低分辨率 noise 覆盖的 occupied voxels 中添加新点，从而减少均匀超密填充。

### 4. Depth pruning 与 erosion

对每个 training view，系统沿 ray 使用 3DGS depth ordering 检查 noise 与 surface：若 noise 出现在该像素第一个 surface Gaussian 之前，就说明它在可见外部，应删除。对所有视图累计该 pruning mask，保留从训练相机看始终被 surface 遮挡的 noise。

之后对 voxel occupancy 做 binary erosion，让内部 noise 与估计表面之间保留最小 buffer：

- buffer 太小，noise 会直接参与表面颜色，虽然 SOS 很高但会损坏 NVS；
- buffer 太大，薄处或凹处缺少 barrier，false transparency 可能残留。

消融中 `w/o Erosion` 达到 SOS 1.000，却让 PSNR 从 32.201 降到 30.785、LPIPS 从 0.145 恶化到 0.225，直接证明“最不透明”不等于“最好重建”。

### 5. Noise-only fine-tuning

完成几何 pruning 后，surface Gaussians 冻结约 1000 steps，只训练 noise opacity。离表面过近或仍可见的 noise 无法稳定隐藏，opacity 会降低并被原有 prune rule 删除；真正内部的 noise 则保留。

这一步的作用是用 optimization 再做一次 soft visibility filtering，补足纯 depth-order carving 的离散误差。`w/o Pruning` 的 SOS 仍有 0.952，但 PSNR/SSIM 低于完整方法，说明清掉不合适 noise 对视觉质量有帮助。

### 6. 为什么 noise color 必须持续随机

若内部噪声颜色固定，surface 可能再次找到互补解：学习一种半透明颜色，与固定 noise 混合后仍匹配 GT。NGS 在每个 iteration 从 RGBCMY 六色中重新随机 noise color；这些 RGB primary/secondary colors 在 RGB/HSV 空间中通常与自然颜色具有较强对比。

持续随机化使后方颜色成为高方差干扰。最稳妥的降 loss 方式不再是拟合某个固定组合，而是让第一层 surface 的 opacity 变高，把所有内部随机颜色都遮住。

`w/o Color reset` 仍有 0.962 SOS，但标准质量略低于完整方法；说明颜色随机化不是决定 SOS 的唯一因素，却有助于避免 surface/noise 共适应。

### 7. Guided surface training 与学习率重置

noise fine-tuning 完成后：

- noise 的 means、opacity、scale 和 rotation 被冻结；
- surface 解冻；
- surface Gaussian means 的 learning-rate schedule 被重置；
- noise color 继续随机；
- 其余训练沿用默认 GSplat。

noise barrier 会突然改变 alpha blending 和 loss landscape。若 surface 的 position LR 已在 6000 steps 前衰减过多，它无法重新排列以适应新遮挡条件。消融 `w/o LR reset` 的 PSNR 只有 26.136、LPIPS 0.416、SOS 0.545，是全部变体中最严重的质量退化，说明 LR reset 是不可省略的配套设计，而不是普通实现细节。

### 8. Surface Opacity Score（SOS）

评测时将 surface Gaussians 统一染红、内部 infill 染绿。若表面正确不透明，绿色被完全挡住；若存在 false transparency，green channel 会泄漏。对 view $i$，把绿色通道视作 transmittance map $T_i$，mask 为 $M_i$：

$$
SOS_i=
\frac{\log\left(\frac{\sum T_i}{\sum M_i}+\epsilon\right)}
{\log(\epsilon)},
\qquad \epsilon=10^{-10}.
$$

因为分母为负数：

- 完全不透明时 $T\approx0$，$SOS\approx1$；
- 完全透明时平均 transmittance 接近 1，$SOS\approx0$。

对数归一化让极小的漏光仍能被区分。除了 SOS，作者还比较插入绿色 infill 前后的 PSNR/SSIM/LPIPS，记为带 `*` 的指标。如果 surface 半透明，绿色噪声会大幅破坏重建图像，标准指标明显下降；若 surface 真正不透明，插入前后指标应几乎一致。

**代码审查注意点**：论文称有 mask 时应在前景内计算 $T_i$；当前 `evaluator.py` 实现直接对整个裁剪后的 green channel 求和，再除以 mask 像素数，没有显式乘 $M_i$。如果 irregular mask 外存在可见绿色，SOS 会被额外影响。作者提供的 infill/黑背景可能使影响较小，但独立复现时应加入 `T_i * M_i` 对照。

## 实验关键数据

### 实验设置

- **DTU**：appendix 报告 7 个对象。
- **OmniObject3D**：appendix 报告 7 个对象。
- **Stone Dataset**：作者采集超过 100 种石头，每个样本 240 张、3000×4000 图像，覆盖 6 个纬度 × 40 个经度的上半球视角，并保留 16-bit raw Bayer；主结果 appendix 列出 10 个样本。
- 所有数据用 MVANet 生成高质量 foreground masks，采用 7:1 train/test split。
- 训练基于 GSplat，在 NVIDIA L40S 上运行；base method 通常不超过 8 GB VRAM。
- baseline：3DGS、Gaussian Opacity Fields（GOF）、StopThePop，以及加入 $\mathcal L_a$ 的 `GSplat+α`。

NGS 默认在 step 6000 注入 noise，noise-only fine-tuning 1000 steps，总训练 30k。基础训练时间 DTU 16 min、OmniObject3D 11 min、Stone 18 min；NGS 约增加 1 min，并因内部 primitives 增加约 50% memory。Discussion 将 Gaussian count overhead 概括为 30–50%。

### 主实验

| Dataset / 方法 | PSNR ↑ | PSNR* ↑ | LPIPS ↓ | LPIPS* ↓ | SOS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| DTU / 3DGS | 25.575 | 22.967 | 0.180 | 0.250 | 0.147 |
| DTU / GSplat+α | **25.435** | 25.263 | **0.183** | 0.186 | 0.598 |
| DTU / **NGS** | 25.428 | **25.427** | 0.192 | **0.192** | **0.749** |
| Stone / 3DGS | **34.610** | 27.551 | 0.055 | 0.222 | 0.140 |
| Stone / GSplat+α | 33.832 | 33.823 | 0.062 | 0.062 | 0.891 |
| Stone / **NGS** | 34.148 | **34.148** | **0.053** | **0.053** | **0.922** |
| OmniObject / 3DGS | 29.300 | 27.456 | 0.069 | 0.116 | 0.215 |
| OmniObject / GSplat+α | 33.575 | 33.350 | 0.060 | 0.064 | 0.642 |
| OmniObject / **NGS** | **33.619** | **33.578** | 0.060 | **0.060** | **0.736** |

最有说服力的不是普通 PSNR，而是插入 infill 后的稳定性：

- DTU 3DGS 的 PSNR 因绿色 infill 从 25.575 跌至 22.967，NGS 只从 25.428 变为 25.427；
- Stone 3DGS 从 34.610 跌至 27.551，NGS 的 34.148 完全不变；
- OmniObject 3DGS 从 29.300 跌至 27.456，NGS 仅下降 0.041 dB。

相对 `GSplat+α`，NGS 的 SOS 分别提高 0.151、0.031 和 0.094，说明内部 noise 的贡献超出 alpha silhouette loss。不过 DTU 上 NGS 的普通 PSNR/SSIM/LPIPS 略逊于 `GSplat+α`，OmniObject 的 SSIM 也低 0.001，所以准确结论是 **透明度显著改善、标准质量基本保持 competitive**，而不是所有指标一致提升。

GOF 与 StopThePop 虽然针对排序/表面一致性问题设计，三个数据集的 SOS 仍低，支持 false transparency 与 popping/depth approximation 是不同故障机制。

### 消融实验

| Stone ablation | PSNR ↑ | LPIPS ↓ | SOS ↑ | 说明 |
| --- | ---: | ---: | ---: | --- |
| **完整 NGS** | **32.201** | 0.145 | 0.969 | 质量与 opacity 的折中 |
| w/o Erosion | 30.785 | 0.225 | **1.000** | noise 太贴近表面，最不透明但画质明显下降 |
| w/o Pruning | 31.496 | 0.151 | 0.952 | 错放 noise 污染 surface optimization |
| w/o $\mathcal L_f$ | 31.252 | **0.126** | 0.379 | 感知指标不差，但 false transparency 严重，说明普通 IQA 会漏诊 |
| w/o LR reset | 26.136 | 0.416 | 0.545 | surface 无法适应突然改变的 blending landscape |
| Random Background | 30.205 | 0.136 | 0.467 | 随机背景不能替代内部 line-of-sight barrier |
| w/o Color reset | 31.442 | 0.154 | 0.962 | 固定 noise 允许一定共适应，质量略差 |

`w/o L_f` 是论文最有启发性的消融：LPIPS 反而从 0.145 降到 0.126，看似“更好”，但 SOS 从 0.969 崩到 0.379，且 infill-conditioned LPIPS 从 0.126 变成 0.135。这直接展示了为什么必须引入专门的 transparency metric。

需要注意，Table 1 的 Stone NGS 为 PSNR 34.148 / SOS 0.922，而 Table 2 “Mean over Stone Dataset” 的完整方法为 32.201 / 0.969。论文没有清楚解释两张表是否采用不同样本、设置或重复实验，因此不能把两组数值视为完全同 protocol 的复现。

### 效率与模型规模

- NGS 在论文硬件上平均只增加约 1 分钟训练时间；
- 内存约增加 50%，noise count 取决于体积和 voxel resolutions；
- surface 和 inside Gaussians 会分开保存，正常部署可只加载 surface；
- appendix 的 surface Gaussian 数量并非总是增加：DTU/OmniObject 上 NGS surface count 低于 `GSplat+α`，Stone 上更高；但论文没有单独汇总 infill 数量与总训练峰值。

作者建议后续把 noise 简化为只含 sphere scale、opacity 和 color 的 primitives，去掉 spherical harmonics、quaternion 和各向异性 scale，以降低训练内存。

### 关键发现

1. False transparency 是 alpha-compositing 与 image-only supervision 的可辨识性问题，不等同于 depth sorting popping。
2. 前景 alpha loss 已能显著改善 opacity，但在 360° 前后表面都属于 foreground 时仍不充分。
3. 内部随机颜色 barrier 让“提高第一层 opacity”成为跨 iteration 最稳定的解。
4. SOS 与 infill-conditioned IQA 能暴露普通 NVS 指标隐藏的透明度错误。
5. NGS 对受控 softbox Stone 最有效；DTU/Omni 的方向光、阴影和 view-dependent appearance 会削弱前景/透明度区分。
6. Erosion、foreground loss 与 LR reset 共同决定 quality-opacity tradeoff，不能只追求 SOS=1。

## 亮点与洞察

### 论文亮点

- **识别了一个真实但长期缺少名称的 artifact**：false transparency 的动态症状、数学退化解和标准指标盲区对应清楚。
- **干预机制直接针对可辨识性**：不是再加一个泛化 opacity penalty，而是改变 ray 上可用的解释，使后层颜色不可预测。
- **训练与评测共享同一资产**：noise infill 既是 optimization barrier，也是可插入任意模型的 diagnostic probe。
- **方法对 renderer 改动小**：建立 infill 后仍使用标准 splatting、photometric training 和 ADC，只需管理两组 Gaussians 的冻结与颜色。
- **消融有机制解释力**：尤其 `w/o L_f` 和 `w/o Erosion` 证明好 NVS 指标、高 SOS 与真实质量并非单调一致。
- **公开实现较完整**：提供从训练、独立 PLY 导出到 PSNR*/SOS evaluation 的可执行代码，并固定 gsplat commit。

### 我的洞察

- **个人分析：NGS 类似一种 adversarial occlusion probe。** 内部随机色不是要被重建的内容，而是持续挑战表面：“如果你不够不透明，我就让 loss 变坏。”这比直接规定每个 splat 的 alpha 更尊重多层 Gaussian 的表示自由。
- **个人分析：它在修复一个 gauge freedom。** RGB compositing 允许颜色/opacity 沿 ray 重新分配而输出不变；noise 随机化相当于选定一个偏好的 gauge——把解释集中到首个表面。
- **个人分析：foreground mask 只约束 silhouette，noise 才约束 termination depth。** $\mathcal L_a$ 说“这条 ray 最终要不透明”，NGS 进一步说“应该在前表面就结束”。
- **个人分析：SOS 是 intervention-based metric。** 它不是从原模型直接读取 opacity，而是插入已知 probe 后观察响应；这种思想可迁移到几何空洞、遮挡鲁棒性和 view-dependent leakage 测试。
- **个人分析：最佳 noise 不是越近、越密越好。** `w/o Erosion` 的 SOS=1 但外观更差，说明必须在 surface 与 probe 间保留可优化的安全带。
- **个人分析：受控光照提升来自更稳定的 surface appearance，不一定只来自 opacity。** DTU/Omni 的阴影和高光可能需要 appearance model 或 lighting decomposition 配合，单靠 noise barrier 无法区分真实 view dependence 与错误透射。

## 局限与展望

### 作者明确承认的局限

- 依赖高质量 foreground segmentation；复杂边缘或错误 mask 会导致 noise 放错位置；
- 假设目标材料完全不透明，对玻璃和半透明塑料会错误地强制 opacity；
- 需要已有 surface Gaussian cloud 足以形成覆盖整个对象的合理 convex hull；
- 尚未验证 large-scale scene，多个对象和复杂空间关系需要新的初始化与分割策略；
- thin structures 难以容纳内部 noise；
- noise 通常令总 Gaussian 数与内存/渲染训练成本增加 30–50%。

### 独立分析

- **SOS 依赖 probe 本身**：infill density、opacity、颜色、与 surface 的距离和可见区域都会影响分数；跨论文比较必须固定同一 infill 和 evaluator。
- **SOS 不是材料透明度估计**：它衡量“指定绿色内部 probe 穿过 surface 的程度”，不能直接转换为真实世界 transmittance 或 extinction coefficient。
- **当前代码的 mask 实现与公式略有差异**：green-channel numerator 未显式乘 foreground mask，可能把 irregular mask 外 leakage 算入分数。
- **Convex hull 对非凸/多孔拓扑是粗先验**：depth pruning 只利用训练视图；未观测凹腔、洞和底部可能保留本应位于外部的 noise。
- **评测对象数量有限**：appendix 主表只有 10 Stone、7 DTU、7 OmniObject entries；“超过 100 个 Stone”没有完整逐样本主结果。
- **Stone capture 分布较理想化**：转台、softbox、密集上半球视角有利于 mask、COLMAP 和稳定光照，不能代表随手手机扫描。
- **真实透明与错误透明的区分需要语义**：单靠 mask 和 photometric loss 不知道对象是否是玻璃；未来需要 material prior 或用户标注。
- **内部 noise 可能影响几何提取**：虽然部署可导出 surface-only PLY，但训练期间的强遮挡偏置是否会令 surface 厚度、scale 或位置产生系统偏差，论文没有 mesh ground truth 评价。
- **baseline 范围偏窄**：主要比较 3DGS、GOF、StopThePop 和自己的 alpha-loss baseline，缺少专门的 opacity/surface regularization、2DGS、SuGaR 或 mesh-oriented methods 的统一对照。
- **缺少多 seed 方差**：noise color、初始化、ADC 与数据 split 都含随机性，主表没有重复训练置信区间。
- **两张 Stone 表的默认结果不一致**：主实验与消融完整方法分数差距明显，实验条件说明不足。
- **“plug-and-play”仍有前置成本**：需要 masks、convex hull、完整逐视图 depth pruning、额外 freeze schedule 和 LR reset，不是只加一行 loss。
- **许可选择需留意**：仓库把软件整体置于 CC BY 4.0，而非典型 software license；商用或二次分发前应确认依赖和数据集各自许可。

### 建议的后续实验

1. 用合成 opaque/translucent objects 提供真实 opacity/geometry ground truth，验证 SOS 与实际 transmittance、surface thickness 的相关性。
2. 固定一组标准 infills，扫描 voxel resolution、erosion distance、noise opacity 和 density，报告 SOS 对 probe 参数的敏感性。
3. 修正 evaluator 为显式 `T_i * M_i`，并比较 binary mask、rendered alpha mask 与无 mask 设置。
4. 对每个数据集运行多个 random seeds，报告 NVS、SOS、surface Gaussian count、noise count 与 peak VRAM 的 mean/std。
5. 与 2DGS、GOF、SuGaR、opacity regularization 和 mesh extraction-oriented methods 做同一 SOS protocol 对比。
6. 在真实玻璃、蜡、玉石与混合材质数据上加入 transparency classifier 或 per-region opacity prior，避免一刀切。
7. 对杯子、环、网格、细枝和开放薄壳测试 convex-hull carving failure，比较 visual hull、TSDF/SDF 或 learned occupancy 初始化。
8. 将 object segmentation 与 Segment Any 3D Gaussians 结合，评估多对象 scene-level NGS 的内存和遮挡冲突。
9. 移除训练 noise 后提取 mesh，比较 Chamfer、normal consistency、watertightness 和物理碰撞稳定性。

## 与相关工作的对比

| 方法 | 主要问题 | 核心干预 | 是否直接约束 false transparency | 主要代价 |
| --- | --- | --- | ---: | --- |
| Original 3DGS | photometric reconstruction | alpha blending + ADC | 否 | opacity 沿 ray 不可辨识 |
| StopThePop | view-dependent depth ordering / popping | hierarchical sorting | 否 | 改善排序一致性，不保证表面不透明 |
| Gaussian Opacity Fields | surface extraction 与 opacity field | opacity-field / level-set interpretation | 间接 | 仍可能在 SOS probe 下泄漏 |
| GSplat + $\mathcal L_a$ | silhouette alpha consistency | mask 内满 alpha、mask 外零 alpha | 部分 | 只约束累计 alpha，不指定首表面 |
| Random background | 背景颜色过拟合 | 每步改变背景颜色 | 间接 | 不阻断对象内部前后表面共同解释 |
| **NGS** | **前后表面 opacity-color 退化解** | **内部随机色 noise barrier + alpha loss** | **是** | **mask、体积假设、30–50% 训练内存** |

## 启发与关联

- **通过主动干预评价不可观测属性**：当普通 test image 无法区分两个解时，向系统插入已知 probe，再测响应，比继续设计被动图像指标更有效。
- **训练期增加约束、推理期移除辅助变量**：noise 是 optimization scaffold，最终 surface-only 模型仍可按标准 3DGS 渲染。
- **Mask loss 与 3D barrier 互补**：2D silhouette 规定对象投影，3D infill 规定 ray termination；类似组合可用于 hollow geometry、密度场和 volumetric rendering。
- **假设：用 uncertainty-guided sparse probes 替代均匀填充**。仅在低 texture、前后表面颜色相似或 SOS 预估差的 ray 附近注入 noise，可能降低 30–50% overhead。
- **假设：对 noise 做 curriculum**。从低 opacity/低频颜色变化逐渐增强，并根据局部 surface opacity 自动退出，可减少突然改变 loss landscape 对 LR reset 的依赖。
- **假设：从硬 RGB noise 扩展到 feature noise**。在 latent/feature splatting 中随机化内部语义特征，可能同时约束 appearance 与 downstream feature leakage。
- **与 Blur Trap 的联系**：Blur Trap 论文通过随机播种/分裂修复“缺探索和容量”的 Gaussian allocation；NGS 则故意加入内部 Gaussians 改写遮挡梯度，解决“已有多层解释但 opacity 分配错误”。二者都把额外 primitives 当训练期优化工具，但目的相反。

## 评分

| 维度 | 评分（10 分） | 理由 |
| --- | ---: | --- |
| 创新性 | 8.4 | 对 false transparency 的命名、数学解释，以及用内部随机 noise 同时训练和评价都很有辨识度；核心机制简单但新颖。 |
| 技术可靠性 | 7.8 | 公式与消融支持机制，三个数据集均提高 SOS；依赖 convex hull/mask，SOS 对 infill 参数敏感，当前 evaluator 的 mask 实现还有核对空间。 |
| 实验充分度 | 7.4 | 有主表、逐样本 appendix、消融、时间和内存；主表仅 24 个对象条目、无多 seed、baseline 较少，Stone 两表条件不清。 |
| 写作清晰度 | 8.0 | 前后表面退化解和训练 schedule 直观；“factor of two”“across methods”等部分概括比表格更强，protocol 细节仍可更严谨。 |
| 实用 / 研究价值 | 8.2 | 代码和数据已公开，surface-only 导出不增加部署成本，SOS 也能独立评价其他 PLY；适用面目前集中于 opaque object-centric capture。 |

**总体推荐：值得细读并复现 SOS evaluator。** 即使未来有更轻量的透明度修复方法，论文提出的“插入内部 probe 检测 surface leakage”仍可能成为独立、有生命力的评测工具。

## 阅读结论

- **最值得记住的点**：3DGS 可以在每张静态图上都看似正确，却因前表面与后层共同解释 RGB 而在运动时呈现虚假透明；NGS 用每步变色的内部噪声直接破坏这种共谋。
- **最需要怀疑的点**：SOS 是否在不同 infill 构造、mask 质量和非凸拓扑间保持可比，以及高 SOS 是否同时对应正确 surface geometry，而不只是更强 opacity。
- **最值得复现或继续验证的点**：固定 surface PLY，系统扫描 infill density/erosion/opacity，修正 mask numerator 后复现 SOS，并在真实 opaque、translucent、thin-shell 与 scene-level 数据上验证。

## 相关论文与资源

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 基础 renderer、alpha compositing 与 ADC。
- [StopThePop](https://arxiv.org/abs/2402.00525) — 处理 Gaussian depth sorting 与 popping；和 false transparency 属于不同 view inconsistency 机制。
- [Gaussian Opacity Fields](https://arxiv.org/abs/2404.10772) — 从 opacity field 角度改进表面重建，是论文的主要比较方法之一。
- [3D Gaussian Splatting as Markov Chain Monte Carlo](https://arxiv.org/abs/2404.09591) — 以 opacity distribution 和 relocation 重写 densification，论文讨论其低 opacity 倾向与本问题的关系。
- [Revising Densification in Gaussian Splatting](https://arxiv.org/abs/2404.06109) — alpha regularization / Revised ADC 思路，foreground loss 的机制前身。
- [Segment Any 3D Gaussians](https://arxiv.org/abs/2312.00860) — 未来把 object-centric NGS 扩展到多对象场景时可用的 3D segmentation 工具。
- [Exploration Matters for Escaping the Blur Trap 阅读笔记](/papers/exploration-matters-blur-trap/) — 同样通过训练期额外 Gaussians 修复优化病理，但关注 depth exploration 与 densification failure。
- [项目页](https://opsiclear.github.io/ngs/) / [论文](https://arxiv.org/abs/2510.15736) / [官方代码](https://github.com/OpsiClear/noise_guided_splatting) / [数据集](https://huggingface.co/datasets/OpsiClear/NGS)
