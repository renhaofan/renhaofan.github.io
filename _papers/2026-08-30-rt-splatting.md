---
title: "RT-Splatting: Joint Reflection-Transmission Modeling with Gaussian Splatting"
subtitle: "用统一的表面—体积高斯表示联合建模反射与透射"
authors: "Ji Shi, Xianghua Ying, Bowei Xing, Ruohao Guo, Wenzhen Yue"
venue: "CVPR 2026 Highlight"
year: 2026
date: 2026-08-30
paper_url: "https://arxiv.org/abs/2605.18263"
code_url: "https://github.com/sjj118/RT-Splatting"
project_url: "https://sjj118.github.io/RT-Splatting"
tags:
  - Gaussian Splatting
  - Reflection
  - Transmission
  - Transparent Reconstruction
  - Novel View Synthesis
  - Deferred Shading
summary: "RT-Splatting 将每个 2D Gaussian 的几何占据率与光学不透明度解耦，使同一组 primitives 能以表面形式渲染高频反射、以体积形式累积清晰透射，并用高光感知梯度门控减少透射分支为反射残差生成的漂浮伪影。"
permalink: /papers/rt-splatting/
---

> **阅读依据**：本文笔记基于 arXiv:2605.18263 的正文、补充材料和官方代码仓库。论文与仓库将其标注为 CVPR 2026 Highlight；实验表格、公式和实现超参数均以论文源文件为准。仓库已公开训练代码、透明区域 masks 和作者自采数据。

## 背景与动机

- **领域现状**：3D Gaussian Splatting（3DGS）能以实时速度完成 novel view synthesis，但标准 SH appearance 难以稳定拟合高频镜面反射。GaussianShader、3DGS-DR、Ref-GS、EnvGS 等工作因此引入显式 shading 或 deferred shading。
- **现有痛点**：薄玻璃、车窗和塑料膜同时具有清晰透射与强反射。为了拟合反射，普通 3DGS 容易在表面后方生成错误 Gaussian “floaters”；它们又会遮挡本应透过透明表面看到的背景。
- **单一 opacity 的冲突**：deferred shading 希望透明表面拥有高 opacity，以便稳定提取 normal、roughness 等首表面属性；体渲染则希望它拥有低 opacity，让背景光穿过。一个参数无法同时表达“几何上存在、光学上透明”。
- **已有透明重建的限制**：TransparentGS 等方法常先单独重建并冻结背景，再处理透明物体。如果车内等背景只通过车窗可见，遮掉透明区域后就没有直接背景观测，分阶段重建会失败。

**核心研究问题：能否用同一组 Gaussian primitives，在不预先分离背景的条件下，同时恢复薄透明表面的高频反射和其后方的清晰内容，并保持实时渲染？**

## 核心问题

1. **如何同时表示“实心表面”和“低光学遮挡”？** 这要求把几何存在性与透光性从标准 opacity 中拆开。
2. **如何在一条渲染管线中兼容反射与透射？** 高频反射适合 first-hit surface + deferred shading，透射背景则需要沿射线进行 volumetric compositing。
3. **如何避免两分支互相代偿？** 当反射网络拟合不了复杂高光时，图像损失可能迫使透射分支制造虚假几何来解释残差。
4. **如何约束新增自由度？** 任意位置都可能出现“高 occupancy、近零 opacity”的不可见 ghost surface，需要额外监督和 density control。

## 方法详解

### 整体框架

RT-Splatting 建立在 2DGS 的 surface-aligned Gaussian disks 上，完整流程为：

1. 每个 Gaussian 学习几何占据率 $\sigma$ 与光学不透明度 $\alpha$，同时维护 normal、roughness、material feature、散射颜色和 transmissivity 等属性。
2. **Deferred surface pass** 使用 $\sigma$ 聚合第一命中表面的属性，形成 G-buffers，再预测高频 specular reflection。
3. **Forward volume pass** 使用有效不透明度 $\alpha_{\mathrm{eff}}=\sigma\alpha$ 沿射线累积背景 radiance，使透明表面不会错误遮挡背景。
4. 将透射背景与材料内部散射合成 subsurface component，再与 specular component 合成最终颜色。
5. 反向传播时，根据局部 specular variance 抑制流向 transmission branch 的误导梯度。
6. 用 SAM2 透明区域 mask、normal consistency 和图像重建损失联合优化全部 Gaussian 与 shading network。

### Occupancy–Opacity Factorization

标准 3DGS 的 pixel weight 为：

$$
w_i=\alpha_i\mathcal G_i\prod_{j<i}(1-\alpha_j\mathcal G_j),
$$

其中 $\mathcal G_i$ 是第 $i$ 个 Gaussian 在该像素处的 2D kernel value。这里的 $\alpha_i$ 同时决定 Gaussian 是否像表面一样被命中，以及它遮挡后方光线的程度。

RT-Splatting 将其拆成：

- **Geometric occupancy** $\sigma\in[0,1]$：射线与该 Gaussian 所代表物质发生几何交互的概率；
- **Optical opacity** $\alpha\in[0,1]$：发生交互后，光被吸收或散射的条件概率；
- **Effective opacity** $\alpha_{\mathrm{eff}}=\sigma\alpha$：真正用于体积颜色合成的遮挡强度。

透明表面因此可以取“高 $\sigma$、低 $\alpha$”：在几何上形成稳定表面，但对透射光只有较弱衰减。

对 normal、roughness 等表面属性 $\mathbf a_i$，作者只用 occupancy 计算 first-hit probability：

$$
\mathbf A=\sum_i p_i\mathbf a_i,\qquad
p_i=\sigma_i\mathcal G_i\prod_{j<i}(1-\sigma_j\mathcal G_j).
$$

$p_i$ 表示第 $i$ 个 Gaussian 是射线首个命中表面的概率。直观上，surface pass 不关心物体是否透明，只关心“表面在哪里”；volume pass 才关心它吸收多少光。

### 混合 Deferred–Forward Renderer

#### 表面反射

occupancy pass 先把 normal $\mathbf n$、roughness $\rho$ 和 material feature $\mathbf z$ 聚合进 G-buffers。随后采用与 Ref-GS 相近的 specular shading network：

$$
\mathbf C_{\mathrm{spec}}=f_{\mathrm{spec}}(\mathbf v,\mathbf n,\rho,\mathbf z),
$$

其中 $\mathbf v$ 为观察方向。逐像素 shading 避免每个 Gaussian 独立预测高频颜色，更适合表达细锐、空间一致的反射。

#### 透射与内部散射

forward pass 使用 $\alpha_{\mathrm{eff}}$ 累积背景颜色 $\mathbf C_{\mathrm{trans}}$。为表达有色玻璃或塑料膜的吸收、内部散射，作者为表面再学习散射颜色 $\mathbf C_{\mathrm{scatter}}$ 与 transmissivity ratio $\tau$：

$$
\mathbf C_{\mathrm{sub}}
=\tau\mathbf C_{\mathrm{trans}}
+(1-\tau)\mathbf C_{\mathrm{scatter}}.
$$

$\tau$ 越大，最终外观越接近表面后的真实背景；$\tau$ 越小，材料自身颜色与散射贡献越强。

#### 最终合成

specular network 额外输出 attenuation $\beta\in[0,1]$：

$$
\mathbf C=\mathbf C_{\mathrm{spec}}+\beta\mathbf C_{\mathrm{sub}}.
$$

强反射区域可以通过较小的 $\beta$ 压低背景透射，弱反射区域则保留细节。作者认为直接使用 Fresnel blend 难以适配 tone mapping 和真实相机非线性，因此采用 learnable attenuation。

这是一种实用 appearance model，而不是严格能量守恒的光传输：$\mathbf C_{\mathrm{spec}}$ 与 $\beta$ 都由网络学习，反射、吸收、曝光和相机响应仍可能互相补偿。

### Specular-Aware Gradient Gating

即使 forward pass 已分开，最终图像损失仍同时优化两个分支。高频反射的残差会给 transmission branch 错误信号，促使它在透明表面后生成 floaters。

作者以 specular image 的局部方差估计反射复杂度：

$$
g(x)=\exp\left(
-k\operatorname{Var}_{p\in\mathcal N(x)}
[\mathbf C_{\mathrm{spec}}(p)]
\right).
$$

- 局部反射平滑时，variance 较小，$g(x)\approx1$，背景继续接受完整监督；
- 高频反射复杂时，$g(x)$ 变小，流向 transmission branch 的梯度被削弱；
- 它不改变 forward rendering，只改变 backward gradient。

具体用 partial stop-gradient 实现：

$$
\widetilde{\mathbf C}_{\mathrm{trans}}
=(1-g)\operatorname{sg}(\mathbf C_{\mathrm{trans}})
+g\mathbf C_{\mathrm{trans}}.
$$

前向计算中，门控后的 transmission color 与原值完全相同；反向时梯度才会乘以 $g$。补充材料使用 $3\times3$ window，并设置 $k=4$。

### Mask Regularization 与 Density Control

factorization 会产生新歧义：高 $\sigma$、近零 $\alpha$ 的 Gaussian 几乎不影响颜色，所以可能在无高光区域任意堆积。作者利用 SAM2 得到透明 mask $\mathbf M$，监督首表面 opacity map：

$$
\mathcal L_{\mathrm{mask}}
=\operatorname{BCE}(1-\mathbf M,\alpha).
$$

透明区域 $\mathbf M=1$ 被鼓励学习低 $\alpha$，非透明区域学习高 $\alpha$。与分割后分别训练不同场景组件的方法不同，mask 在这里仅作为 regularization，所有部分仍联合优化。

标准 2DGS 每 3K iterations 重置 opacity。RT-Splatting 改为每 1.5K iterations 在 $\sigma$ 与 $\alpha$ 之间交替 reset，并依据 $\sigma$ 而非 $\alpha$ pruning，避免把光学透明但几何真实的表面删掉。

### 损失函数

图像损失为 L1、D-SSIM 与 EnvGS perceptual loss 的组合：

$$
\mathcal L_{\mathrm{img}}
=(1-\lambda)\mathcal L_1
+\lambda\mathcal L_{\mathrm{D\text{-}SSIM}}
+\lambda_{\mathrm{perc}}\mathcal L_{\mathrm{perc}},
$$

其中 $\lambda=0.2$、$\lambda_{\mathrm{perc}}=0.01$。总目标为：

$$
\mathcal L=\mathcal L_{\mathrm{img}}
+\lambda_n\mathcal L_n
+\lambda_{\mathrm{mask}}\mathcal L_{\mathrm{mask}},
$$

$\mathcal L_n$ 是继承自 2DGS 的 rendered-normal/depth-gradient consistency，$\lambda_n=0.05$、$\lambda_{\mathrm{mask}}=0.01$。

## 实验关键数据

### 实验设置

- 公共数据共 6 个场景：Ref-Real 的 Sedan、Toycar，NeRF-Casting 的 Compact、Hatchback，EnvGS 的 Audi，以及 Tanks & Temples 的 Truck。
- 作者自采 Van、Swab 两个场景，每个使用手机拍摄 220–240 个 views。
- 指标同时在整幅图与透明区域计算：PSNR、SSIM 越高越好，LPIPS 越低越好。
- Baselines：3DGS、2DGS、GaussianShader、3DGS-DR、Ref-GS、EnvGS，均使用公开代码与配置。
- 论文遵循 3DGS-DR 和 Ref-GS 的 evaluation protocol；所有实验在单张 NVIDIA RTX 4090 上完成。

### 公共场景主实验

| 方法 | 全图 PSNR ↑ | 全图 SSIM ↑ | 全图 LPIPS ↓ | 透明区 PSNR ↑ | 透明区 SSIM ↑ | 透明区 LPIPS ↓ | FPS ↑ | 训练时间 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DGS | 26.493 | 0.816 | 0.181 | 37.673 | 0.990 | 0.012 | **218.95** | **0.3 h** |
| Ref-GS | 26.599 | 0.819 | 0.188 | 37.761 | 0.989 | 0.013 | 38.41 | 0.8 h |
| EnvGS | 27.141 | 0.821 | 0.182 | 37.953 | 0.990 | 0.012 | 18.31 | 2.9 h |
| **RT-Splatting** | **27.490** | **0.831** | **0.167** | **39.765** | **0.992** | **0.010** | 33.28 | 0.9 h |

数据来自正文 Table 1，平均覆盖 6 个公共场景。

- 相比最强全图 PSNR baseline EnvGS，RT-Splatting 提升 **0.349 dB**；透明区域 PSNR 提升 **1.812 dB**。
- 相比 3DGS，透明区域 PSNR 提升 **2.092 dB**，LPIPS 从 0.012 降到 0.010。
- 逐场景透明区域 PSNR 中，RT-Splatting 在 8/8 个公共与自采场景均为第一；Compact 相对第二名提升 3.484 dB，Swab 提升 4.070 dB（补充 Table 4）。
- 33.28 FPS 达到实时，但明显慢于 3DGS/2DGS；它与 Ref-GS 的 38.41 FPS 接近，并快于 EnvGS 的 18.31 FPS。

### 自采场景

| 方法 | 全图 PSNR ↑ | 全图 SSIM ↑ | 全图 LPIPS ↓ | 透明区 PSNR ↑ | 透明区 SSIM ↑ | 透明区 LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3DGS | 27.507 | 0.863 | 0.213 | 32.567 | 0.964 | 0.048 |
| EnvGS | 26.847 | 0.847 | 0.260 | 31.726 | 0.963 | 0.057 |
| **RT-Splatting** | **28.780** | **0.871** | **0.197** | **35.490** | **0.970** | **0.042** |

在 Van 与 Swab 上，RT-Splatting 相对最强 baseline 的全图 PSNR 提升 1.273 dB，透明区域提升 2.923 dB（正文 Table 2）。这组结果支持方法能处理“背景主要通过透明材料可见”的自采配置，但只有两个场景，尚不能代表更广泛手机采集条件。

### 消融实验

| 设置 | 透明区 PSNR ↑ | SSIM ↑ | LPIPS ↓ | 相对完整方法 PSNR |
| --- | ---: | ---: | ---: | ---: |
| w/o occupancy factorization | 36.919 | 0.9885 | 0.0113 | -1.064 dB |
| w/o joint optimization | 36.288 | 0.9876 | 0.0120 | **-1.695 dB** |
| w/o scattering | 37.597 | 0.9897 | 0.0102 | -0.386 dB |
| w/o attenuation | 37.541 | 0.9897 | 0.0102 | -0.442 dB |
| w/o gradient gating | 37.754 | 0.9899 | 0.0101 | -0.229 dB |
| w/o $\mathcal L_{\mathrm{mask}}$ | 37.167 | 0.9894 | 0.0106 | -0.816 dB |
| **完整方法** | **37.983** | **0.9901** | **0.0095** | -- |

正文 Table 3 在 Sedan 与 Truck 的透明区域上取平均。

- **联合优化的影响最大**：分开训练使 Truck 内部无法被重建，因为它只通过车窗可见；这正面验证了方法针对的场景设定。
- **factorization 是核心表示贡献**：去掉 occupancy 后，反射需要高 opacity、透射需要低 opacity 的冲突重新出现，PSNR 降 1.064 dB。
- **mask regularization 不只是辅助项**：去掉后下降 0.816 dB，说明方法对外部透明区域先验有明显依赖。
- **gating 的平均数值收益较温和**：两场景消融为 +0.229 dB。全 8 场景 sensitivity 中，$k=4$ 相对 $k=0$ 只提高 0.122 dB PSNR，但 LPIPS 从 0.0179 改善到 0.0175；其主要价值可能更多体现在局部 floater 抑制和视觉清晰度。
- $k$ 从 2 到 16 的 PSNR 仅在 38.523–38.696 dB 间变化，说明该超参数在测试范围内不算敏感。

### 编辑能力

由于 reflection、transmission、roughness、tint 和散射颜色在表示中被显式拆开，作者展示了：

- 修改车窗或塑料膜的 roughness；
- 调节透明程度；
- 移除 specular reflection；
- 改变材料 tint。

这些结果证明了表示的可控性，但属于定性展示，没有 perceptual study、编辑后 ground truth 或跨渲染器一致性评估。

### 关键发现

1. occupancy–opacity factorization 解决了 deferred surface extraction 与 volumetric transmission 对单一 opacity 的冲突。
2. 在 6 个公共场景中，RT-Splatting 的透明区域 PSNR 比最强 baseline 高 1.812 dB，同时保持 33.28 FPS。
3. 联合训练对于“背景只能透过透明物体观察”的场景尤其关键，独立背景重建并不适用。
4. mask loss 和 factorization 的消融影响大于 gradient gating；gating 更像面向局部伪影的优化稳定器，而不是平均分的主要来源。
5. 方法的有效范围是薄、近似直线透射的半透明表面，不覆盖厚玻璃的显著折射与多次光传输。

## 亮点与洞察

### 论文亮点

- **表示设计抓住了真正的变量冲突**：把“是否存在表面”与“表面是否遮光”分开，比单纯增加更强的 view-dependent color network 更直接。
- **同一组 Gaussians 支持 surface 与 volume 两种解释**：不需要为透明物体和背景维护完全独立的表示，避免前景 mask 抹掉唯一背景观测。
- **gradient gating 实现简洁**：partial stop-gradient 保证 forward image 不变，只调整 credit assignment，容易迁移到其他多分支 inverse-rendering 系统。
- **实验兼顾质量与效率**：同时报告全图、透明区、FPS、训练时间、逐场景数据和关键组件消融。
- **输出具备编辑语义**：反射、透射、roughness 与 tint 可以独立控制，比不可解释的 SH appearance 更适合内容编辑。

### 我的洞察

- **个人分析：论文的核心是 credit assignment，而不只是透明渲染。** 两条分支都能解释同一个像素，真正困难的是图像误差应该由哪条光路负责。factorization 解决表示职责，gradient gating 解决反向传播职责。
- **个人分析：$\sigma$ 更接近“可用于提取表面的 occupancy”，不应直接等同真实物质密度。** 它仍由 image-space reconstruction 和 mask prior 学得，没有几何 ground truth 或真实 hit probability 校准。
- **个人分析：learnable $\beta$ 提升了真实照片拟合能力，也扩大了分解歧义。** 网络可以用 specular color、attenuation、transmissivity 和 scattered color 多种方式解释同一观测，因此漂亮的 reflection/transmission layer 不等于物理参数唯一或准确。
- **个人分析：mask 是方法成功的重要监督预算。** 与不使用透明标注的 baselines 比较时，提升同时来自表示创新和额外语义先验；论文消融已经显示去掉 mask 会明显退化。
- **个人分析：该设计可迁移到其他“几何存在但光学贡献弱”的现象。** 例如薄纱、稀疏植被、反射显示屏或部分遮挡材质，但不同介质可能需要额外折射、偏振或多层 surface model。

## 局限与展望

### 作者承认的局限

- 只针对薄半透明表面，假设折射可以忽略、光线近似直线穿过。
- 没有显式建模 refraction 或 multiple light bounces。
- 因此不适合水体、厚玻璃、实心透明物体等具有明显弯折与内部传播的介质。

### 独立分析

- **额外 mask supervision 影响公平性**：RT-Splatting 使用 SAM2 透明 mask 训练，而主表中的反射类 baselines 未获得等价透明区域监督。更严格的实验应增加“所有方法共享 mask”或“RT-Splatting 完全无 mask”设置。
- **最直接的透明重建 baselines 缺席**：论文讨论 TransparentGS、TSGS 等方法，却没有在主表中比较。它们的捕获假设可能不同，但至少应在适用子集上给出定量或清楚说明不可比较原因。
- **评测只有 8 个真实场景**：且场景主要是汽车玻璃和塑料膜。复杂曲面、不同透明度、强近场反射和大尺度室内玻璃仍缺少系统覆盖。
- **缺少真实分解 ground truth**：PSNR/SSIM/LPIPS 衡量最终图像，不直接验证 reflection layer、transmission layer、normal、roughness、$\tau$ 或 $\beta$ 的物理正确性。
- **合成公式不保证能量守恒**：$\mathbf C_{\mathrm{spec}}+\beta\mathbf C_{\mathrm{sub}}$ 可能通过网络吸收曝光、tone mapping 与错误几何，适合 image fitting，但不能直接用于需要 calibrated material 的 inverse rendering。
- **gating 依据预测反射计算**：早期训练中 $\mathbf C_{\mathrm{spec}}$ 本身不可靠，可能错误抑制本应进入背景的监督。论文没有报告 gating schedule、训练早期稳定性或预测 variance 的校准。
- **平均指标可能低估局部价值**：透明区 PSNR 很高，可能包含大面积平滑背景；建议增加 edge PSNR、reflection/transmission boundary error、floater depth error 与 temporal consistency。

### 建议的后续实验

1. 构建带 path-traced reflection、transmission、depth、normal 和 material GT 的薄透明 synthetic benchmark，直接验证分解质量。
2. 在相同 mask supervision 下比较 RT-Splatting、TransparentGS、TSGS 和反射类 deferred GS，拆分表示与监督带来的收益。
3. 将单 first-hit surface 扩展为 multi-layer G-buffer，处理多层玻璃、双面薄片和透明物体重叠。
4. 让 gradient gate 同时考虑训练阶段、uncertainty 与跨视角一致性，而不只依赖单帧局部 specular variance。
5. 加入 Snell refraction 与 learned thickness，并报告质量—速度曲线，检验方法能否从薄膜扩展到厚介质。

## 与相关工作的对比

| 方法 | 表示与渲染机制 | 透明背景处理 | 主要限制 / 特点 |
| --- | --- | --- | --- |
| 3DGS / 2DGS | 单 opacity alpha blending；SH appearance | 原生 blending 可模拟透明，但几何存在与透光性耦合 | 快速，难同时得到锐利反射与清晰透射 |
| 3DGS-DR / Ref-GS | first-surface G-buffer + deferred reflection | 最近表面属性会压制后方 transmission | 高频反射强，透明处理不是核心目标 |
| EnvGS | environment Gaussians + ray tracing | 能表达近场 reflection，但仍偏反射建模 | 公共场景中最强 baseline，速度 18.31 FPS |
| TransparentGS | 透明 Gaussian、deferred refraction、背景预重建 | 通常先重建并冻结背景 | 可处理折射，但背景只透过透明面可见时受限 |
| **RT-Splatting** | **$\sigma/\alpha$ factorization + deferred surface / forward volume** | **同一表示联合恢复透明表面与其后背景** | **实时且结果强，但依赖 mask，不建模显著折射** |

## 启发与关联

- **对 Gaussian surface reconstruction**：opacity 不必同时承担 geometry confidence 与 radiometric attenuation；把两种语义分离后，pruning、surface extraction 和 rendering 可以采用不同变量。
- **对多分支 inverse rendering**：如果多个分支能解释同一 residual，应显式设计 gradient routing，而不只依靠最终 reconstruction loss 自动分工。
- **对透明资产编辑**：当前分解适合 view synthesis 与视觉编辑；若目标是导出 PBR 资产，还需要 calibrated IOR、thickness、absorption coefficient 和跨 renderer 验证。
- **假设**：把 occupancy–opacity factorization 与显式 mesh/UV carrier 结合，可以先稳定恢复玻璃表面，再在 UV 空间学习 transmissivity 与 tint，同时用体表示保留表面后的背景。

## 评分

| 维度 | 评分（10 分） | 理由 |
| --- | ---: | --- |
| 创新性 | 8.7 | 用 occupancy–opacity 解耦统一 surface 与 volume，并配合 gradient routing，问题与机制对应清楚 |
| 技术可靠性 | 8.1 | 公式、实现和消融完整，但合成非严格物理、分解无 GT，且依赖 SAM2 mask |
| 实验充分度 | 7.8 | 有 8 个真实场景、逐区域指标和消融；缺透明专用 baseline、公平 mask 对照与更广材质覆盖 |
| 写作清晰度 | 8.8 | 从 opacity 冲突到混合 renderer、再到梯度歧义的论证链条简洁明确 |
| 实用 / 研究价值 | 8.8 | 33 FPS、代码公开且支持独立编辑 reflection/transmission，对透明场景 NVS 很有价值 |

**总体推荐：值得细读。** 它提供了一个很干净的建模原则：当一个 radiometric 参数被迫同时表达几何与光学语义时，应先拆分变量，再为不同渲染路径设计明确的梯度职责。

## 阅读结论

- **最值得记住的点**：高 occupancy、低 opacity 让同一 Gaussian 既能成为稳定反射表面，又不会遮住透射背景。
- **最需要怀疑的点**：最终 reflection/transmission decomposition 缺少物理 GT，且方法比主 baselines 使用了额外透明 mask supervision。
- **最值得复现或继续验证的点**：在统一 mask 条件下复现 factorization、joint optimization 与 gating，并用带光路分解 GT 的数据验证各分支是否真的学到正确成分。

## 相关论文

- [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/) — RT-Splatting 的 surface-aligned Gaussian 基础。
- [Ref-GS: Directional Factorization for 2D Gaussian Splatting](https://ref-gs.github.io/) — 本文 specular shading network 的直接基础与主要 baseline。
- [EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](https://zju3dv.github.io/envgs/) — 近场反射建模与公共场景中最强对照方法。
- [TransparentGS: Fast Inverse Rendering of Transparent Objects with Gaussians](https://doi.org/10.1145/3730892) — 显式透明物体与折射建模，但采用不同的背景处理路线。
- [NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections](https://dorverbin.github.io/nerf-casting/) — 通过反射路径 tracing 建模近场一致反射，质量强但不以实时 GS 渲染为目标。
