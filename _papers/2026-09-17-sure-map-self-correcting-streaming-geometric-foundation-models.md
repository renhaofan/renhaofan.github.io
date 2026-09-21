---
title: "SURE-Map: Self-Correcting Streaming Geometric Foundation Models"
subtitle: "以跨视图几何不确定性修局部平移与点云，以稀疏 keyframe window 周期校正长期尺度漂移"
authors: "Mingkai Liu, Hao Zhao, Xingxing Zuo"
venue: "arXiv preprint"
year: 2026
date: 2026-09-17
paper_url: "https://arxiv.org/abs/2609.15795"
code_url: "https://github.com/RCL-Robotics/SURE-map"
project_url: "https://mingkai-liu.github.io/projects/sure-map/"
tags:
  - Streaming 3D Reconstruction
  - Geometric Foundation Models
  - Online Mapping
  - Uncertainty Estimation
  - Scale Drift
summary: "SURE-Map 为冻结的 LingBot-Map 流式几何 backbone 增加两层 self-correction：以 pose-depth 诱导的跨帧 flow 残差训练 per-pixel cross-view geometric uncertainty，用它过滤坏点并做仅平移的局部 Gauss–Newton；再以稀疏 keyframe window 的 full-attention 推理周期估计尺度并融合近期相对平移。无 loop closure 时，它在 KITTI、Oxford Spires、VBR 的 Sim(3)-aligned ATE-RMSE 分别为 17.24、4.74、28.58 m，优于 LingBot-Map 的 24.00、5.11、31.37 m。"
permalink: /papers/sure-map-self-correcting-streaming-geometric-foundation-models/
---

> **阅读依据**：arXiv:2609.15795v1（2026-09-14，9 页）及官方仓库实现。仓库将其称为 *Self-Correcting Streaming Geometric Foundation Models*，论文中 SURE 指 **Scale- and Uncertainty-aware REconstruction**。以下“无 LC”指不使用 loop closure / pose-graph optimization；ATE 均在标准 Sim(3) trajectory alignment 后报告，因此不等同于未对齐的绝对尺度定位误差。

## 一句话总结

SURE-Map 不替换流式几何 foundation model，而是让它能自我诊断并分层纠错：短时间尺度上学习“pose 与 depth 联合起来是否给出了可信跨视图对应”，据此删坏点、只修正相邻帧平移；长时间尺度上用稀疏 keyframe 的 full-attention 推理重新估尺度，避免 bounded-context streaming 的局部误差积累为整段轨迹 scale drift。

## 背景与核心问题

VGGT、LingBot-Map、HorizonStream 一类 feed-forward streaming 模型直接从图像流预测 pose、depth 与 dense geometry，省去传统 SLAM 的特征匹配、BA 和重型后端。但严格因果、有限 memory 的代价是：动态物体、弱纹理、重复结构和几何退化会使当前帧的隐式 data association 出错；局部 pose/depth 错误既污染点云，又会在长轨迹中逐段累积为尺度漂移。

作者的切入点是区分两个经常混在一起的可靠性问题：

- 单视图 depth/point confidence 说的是“这一像素本身的几何预测像不像”；它未必表示相邻帧之间是否能对上。
- 长程 drift 不是把每一对相邻帧局部对齐好就会消失；需要偶尔引入更长 temporal context。

因此问题变为：**能否在不放弃流式低延迟的条件下，学习跨视图一致性，并以稀疏、较昂贵的长窗口证据校正长期尺度？**

## 方法详解

### 整体框架

<div class="mermaid">
flowchart LR
    A[因果 RGB 流] --> B[冻结 LingBot-Map：geometry-context attention]
    B --> C[每帧 pose、depth、intrinsics、geometry tokens]
    C --> D[跨视图 uncertainty head]
    D --> E[高 uncertainty 点过滤]
    D --> F[加权局部 Gauss-Newton 平移修正]
    C --> G[稀疏 keyframe 选择]
    G --> H[full-attention keyframe window 推理]
    H --> I[尺度估计与近期轨迹重标定]
    F --> J[多时间尺度平移融合]
    I --> J
    J --> K[流式点云地图与轨迹]
</div>

正常帧只走快速的 causal backbone；keyframe window 异步/周期性运行。训练时只训练 uncertainty head，backbone 保持 frozen。可选的 loop closure 是附加 PGO 后处理，并非 SURE-Map 核心的 online self-correction。

### 1. 跨视图几何不确定性：评估 pose-depth 共同诱导的对应

流式 backbone 在时刻 $i$ 接收当前图像和紧凑历史 memory：

$$
f_\theta(I_i,M_{i-1})\rightarrow(T_i,D_i,K_i,M_i),
$$

输出相机 pose $T_i=[R_i\mid t_i]$、depth $D_i$ 与 intrinsics $K_i$。对当前像素 $u$，以当前 depth back-project、用预测相对 pose 变换到上一相机并投影，即可得到不是网络显式预测、而是由 pose+depth **解析诱导**的 backward flow：

$$
p_{\mathrm{cur}}=D_i(u)K_i^{-1}\bar u,
$$

$$
p_{\mathrm{prev}}=R_{i-1}^{\top}(R_ip_{\mathrm{cur}}+t_i-t_{i-1}),
\qquad
f_{i\to i-1}(u)=\pi(K_{i-1}p_{\mathrm{prev}})-u.
$$

uncertainty head 从相邻帧 geometric tokens $(G_{i-1},G_i)$ 输出两个 log-variance map：

$$
S_i=[S_i^u;S_i^v],\qquad
\Sigma_i(u)=\operatorname{diag}(e^{S_i^u(u)},e^{S_i^v(u)}).
$$

监督不是 depth error，而是 induced flow 与 ground-truth flow 的残差 $\epsilon_i=f_{i\to i-1}-f^{\mathrm{gt}}_{i\to i-1}$。在有效 flow 像素上最小化异方差高斯 NLL：

$$
\ell_i(u)=\epsilon_i(u)^\top\Sigma_i(u)^{-1}\epsilon_i(u)
+\log\det\Sigma_i(u).
$$

第一项要求大残差区域提高 uncertainty，第二项阻止把所有 uncertainty 无限放大。故它学习的是“当前 pose 与 depth 是否一起支持一个可信 cross-view correspondence”，比 backbone 的单视图 depth confidence 更贴近点云/位姿后端真正需要的信号。

从代码看，head 使用当前帧深层 patch tokens 为 Query、上一帧 tokens 为 Key/Value 做 cross-attention，再经 DPT decoder 输出 $\log\sigma_u^2,\log\sigma_v^2$。训练使用 TartanAir 的 GT flow、8–24 帧 clips、$518\times392$、20k steps；LingBot-Map backbone 冻结。

### 2. 同一 uncertainty 服务两个局部纠错任务

用于点云时，作者把二维 covariance 合成标量：

$$
\sigma_i^{\mathrm{geo}}(u)=\sqrt{\operatorname{tr}(\Sigma_i(u))}
=\sqrt{e^{S_i^u(u)}+e^{S_i^v(u)}}.
$$

只 back-project 有效 depth 且 $\sigma_i^{\mathrm{geo}}$ 低的像素。它并非“越少点越准”的 trivial filter：实验在固定 removal ratio 下比较，说明 native depth confidence 往往删掉可用结构，因其主要反映 range-dependent 的单帧可靠性。

用于位姿时，作者对 current-to-previous 的 translation 做 point-to-plane 对齐。令 $q$ 是上一帧对应像素反投影的表面点、$n$ 为其法线，则：

$$
r_i(u;t)=n^\top(p_{\mathrm{prev}}(u;t)-q).
$$

将像素二维 covariance 经 residual 对 pixel coordinate 的一阶 Jacobian $J_i^r$ 传播：

$$
\sigma_{r,i}^2(u)=J_i^r(u)\Sigma_i(u)J_i^r(u)^\top+\sigma_0^2,
$$

再求加权的 Gauss–Newton 目标：

$$
t_i^{\mathrm{GN}}=\arg\min_t
\sum_{u\in\Omega_i^g}\frac{r_i(u;t)^2}{\sigma_{r,i}^2(u)}.
$$

关键约束是 **rotation 固定，只更新 translation**。作者担心由 scale/correspondence 残差驱动的 rotation error 会在长序列中乘法积累；仓库实现也确实只解一个 $3\times3$ translation Hessian，默认 3 次 GN iteration。

### 3. 多时间尺度尺度重标定：快路径不变，慢路径定期给尺度证据

快速路径对连续帧使用 geometry-context-attention（LingBot-Map 的流式机制）。慢路径维护固定大小 keyframe window：按固定 stride 插入，若相对最近 keyframe 的 pose-depth-induced flow 足够大也插入；周期性以 full-attention 联合处理窗口内帧，得到较长上下文下的 $(D_c^k,T_c^k)$。

先把 full-attention depth 与 streaming depth 在 inverse-depth 空间拟合：

$$
\eta=\arg\min_{\eta>0}
\sum_{c\in K}\sum_{u\in\Omega_c}
\left\lvert D_c(u)^{-1}-\eta(D_c^k(u))^{-1}\right\rvert^2.
$$

接着从窗口中成对 keyframe 的相对 translation length，取尺度比的 median：

$$
\phi=\operatorname{median}_{(m,n)\in\mathcal P}
\frac{\left\lVert\operatorname{trans}((T_m^k)^{-1}T_n^k)\right\rVert/\eta}
{\left\lVert\operatorname{trans}(T_m^{-1}T_n)\right\rVert}.
$$

对最近 segment 的相对平移先得 $t_i^{\mathrm{scale}}=\phi\operatorname{trans}(T_{i-1}^{-1}T_i)$，再和局部 GN 结果做二次融合：

$$
t_i^\star=
\arg\min_t\left[(1-\mu)\lVert t-t_i^{\mathrm{scale}}\rVert_2^2
+\mu\lVert t-t_i^{\mathrm{GN}}\rVert_2^2\right].
$$

可理解为：长窗口负责“这段轨迹总体走多远”，相邻帧优化负责“这一小步怎么对齐”。仓库实现中，inverse-depth scale 用 median 初始化、Huber reweighting 做鲁棒拟合；默认 KITTI 配置为 20 keyframe window、每 10 个 keyframe 触发、scale damping 0.8。

## 实验证据

### 设置

长程轨迹评测为 KITTI、Oxford Spires、VBR，以 Sim(3)-aligned ATE-RMSE（m，低更好）报告；dense reconstruction 使用 Neural RGB-D、7-Scenes，报告 Accuracy、Completeness、Chamfer Distance（m，低更好）和 F1（0.05 m threshold，高更好）。比较对象包含 streaming feed-forward 的 LingBot-Map、HorizonStream、LongStream、CUT3R 等，以及优化型和 offline 方法。

### 主结果：无 loop closure 的长程轨迹

| 方法 | KITTI ATE ↓ | Oxford ATE ↓ | VBR ATE ↓ |
| --- | ---: | ---: | ---: |
| CUT3R | 91.62 | 32.47 | 66.25 |
| TTT3R | 72.86 | 25.05 | 64.99 |
| LongStream | 51.90 | 15.92 | 77.93 |
| LingBot-Map | 24.00 | 5.11 | 31.37 |
| HorizonStream | 22.38 | 8.87 | 29.68 |
| **SURE-Map** | **17.24** | **4.74** | **28.58** |
| SURE-Map + LC | 15.17 | 4.63 | 22.12 |

相对其 backbone LingBot-Map，无 LC 时降低 28.2%（KITTI）、7.2%（Oxford）、8.9%（VBR）的 ATE。LC 进一步提高全局一致性，但它使用 DINOv2 retrieval 后加入 pose-graph optimization，已超出纯 feed-forward streaming 的核心设定，应单列比较。

### Dense geometry 与 uncertainty filtering

| 方法 | Neural RGB-D：Acc. / Comp. / F1 | 7-Scenes：Acc. / Comp. / F1 |
| --- | --- | --- |
| LingBot-Map | 0.074 / **0.030** / 65.10 | 0.035 / **0.043** / 81.77 |
| **SURE-Map** | **0.067** / 0.035 / **66.20** | **0.033** / 0.045 / **81.93** |

SURE-Map 提高 accuracy 与 F1，但 completeness 略降，这是过滤高 uncertainty 点的正常 precision–coverage 取舍。关键消融更能说明 signal 是否对：

| 数据集 | 变体 | CD ↓ | Comp. ↓ | F1 ↑ |
| --- | --- | ---: | ---: | ---: |
| Neural RGB-D | 完整 SURE-Map | **0.051** | 0.035 | **66.20** |
|  | 不做 uncertainty filter | 0.052 | **0.030** | 65.10 |
|  | 改用 depth-confidence filter | 0.084 | 0.113 | 65.00 |
| 7-Scenes | 完整 SURE-Map | **0.039** | 0.045 | **81.93** |
|  | 不做 uncertainty filter | **0.039** | **0.043** | 81.77 |
|  | 改用 depth-confidence filter | 0.051 | 0.079 | 80.90 |

这表明模型的贡献不是任意剔除点，而在于 cross-view uncertainty 比 depth confidence 更契合“该点是否破坏跨视图 map”的判断；但收益在相对干净的 7-Scenes 较小。

### 消融与运行代价

| 移除项 | Oxford ATE ↓ | KITTI ATE ↓ | VBR ATE ↓ |
| --- | ---: | ---: | ---: |
| 完整 + LC | 4.63 | 15.17 | 22.12 |
| 去 LC（完整 online SURE-Map） | 4.74 | 17.24 | 28.58 |
| 再去 uncertainty-weighted optimization | 4.91 | 17.57 | 29.96 |
| 再去 scale recalibration（即 LingBot-Map） | 5.11 | 24.00 | 31.37 |

scale recalibration 是主要来源，尤其 KITTI 从 24.00 到 17.24；uncertainty-weighted optimization 则提供互补的局部修正。运行上，SURE-Map 比 LingBot-Map 每帧多 15–30 ms；KITTI sequence 04 从 11.68 降到 9.17 FPS，仍维持流式处理，但并非零成本。

## 亮点与局限

**论文亮点**：

- 把 uncertainty 从“本帧深度像不像”提升为“pose-depth 联合推导的 correspondence 是否自洽”，监督目标与后端几何用途一致。
- 同一个 uncertainty 同时服务 point filtering 与 residual weighting，信号—决策链完整，而非仅画一张 confidence map。
- 局部 translation-only 与周期 scale calibration 分工明确，避免用单一优化器同时应对短时错配和长时 drift。
- 只训练一个轻量 head，backbone 冻结；工程上容易嫁接到同类 streaming geometry model。

**作者未解决/公开指出的边界**：局部有限 context 仍会受 dynamic/weak texture 影响；SURE-Map 通过 sparse full-attention 缓解而非彻底消除长期错误。论文同时提供可选 LC，说明最强全局一致性仍受传统 retrieval/PGO 助力。

**独立分析**：

- uncertainty head 训练依赖 TartanAir 的 GT optical flow；从合成/仿真动态到真实 driving、室内反光或 rolling shutter 的 calibration 可能不足。应按场景类型报告 uncertainty calibration curve、risk-coverage 与 OOD failure。
- flow residual 同时混合 depth、intrinsics、rotation 和 translation 的误差，但优化阶段只修 translation。固定 rotation 能防止长期放大，却会留下 rotation-dominated 失败情形；可用 Hessian conditioning 或保守 rotation prior 做选择性 6-DoF 更新。
- ATE 在 Sim(3) 后对齐，适合比较轨迹形状/drift，却掩盖全局绝对 scale 与坐标系误差。应补 SE(3)-aligned ATE、relative pose error 和真实部署的无 GT alignment 结果。
- full-attention keyframe 路径、Flow head 与 15–30 ms/frame 开销在资源受限机器人上并不轻；论文未详报 peak VRAM、keyframe-job latency 或最坏延迟。

## 与相关工作对比

| 方法 | 流式 / 因果 | 主要长程机制 | 局部可信度 | 与 SURE-Map 的差别 |
| --- | --- | --- | --- | --- |
| VGGT | 否 | 全序列多视图 attention | 点/深度 confidence | 准确但不适合 strict online |
| LingBot-Map | 是 | compact geometry context | 原生 depth confidence | SURE-Map 的冻结 backbone |
| HorizonStream | 是 | long-horizon attention / 几何证据传播 | 非 pose-depth correspondence uncertainty | 仍以 model-internal context 为主 |
| SLAM hybrid / PGO | 可在线 | BA、pose graph、LC | 显式 correspondence 或 learned prior | 全局强但优化重 |
| **SURE-Map** | **是** | sparse full-attention scale recalibration | **cross-view flow-residual uncertainty** | 轻后端纠错，LC 可选而非必需 |

## 评分

| 维度 | /10 | 依据 |
| --- | ---: | --- |
| 创新性 | 8.3 | 将 pose-depth induced correspondence uncertainty 明确接入 streaming 后端，并以双时间尺度校正。 |
| 技术可靠性 | 8.0 | 监督、uncertainty propagation、translation-only GN 与消融链条完整。 |
| 实验充分度 | 8.0 | 三个长程数据集、两个 dense benchmark、filter/scale/LC 消融齐全；缺 uncertainty calibration 与资源峰值分析。 |
| 写作清晰度 | 8.4 | 明确区分 core online correction 与 optional LC，公式到实现对应清楚。 |
| 实用/研究价值 | 8.5 | 对已有 streaming geometry backbone 是低侵入插件，适合机器人/AR 长序列。 |

**总体推荐：值得细读。**特别适合关注“foundation model 是否能成为地图系统”的读者：它说明光靠更长 cache 不够，模型还需要判断跨帧几何何时不可信，并有受控机制改正错误。

## 阅读结论

- **最值得记住的点**：有价值的几何 confidence 应衡量跨视图 correspondence 是否自洽，而不仅是单帧 depth 是否像真。
- **最需要怀疑的点**：Sim(3)-aligned ATE 与 TartanAir-flow 训练都可能高估真实部署中的绝对尺度和跨域可靠性。
- **最值得复现或继续验证的点**：在真实动态/OOD 数据上画 uncertainty risk-coverage 曲线，并比较固定 rotation 与条件式 6-DoF 局部优化。

## 相关论文

- [LingBot-Map / Geometric Context Transformer](https://arxiv.org/abs/2604.14141) — SURE-Map 的冻结流式 backbone。
- [HorizonStream](https://arxiv.org/abs/2605.23889) — 长程 streaming 几何证据传播的直接对照。
- [VGGT](https://arxiv.org/abs/2503.11651) — 多视图 geometry foundation model 的基础。
- [MASt3R-SLAM](https://arxiv.org/abs/2412.12392) — 使用 3D reconstruction prior 的优化型 dense SLAM 对照。
- [DROID-SLAM](https://arxiv.org/abs/2108.10869) — 学习式视觉 SLAM / 局部几何优化的经典参照。
