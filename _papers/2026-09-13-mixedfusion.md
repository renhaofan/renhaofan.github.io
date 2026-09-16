---
title: "MixedFusion: Real-Time Reconstruction of an Indoor Scene with Dynamic Objects"
subtitle: "用 Sigmoid-ICP 解耦相机与物体运动，并以混合体素分配同时融合静态场景和非刚性对象"
authors: "Hao Zhang, Feng Xu"
venue: "IEEE Transactions on Visualization and Computer Graphics (TVCG), 24(12)"
year: 2018
date: 2026-09-13
paper_url: "http://xufeng.site/publications/2018/MixedFusion-tvcg.pdf"
code_url: ""
project_url: ""
tags:
  - RGB-D Reconstruction
  - Dynamic Scene Reconstruction
  - TSDF Fusion
  - Non-rigid Tracking
  - Robust ICP
  - Voxel Hashing
summary: "MixedFusion 面向含移动人物、飘动布料等动态对象的室内 RGB-D 扫描，用饱和 Sigmoid 损失把 ICP 改写为近似最小化不匹配 correspondence 数量，在迭代相机配准的同时产生动静残差；随后以 deformation-node 连通分量把标签传播到不可见表面，只对动态组件运行 DynamicFusion 式非刚性跟踪，并分别采用模型驱动的动态体素扩张与深度驱动的静态体素分配更新统一 canonical TSDF。系统在 GTX TITAN X 上处理动态场景约 41 ms/帧、平均 25 FPS，但评测主要是定性图和单序列 pose curve，且无法处理拓扑变化、静转动状态迁移、动静大面积接触、颜色与回环。"
permalink: /papers/mixedfusion/
---

> **阅读依据**：本笔记基于 Feng Xu 作者主页公开的 [MixedFusion 官方 PDF](http://xufeng.site/publications/2018/MixedFusion-tvcg.pdf) 全文（10 页，Eqs. 1–15）、[IEEE Xplore 记录](https://ieeexplore.ieee.org/document/8241434/)及[官方演示视频](https://www.youtube.com/watch?v=3_iuHty3OQ4)。正式信息为 **IEEE TVCG 24(12): 3137–3146, 2018**，DOI：[10.1109/TVCG.2017.2786233](https://doi.org/10.1109/TVCG.2017.2786233)；在线发表日期为 2017-12-28。
>
> **资源状态**：作者主页只提供论文和视频，未提供项目页、代码或数据；针对标题与作者的 GitHub 检索也未找到官方实现。因此本文可以核对公式、算法和论文数值，但不能检查 S-ICP 线性化、阈值、GPU kernel 或重现实验。论文虽使用 RGB-D sensor，当前算法只使用 depth geometry；colour image 仅用于展示。

## 一句话总结

MixedFusion 把“静态 SLAM”和“单物体 DynamicFusion”接成一个实时混合系统：它先用 bounded sigmoid robust loss 让 ICP 的相机位姿主要由占多数的静态 correspondence 决定，同时把高残差视为动态线索；再以 deformation-node graph 的 connected components 将动态标签扩展到当前不可见表面，只对该区域估计非刚性 warping；最后用**动态部分沿已有 canonical model 局部生长、静态部分沿去除动态像素后的 depth rays 分配**的 mixed voxel allocation 更新稀疏 TSDF。这个分工在 2018 年首次较系统地同时保留房间和移动对象，并达到约 25 FPS，但“实时”建立在动态区域较小、固定拓扑、连续运动和无回环的强假设上。

## 背景与动机

- **传统 indoor RGB-D reconstruction 假设世界静止**：KinectFusion、InfiniTAM、Voxel Hashing、BundleFusion 只需估计 6-DoF camera motion，把每帧 depth 变换到统一坐标后累积 TSDF。
- **动态对象会同时破坏 tracking 与 fusion**：ICP 会调整 camera pose 去解释局部 object motion；错 pose 又会让静态墙面/家具重影。即使 camera pose 正确，直接把动态 depth 反投影到 canonical frame 也会留下多个位置的 ghost geometry。
- **DynamicFusion 的范围较小**：它把整个 view 中的 surface 当作一个 non-rigid object，用 deformation graph 对齐并融合，适合单个近距离动态目标；若套到整间房间，motion variables 和计算量都过大，新增远处静态 geometry 也难以正确分配。
- **简单删除 moving people 不总合理**：人物可以移除，但飘动 curtain、被移动的室内物品仍属于用户想保留的 scene content；更理想的是分别融合 static/dynamic geometry，并记录 object motion。
- **camera estimation 与 motion segmentation 是鸡生蛋问题**：知道 camera motion 后，可按 residual 找 dynamics；知道 static mask 后，又可只用静态点估 camera。MixedFusion 用一个 sigmoid robust objective 在迭代中同时逼近两者。

**核心研究问题：能否在单深度相机在线扫描中，实时解耦全局相机运动与局部刚性/非刚性物体运动，并让静态大场景与动态对象在同一稀疏 TSDF 中以不同规则正确扩张和融合？**

## 核心问题

1. **相机 motion 被局部 motion 污染**：普通 ICP 即使有 distance/normal rejection，许多移动对象 correspondence 的 residual 仍低于 hard threshold，会把 pose 拉偏。
2. **动静 segmentation 只能直接观察正面**：当前 depth 看不到人物背面；若只标可见 pixels，完整 dynamic canonical model 无法一起 warp。
3. **非刚性优化不能覆盖整间房间**：必须把昂贵 deformation-graph fitting 限制到动态组件，同时保证 static background 仍能用刚体 camera tracking。
4. **两类 geometry 需要不同 voxel allocation**：static surface 应沿当前 depth ray 新建 blocks；dynamic depth 已相对 canonical space 运动，若同样反投影会在错误位置开辟 blocks。
5. **状态会随时间变化**：物体可从 static 变 dynamic 或再次静止；若 canonical TSDF 没有 object ownership/去融合机制，旧 geometry 会残留。

## 方法详解

### 整体框架

<div class="mermaid">
flowchart TB
    A[当前 depth frame] --> B[Bilateral filtering]
    B --> C[depth vertices + normals]
    D[上一帧 live model] --> E[S-ICP]
    C --> E
    E --> F[global camera increment]
    E --> G[sigmoid fitting residual]

    G --> H[FOV deformation nodes]
    H --> I[node graph connected components]
    I --> J[按高残差 node 比例分类]
    J --> K[static model]
    J --> L[dynamic model 含不可见连通表面]

    L --> M[DynamicFusion-style non-rigid ICP]
    M --> N[node SE3 + dual quaternion warp]
    N --> O[动态模型驱动 voxel block 扩张]
    O --> P[warp voxels to live frame]
    C --> P
    P --> Q[更新 dynamic TSDF]

    Q --> R[将 live dynamic model 投影到 depth]
    R --> S[遮掉 dynamic pixels]
    C --> S
    S --> T[静态 depth-ray voxel allocation]
    T --> U[更新 static TSDF]
    F --> U

    Q --> V[warp back to canonical model]
    U --> V
    V --> W[下一帧 canonical/live model]
</div>

系统维护：

- **canonical model**：第一帧 camera coordinate 下的统一 TSDF，是长期融合结果；
- **live model**：把 canonical model 经 dynamic warp $W$ 和 global camera pose $T_g$ 变到当前 camera frame 后的 surface；
- **static/dynamic model**：每帧根据 S-ICP residual 与 node connectivity 从当前 model 临时分出，使用不同 motion/fusion 路径。

执行顺序必须是 camera tracking → segmentation → dynamic motion → dynamic fusion → dynamic depth masking → static fusion；若先做 static depth allocation，moving object 会污染 canonical volume。

### 1. TSDF 与 canonical/live model

场景由 voxel grid 上的 TSDF

$$
S(\mathbf x)=\{v(\mathbf x),w(\mathbf x)\}
$$

表示，$v$ 是 signed distance，$w$ 是 confidence。zero level set 给出 surface mesh（Eq. 1）：

$$
\mathcal M=\{(\mathbf v,\mathbf n)\mid
S_v(\mathbf v)=0,\ 
\mathbf n=\nabla S_v(\mathbf v)/\|\nabla S_v(\mathbf v)\|_2\}.
$$

对 canonical vertex $\mathbf v$，dynamic warping $W$ 先描述 object-local deformation，全局 rigid pose $T_g=[R_g,t_g]$ 再描述 camera coordinate change。live model（Eq. 2）近似为：

$$
\mathbf v_W=T_gW(\mathbf v),
\qquad
\mathbf n_W=T_gW(\mathbf n).
$$

严格说 normal 不应应用 translation，论文以 homogeneous notation 简写；实现通常只对 normal 用 rotational part。

### 2. S-ICP：从最小 residual 转向最少不匹配点

#### 普通 ICP 为什么失败

传统 point-to-plane ICP 最小化所有 correspondence residual；moving person 的局部 motion 会贡献系统性误差，optimizer 可能移动 camera pose 去同时折中 static room 和 person。position/normal hard filtering 只能去掉大离群点，小幅但一致的 object motion 仍会通过。

MixedFusion 的观察是：室内扫描时 dynamic part 通常只占 scene 的小部分。正确 camera pose 应使**多数 static points 拟合良好**，而允许少数 dynamic points 不拟合。因此目标改为近似统计“不拟合 correspondence 数量”。

#### 0–1 objective

当前 depth vertex/normal 为 $V_d^t(\mathbf u),N_d^t(\mathbf u)$，上一帧 live model 为 $V_W^{t-1},N_W^{t-1}$。对 camera increment $\Delta T_g^t$，point-to-plane residual（Eq. 5）为：

$$
\operatorname{Dist}(\mathbf u;\Delta T_g^t)
=
\left|
\left(\Delta T_g^t\bar V_W^{t-1}(\mathbf u)-V_d^t(\mathbf u)\right)^\top
N_W^{t-1}(\mathbf u)
\right|.
$$

理想 objective（Eqs. 3–4）为：

$$
E(\Delta T_g^t)=\sum_{\mathbf u\in\mathcal C}
F(\operatorname{Dist}),
\qquad
F(d)=\begin{cases}
0,&d<\mathrm{thr},\\
1,&d\ge\mathrm{thr}.
\end{cases}
$$

它不是让少数大 outliers 主导 squared error，而是让 optimizer 找到使最多 correspondences 落入 threshold 的 pose，近似 maximum consensus。

#### Sigmoid relaxation

0–1 step 不可导，作者以可微 logistic（Eq. 6）近似：

$$
F(d)=\frac{1}{1+\exp[-k(d^2-\mathrm{thr}^2)]}.
$$

- $d\ll\mathrm{thr}$：$F\approx0$，视为 static inlier；
- $d\gg\mathrm{thr}$：$F\approx1$ 且逐渐饱和，dynamic/outlier 不再按 residual magnitude 无限拉动 pose；
- $k$ 控制从 0 到 1 的 slope；越大越像 hard counting，但 optimization landscape 也更尖锐；
- `thr` 同时是 robust tracking cutoff 和后续 dynamic segmentation 阈值。

每次迭代线性化求 $\Delta T_g$、更新 live model correspondence 并再求解，最后累积 increments。canonical-to-live pose 更新（Eq. 7）：

$$
T_g^t=
[\Delta R_g^tR_g^{t-1},\ 
\Delta R_g^tt_g^{t-1}+\Delta t_g^t].
$$

**一式两用**：converged pose 下 $F\approx0$ 的 correspondence 是 static，$F\approx1$ 是 dynamic。S-ICP 因此把 robust camera estimation 与 visible-region segmentation 合并。

**个人分析**：这依赖“static correspondence 数量占多数”这一隐含 breakdown-point 假设。若人贴近 camera 占据大部分 FOV、整个 curtain 大幅运动，minimum-consensus 可能把 dynamic object 当成 global frame，反而将背景判为 outlier。

### 3. Node connectivity：把动态标签传播到不可见表面

S-ICP residual 只覆盖当前可见 pixels，但 canonical model 已融合了背面。作者借用 DynamicFusion deformation graph：

1. 在当前 FOV 内提取均匀分布的 nodes $\mathcal N^t_{FOV}$，包括 view frustum 中被遮挡的 model nodes；
2. 按 graph connectivity 分成 connected classes $\mathcal N^t_{p_1},\dots,\mathcal N^t_{p_m}$；
3. 每个 visible node 的 fitting error 是其控制 visible surface 的平均 residual；
4. 对每个 component 统计高残差 nodes 数量，超过阈值则整类标 dynamic；
5. 该 component 中 invisible nodes 也一并进入 dynamic node set $\mathcal N_D^t$；受它们控制的 surface 是 dynamic model，其余是 static model。

这样人物正面发生运动时，背面虽没被当前 depth 看到，也会因与正面 graph-connected 而一起 warp，而不是留在 static TSDF 中。

代价是 segmentation 粒度受 connectivity 支配：人手接触桌面、球贴着桌面、布料与墙面大面积连接时，dynamic/static components 可能粘在一起。作者在 limitations 中明确承认无法处理 dynamic object 与 static scene 的 large connections，并建议加入 semantics。

### 4. 动态部分：DynamicFusion 式 deformation graph fitting

每个 dynamic node $V_{D,i}$ 有 local rigid transform $W(V_{D,i})$，mesh vertex warp 由邻近 node transforms 的 dual-quaternion blending（DQB）获得（Eq. 8）：

$$
W(V)=\operatorname{SE3}(\operatorname{DQB}(V)).
$$

优化（Eq. 9）：

$$
E(W)=E_{\mathrm{depth}}(W)+v_{\mathrm{smooth}}E_{\mathrm{smooth}}(W).
$$

Data term（Eq. 10）使用 model-to-depth point-to-plane correspondence：

$$
E_{\mathrm{depth}}=
\sum_{(v,u)\in\mathcal C}
\left[
(V_W(v)-V_d^t(u))^\top N_W(v)
\right]^2.
$$

Smooth term（Eq. 11）让邻接 nodes as-rigid-as-possible：

$$
E_{\mathrm{smooth}}=
\sum_{j\in\mathcal G}\sum_{i\in\mathcal N_j}
\left\|
W(V_{D,i})\bar V_{D,j}-W(V_{D,j})\bar V_{D,j}
\right\|_2^2.
$$

第一项贴合 current depth；第二项防止邻域产生不连续、非物理 deformation。与 DynamicFusion 的区别不是这套 optimizer 本身，而是 MixedFusion 只在被分出的 dynamic components 上运行，因此大幅减少 variables 和 frame time。

Rigid moving object 也可由这套 non-rigid field 表示为近似一致的 node transforms；系统不显式估 instance-level rigid body identity。

### 5. Mixed voxel allocation：动态靠 model 生长，静态靠 depth 开图

系统使用 Voxel Hashing 风格 sparse blocks，每 block 为 $8^3$ voxels。实验 voxel edge 为 5 mm，因此 block 尺寸 $0.04^3$ m；dynamic node sampling radius 为 3.6 cm，略小于一个 block。

#### 为什么传统 depth allocation 污染动态物体

static pipeline 将 live depth 只按 camera pose $T_g$ back-transform 到 canonical space，并沿每条 ray 的 TSDF truncation band 分配 blocks。对 moving object，还缺少 local warp $W^{-1}$；同一 physical surface 会在多个 canonical positions 开 blocks，生成 ghost geometry。

#### Dynamic model-based allocation

动态部分不从 raw depth 决定 canonical block location，而从已有 dynamic nodes 向周围局部扩张：

1. 找到包含每个 dynamic node 的已分配 block；
2. 以 node 在 block 内相对中心的 octant，把每轴分成正/负方向；
3. 沿对应 3 个 axis directions 及组合方向检查 7 个 adjacent blocks；
4. 未分配者被创建，使 canonical dynamic surface 在 boundary 附近增长；
5. 新/旧 voxel centers 经 $W$ 和 $T_g$ warp 到 live frame，再用 current depth 更新 TSDF。

因为 allocation location 由 canonical dynamic model 决定，moving depth 不会直接在错误 canonical location 开洞；同时局部 7-block growth 能逐帧补全先前未见过的邻近 surface。

局限也很直接：它只能从已有 dynamic support 邻域逐步扩张，无法突然生成与旧 surface 有大 gap 的新 component，也无法处理 topology change。

#### Static depth-based allocation

先把更新后的 live dynamic model project 到 current depth image，遮掉其覆盖 pixels；剩余 depth 视为 static。然后按 InfiniTAM/Voxel Hashing 的 conventional ray-based allocation 和 rigid camera pose 更新 static blocks。

这两条规则组成“mixed” allocation：

| 部分 | canonical block 从何处产生 | 为什么 |
| --- | --- | --- |
| Dynamic | 已有 dynamic nodes 周围 7 个邻接 blocks | local motion 已知，避免 moving depth 错投 canonical space |
| Static | 去除 dynamic projection 后的 depth rays | static scene 只受 camera motion，可自由扩展大场景 |

### 6. TSDF 更新

对 voxel $\mathbf x$，先把其中心 warp 到 live frame（Eq. 15）：

$$
\mathbf x_t=T_g^tW(\mathbf x)[\mathbf x^\top,1]^\top.
$$

再由 live depth 得 projective signed distance（Eq. 14），记为 $\operatorname{psdf}(\mathbf x)$。若落在 truncation band 内，以 running weighted average 更新（Eqs. 12–13）：

$$
v_t(\mathbf x)=
\frac{v_{t-1}(\mathbf x)w_{t-1}(\mathbf x)+
\min(\operatorname{psdf}(\mathbf x),\mu)}
{w_{t-1}(\mathbf x)+1},
$$

$$
w_t(\mathbf x)=\min(w_{t-1}(\mathbf x)+1,w_{\max}).
$$

动态 voxel 的 query point 使用 local warp，静态 voxel 等价于 identity local warp。更新后 live dynamic model 再 warp 回 canonical，作为下一帧的长期 model。

### 一个完整示例

设用户移动 RGB-D camera 扫描房间，一名人物在桌边转身：

1. 普通 ICP 会部分跟随人体转动；S-ICP 发现墙、桌、椅等多数 correspondence 可由同一 rigid camera pose 对齐，让人体 residual 饱和，不再主导 pose。
2. 人体正面 nodes 出现大量高 residual；node graph 把与其连接的背面 nodes 一并标为 dynamic。
3. 仅人体 component 进入 deformation-graph optimizer，估计转身的 non-rigid/local rigid motion；墙和家具不增加 deformation variables。
4. 人体 canonical blocks 沿已有 nodes 邻域逐步生长，并先 warp 到当前姿态再融合 depth；不会在每个新姿态复制一具人体。
5. 更新后的人体投影回 depth mask，其余 pixels 用于 conventional static allocation，桌后被人遮住的区域等重新可见后可继续补全。
6. 若人物拿起一个原先被当作 static 融合的球，旧球不会从 static TSDF 中去融合；系统又会在新位置融合 moving ball，最终出现两个球，这正是 Fig. 11 的失败案例。

## 实验关键数据

### 实验设置

- **硬件**：3.40 GHz 8-core CPU、16 GB RAM、NVIDIA GTX TITAN X。
- **输入**：single depth/RGB-D sensor；算法不使用 colour。
- **几何参数**：voxel 5 mm，voxel block $8^3$、边长 4 cm；dynamic node radius 3.6 cm。
- **基线**：traditional ICP、ORB-SLAM；KinectFusion、InfiniTAM；DynamicFusion 及其加入 depth-based allocation 的变体。
- **场景**：论文 Fig. 12 展示 6 个含人物、curtain/cloth 等 dynamic objects 的 sequences，但未公开名称、帧数、depth dataset 或下载链接。
- **指标**：只有一个 60-frame camera pose error curve；geometry/fusion comparison 主要为 figures/video，没有 surface accuracy、completeness、Chamfer、trajectory drift 或统计显著性。

### 实时性能（Table 1）

| Scene | S-ICP | Model segmentation | Local motion | Voxel allocation & fusion | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure static | 6 ms | 4 ms | — | 3 ms | 4 ms | **17 ms** |
| Dynamic object | 6 ms | 7 ms | 20 ms | 4 ms | 4 ms | **41 ms** |

- 17 ms 理论约 58.8 FPS，论文概括为 >50 FPS。
- 41 ms 理论约 24.4 FPS，论文报告平均 25 FPS、通常 20–30 FPS。
- dynamic non-rigid motion estimation 占 20/41 ≈ **48.8%**，是主要额外开销；只对 segmented dynamic region优化是实时性的关键。
- 若低于 25 FPS，系统会 drop frames 以避免延迟。这保持 online responsiveness，但 frame gap 增大会反过来使 ICP/non-rigid tracking 更困难；论文未报告 dropped-frame rate。
- 没有显存、host memory、voxel block 数、scene scale 或序列增长曲线，无法判断长时间/大房间的 memory scalability。

### S-ICP 评估（Figs. 6–7）

作者在人物转动的 sequence 上比较 S-ICP、traditional ICP、ORB-SLAM 前 60 frames：

- ground truth 并非 motion-capture，而是**手工分割掉动态人体后，对纯静态数据运行 ICP**得到的 pseudo-GT；
- pose error 将三 Euler angles（radian）和三 translation components 合并，angle/translation 权重分别为 2/1，沿用 BundleFusion 的做法；
- 图中 S-ICP drift 明显较小；前约 20 frames ICP 接近 S-ICP，随人体 local motion 累积后开始偏离；
- residual maps 中，S-ICP 的高误差集中于人体，ICP/ORB-SLAM 则在 static 与 dynamic regions 都有较大误差，支持“camera pose 错误会污染全景”的解释。

**证据边界**：论文没有给 mean/median translation error、rotation error、ATE/RPE 或数值表；curve 也只有一个 sequence。pseudo-GT 与被测方法同属 ICP family，并依赖 manual segmentation，不等价于外部 tracking ground truth。

### Mixed voxel allocation 评估（Fig. 8）

从 frames 700、800、1010 展示 dynamic object 逐渐移开后原遮挡区的重建：

- MixedFusion 先 mask dynamic depth，再对 static depth allocation，blank zone 被正确开 blocks 并逐渐补全；
- InfiniTAM 式原 allocation 将 moving-object depth 在 canonical static volume 中错误开 blocks，产生明显 false geometry。

该图直观验证分开 allocation 的必要性，但没有 voxel precision/recall、ghost surface area、memory overhead 或多 sequence 数值统计。

### 与静态 fusion 的比较（Fig. 9）

KinectFusion/InfiniTAM 在 object moving 后同时出现：

- dynamic object ghosting；
- camera pose 被 moving object 拉偏，连 static chair/table 也出现 artifacts。

MixedFusion 保留 static background 并融合 dynamic object。比较是同类系统级定性结果，但论文未说明所有 baselines 的 exact parameters、frame dropping 或 GPU implementation 是否严格对齐。

### 与 DynamicFusion 的比较（Fig. 10）

- 原 DynamicFusion 从已有 geometry boundary 分配 dense voxels，无法重建与当前 model 有较大 gap 的新 static monitor。
- 给 DynamicFusion 加 raw-depth allocation 后虽能出现 monitor，却因其把全 FOV 当 dynamic、camera drift 较大，canonical monitor/table 的位置和朝向错误。
- MixedFusion 以 static camera tracking + component-wise dynamic deformation + mixed allocation 得到较完整房间。

这支持 MixedFusion 的“组合贡献”而非证明其 non-rigid optimizer 优于 DynamicFusion；动态 motion 部分本来就基本沿用后者。

### 失败案例（Fig. 11）

一颗最初静止在桌上的篮球已被写入 static TSDF；男孩推动它后，球变 dynamic 并在新位置重建；再次静止时，旧 static geometry 没有被 remove/reintegrate，最终出现两颗球。

这个 failure 揭示比“topology change”更一般的问题：系统每帧有 dynamic/static segmentation，却没有 persistent object identity、voxel ownership 或 reversible fusion。**静态 → 动态的状态变化会留下不可撤销历史。**

## 关键发现

1. **robust camera tracking 和 motion segmentation 可以共享同一 residual model**：S-ICP 先让多数 static points 决定 pose，再把饱和 residual 当 dynamic probability，避免完全交替的鸡生蛋流程。
2. **动静分离的最大系统收益是计算与 fusion policy 同时解耦**：动态区域才需要高维 deformation；静态大场景继续享受成熟 sparse TSDF allocation。
3. **model-based allocation 是 dynamic fusion 的必要补充**：moving depth 不能只靠 global camera pose写入 canonical grid，必须让已有 canonical support 和 estimated warp 决定新 blocks 在哪里生长。
4. **connectivity 可便宜地补 invisible labels**：不需要 semantic network 即可把人的背面带入 dynamic graph；但接触/连接也正是它的 failure mode。
5. **25 FPS 的实时性有明确边界**：Titan X、5 mm voxels、dynamic region 较小，且不足时 drop frames；没有回环、颜色、全局 reintegration。
6. **论文证明更偏“可行性”而非精确度**：系统设计和视觉结果有说服力，但几何、pose、motion 的定量证据远弱于现代 benchmark 标准。

## 亮点与洞察

### 论文亮点

- **问题定义领先当时**：在 static room fusion 与 isolated non-rigid capture 之间提出“同时保留大场景和动态对象”的中间任务。
- **S-ICP 一体化设计简洁**：bounded sigmoid 既是 robust estimator 又是 segmentation cue，不依赖 learning/semantic category。
- **模块职责清晰**：global pose 由 static part，local warp 由 dynamic part；两者分别控制 depth-based/model-based allocation。
- **充分复用成熟组件**：TSDF、Voxel Hashing、DynamicFusion deformation graph 都保留，只改变关键接口，因而能达到实时。
- **性能分解透明**：Table 1 报告每个 pipeline stage 的毫秒数，明确 local motion 是瓶颈。
- **失败案例有信息量**：篮球 duplication 清楚展示 irreversible fusion 与 state transition 的根本问题。

### 我的洞察

1. **S-ICP 是一种 latent segmentation robust estimator**：它假设 residual distribution 由 dominant static inliers + minority moving outliers组成；segmentation 不是外部输入，而是 pose estimator 的隐变量结果。
2. **Mixed allocation 是按“谁拥有 canonical correspondence”分配内存**：static pixel 的 canonical location 由 camera pose确定；dynamic surface 的 canonical location必须由 model+warp确定。两者不能共享同一 allocation rule。
3. **connected component 既是传播先验也是 instance proxy**：在没有 semantics 的年代，它用 topology 近似 object identity；一旦接触发生，topology 和 semantic instance就不再一致。
4. **实时性来自 conditional compute**：昂贵 non-rigid solver 只在 residual-selected region运行。这个思想可迁移到现代 dynamic 3DGS/NeRF：先估 uncertainty/motion mask，再只对动态 primitives启用 deformation capacity。
5. **不可逆 TSDF 是 dynamic SLAM 的核心矛盾**：一旦错误/过时 observation 融入 running average，没有 ownership 就无法撤销。现代系统需要 object-level submaps、time-aware TSDF、de-integration 或 probabilistic occupancy history。
6. **它不是真正的 RGB-D appearance reconstruction**：尽管传感器可提供 color，论文完全是 depth geometry pipeline；后续 DTexFusion 可视为同一研究线对 dynamic texture 的补足。

## 局限与展望

### 作者明确承认的局限

- **不使用 colour**：缺少 photo cues 会限制 camera tracking、真实感和 texture；作者计划结合 colour、loop closure 和 geometry。
- **不支持 topology changes**：沿用 DynamicFusion fixed-topology deformation graph，篮球移动导致 duplicate model。
- **大面积动静连接会破坏 segmentation**：connectivity-based component 无法分开贴合/接触的 dynamic object 与 static scene；作者建议加入 semantic information。
- **动态 reconstruction 能力受基础方法限制**：本文重点是 static/dynamic combination，而非提升单对象 non-rigid tracking；复杂/快速 motion 仍会失败。

### 独立分析

1. **动态占比假设未验证**：S-ICP 的 maximum-consensus逻辑要求 static region占多数；近景人物遮满画面、camera朝向 moving curtain时可能选错 global frame。
2. **motion magnitude 低于 threshold 会 ghost**：缓慢、沿 tangent/point-to-plane不敏感的运动可能被当 static，先污染 static TSDF，之后再变 dynamic 已无法撤销。
3. **只用 connected components 粒度过粗**：同一 mesh component 中局部 waving curtain 和固定 attachment可能被整体标 dynamic；反之，重建孔洞导致 graph断裂又会漏标背面。
4. **没有 persistent instance/state machine**：每帧重新分类 surface，但没有 object identity、static↔dynamic transition处理、submap ownership或 de-integration，这是 Fig. 11 duplication 的根因。
5. **无 loop closure/global consistency**：长 indoor scan 的 camera drift 不会被纠正，canonical TSDF 也没有 BundleFusion式 reintegration；360° room result 可能累积变形。
6. **评测定量严重不足**：无 public dataset、无 geometry GT、无 surface/motion error，S-ICP仅单序列 pseudo-GT curve；无法判断 across-scene robustness。
7. **基线公平性信息不足**：KinectFusion/InfiniTAM/DynamicFusion 的实现、参数和硬件细节未完整列出，视觉比较可能受工程优化差异影响。
8. **关键超参数缺失**：论文未给 S-ICP $k$、`thr`、component dynamic count threshold、$v_{smooth}$、iteration count等完整复现值；无代码进一步放大问题。
9. **dynamic allocation 只能局部生长**：每 node检查 7 个邻块适合连续 surface reveal，不适合快速位移后出现 disconnected unseen parts 或新对象进入 FOV。
10. **动态遮罩投影可能过度删除 static depth**：粗/错误 live dynamic model project 到 image 后，其覆盖区域全部不参与 static fusion；边界误差会减慢背景补全。
11. **frame dropping 是隐藏 trade-off**：低于 25 FPS 时丢帧虽防 latency，却增大 inter-frame motion，可能造成 tracking崩溃；论文没有 robustness curve。
12. **不处理 appearance 和 semantics**：输出只有 geometry/motion，难直接服务真实感 VR/AR；在今天更适合作为 dynamic mapping architecture 而非完整 digital twin pipeline。

### 建议的后续实验

- 用 motion-capture camera trajectory 和 scanned geometry GT，在不同 dynamic pixel ratio、速度、遮挡和接触比例下报告 ATE/RPE、Chamfer、completeness与 segmentation IoU。
- 对 S-ICP 扫描 $k$、`thr`，与 Huber/Tukey/Geman–McClure、RANSAC/trimmed ICP 比较 breakdown point 和 GPU time。
- 引入 object-level static/dynamic submaps与 voxel ownership：物体开始运动时 de-integrate 其 static observations，再迁移到 dynamic volume。
- 用 semantic/instance masks 辅助但不替代 residual segmentation，测试 person touching furniture、curtain attached to wall 等连接场景。
- 集成 BundleFusion/ElasticFusion loop closure，对长走廊和完整 360° room报告 global consistency。
- 报告 scene duration–voxel blocks–memory曲线、drop-frame rate、dynamic region size–runtime曲线，并在现代 GPU/移动设备复现。
- 将 depth geometry 与后续 DTexFusion 式 dynamic texture或 neural appearance联合，评估 relighting/novel-view，而不只 geometry mesh。

## 与相关工作的对比

| 方法 | 场景假设 | Camera tracking | Dynamic motion | Volume allocation/fusion | Mixed scene 结果 |
| --- | --- | --- | --- | --- | --- |
| KinectFusion | 全静态、小/中场景 | 全 depth ICP | 无 | dense/static TSDF | moving object与背景均 ghost/drift |
| InfiniTAM / Voxel Hashing | 全静态、大场景 | rigid ICP | 无 | sparse depth-ray blocks | moving depth错误写入 canonical volume |
| BundleFusion | 静态大场景 | global pose + loop closure | 无 | online reintegration | global一致，但不建模动态对象 |
| DynamicFusion | 视野内整体为单一 non-rigid scene | global+non-rigid耦合 | deformation graph | model boundary growth | 可跟踪单对象，难扩展完整房间/新静态区域 |
| VolumeDeform | 单/少量 dynamic object | SIFT + non-rigid volume | deformation graph | dynamic volume | robustness更强，但仍非 mixed room design |
| **MixedFusion** | static room + 少量 rigid/non-rigid dynamics | S-ICP 由 static majority决定 | 仅 dynamic components用 graph warp | dynamic model-based + static depth-based | 同时融合房间和移动对象，约25 FPS |

关键继承关系：

- **KinectFusion/InfiniTAM** 提供 static TSDF、point-to-plane ICP 与 sparse voxel allocation；
- **DynamicFusion** 提供 canonical/live model、deformation graph、dual-quaternion blending、non-rigid ICP 与 dynamic TSDF update；
- **MixedFusion** 的新增层是 S-ICP 动静解耦、connectivity segmentation 和两套 allocation policy 的调度。

## 启发与关联

- **对 dynamic 3DGS/4D mapping**：可把 S-ICP 的 sigmoid residual转成 Gaussian motion probability，只让高概率 primitives进入 deformation network；static Gaussians共享 rigid camera pose并承担 localization。
- **对 object-level SLAM**：mixed allocation对应 static world submap + movable object submaps。现代实现应显式维护 object ID和 reversible observations，解决篮球 duplication。
- **对 robust registration**：把 robust loss输出同时解释为 segmentation probability，可让 pose和mask相互促进；但应加入 temporal prior和 uncertainty calibration，避免 threshold抖动。
- **对 occlusion-aware fusion**：动态 model projection先占用/解释 depth，再把剩余 observation交给 static map，是一种 explain-away机制；可推广到多人、机器人和移动家具的 layered fusion。
- **与 DTexFusion 的连接**：MixedFusion恢复 geometry/motion但没有 colour；DTexFusion进一步以 basic texture × changing maps重建 pose-dependent appearance。两者组合可形成早期完整 dynamic RGB-D asset pipeline。
- **对实时系统设计**：把高维 solver限制在 residual-selected ROI，比全场统一模型更可控；这种 conditional compute原则在 edge/AR devices仍有效。

## 评分

| 维度 | 评分（10 分） | 理由 |
| --- | ---: | --- |
| 创新性 | 8.7 | 在 static room fusion 与 DynamicFusion 之间提出清晰的 mixed-scene系统，S-ICP + mixed allocation 的组合在当时很有新意。 |
| 技术可靠性 | 7.8 | 公式和各模块因果关系完整，实时分解合理；但依赖 static majority/connectivity/fixed topology，关键超参数与代码缺失。 |
| 实验充分度 | 5.6 | 有 stage timing、pose curve和多组视觉比较，却缺公开数据、geometry/motion数值、跨场景统计与真实 pose GT。 |
| 写作清晰度 | 8.3 | pipeline、S-ICP、segmentation和allocation讲解直观，failure case坦诚；部分线性化/阈值和实现细节不足。 |
| 实用 / 研究价值 | 7.8 | Titan X 上约25 FPS且系统思想可迁移；无 colour/loop closure/state transition与不可逆TSDF限制真实长期部署。 |

**总体推荐：值得细读。** 它是理解 dynamic SLAM 如何从“删掉运动物体”走向“同时建图并保留运动对象”的重要早期系统；今天最有价值的是动静 conditional compute、residual segmentation 和 object-aware fusion policy，而非直接复用其完整实现。

## 阅读结论

- **最值得记住的点**：static 与 dynamic geometry 不只是 motion model不同，连 canonical voxel该从 depth还是从已有 model生长都不同；mixed scene必须在 tracking、segmentation、allocation三层同时解耦。
- **最需要怀疑的点**：25 FPS 和视觉结果证明可行性，但 pose仅单序列 pseudo-GT、geometry无数值，S-ICP又依赖 dynamic minority；实际鲁棒范围无法从论文充分判断。
- **最值得复现或继续验证的点**：实现 S-ICP 与现代 robust ICP作 controlled comparison，并给 TSDF加入 instance ownership/de-integration，专门验证 static→dynamic→static物体是否还会留下重复模型。

## 相关论文

- **KinectFusion: Real-Time Dense Surface Mapping and Tracking**（ISMAR/UIST 2011）— rigid camera tracking与TSDF fusion基础。
- **Real-Time 3D Reconstruction at Scale Using Voxel Hashing**（ACM TOG 2013）— MixedFusion稀疏 voxel block组织与大场景内存基础。
- **DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-time**（CVPR 2015）— canonical/live model、deformation graph和动态融合的直接基础。
- **BundleFusion: Real-Time Globally Consistent 3D Reconstruction**（ACM TOG 2017）— static global pose、loop closure和surface reintegration；MixedFusion明确计划未来结合。
- **VolumeDeform: Real-Time Volumetric Non-rigid Reconstruction**（ECCV 2016）— 以 SIFT correspondences增强动态 non-rigid tracking 的同期方法。
- **Real-Time Geometry, Albedo, and Motion Reconstruction Using a Single RGB-D Camera**（ACM TOG 2017）— 同期单对象 geometry/appearance/motion联合恢复。
- **DTexFusion: Dynamic Texture Fusion Using a Consumer RGBD Sensor**（TVCG 2022，online 2021）— 同一研究线对动态纹理压缩、对齐与 novel-view appearance 的后续补充。
