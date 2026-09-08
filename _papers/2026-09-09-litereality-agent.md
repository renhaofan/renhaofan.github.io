---
title: "LiteReality-Agent: An Agentic System for Interactable 3D Indoor Scene Reconstruction"
subtitle: "从 iPhone LiDAR 与 RoomPlan 扫描出发，以确定性场景初始化和视觉反馈 Agent 循环生成可编辑、可交互的 Blender 室内场景"
authors: "Zhening Huang, Yueyan Li, Johnathan Chiu, Xiaoyang Lyu, Matt Zhou, Yuxin Yao, Joan Lasenby, Shangzhe Wu"
venue: "Technical blog / system preview"
year: 2026
date: 2026-09-09
paper_url: "https://litereality.github.io/agent/litereality-agent-post/litereality-agent.pdf"
code_url: "https://github.com/LiteReality/LiteReality-Agent"
project_url: "https://litereality.github.io/agent/"
tags:
  - 3D Scene Reconstruction
  - Agentic System
  - RoomPlan
  - Blender
  - Image-to-3D
  - Interactive Scene
summary: "LiteReality-Agent 将 iPhone LiDAR、RGB-D、相机轨迹与 RoomPlan 语义布局组织成 ingest→reconstruct→seed→author→publish 流水线：先以规则和生成模型建立尺度正确的 seed room，再让 coding agent 围绕单一 Room.py 执行选视角、渲染、对照、批评和修正，最终导出可编辑 Blender 场景与 GLB。系统的工程闭环和开放实现很有价值，但当前公开材料只有案例视频与 gallery，没有场景数量、重建误差、成功率、耗时、Agent 成本或用户研究，因而更适合作为系统预览而非已完成量化验证的研究论文。"
permalink: /papers/litereality-agent/
---

> **阅读依据**：本笔记基于 [项目页](https://litereality.github.io/agent/)、项目内公开的 17 页[技术博客 PDF](https://litereality.github.io/agent/litereality-agent-post/litereality-agent.pdf)、网页版技术博客，以及截至 2026-09-09 的[官方代码](https://github.com/LiteReality/LiteReality-Agent)。代码采用 Apache-2.0 license。
>
> **材料边界**：项目页的论文按钮仍标为 “Technical report · soon”。现有 PDF 是技术博客的排版版本，不包含标准论文中的数据集说明、定量主表、消融实验或统计分析。下文会分别标注作者展示、代码事实与独立判断，不把 gallery 视频当作 quantitative benchmark。
>
> **定位说明**：这是一个把已有感知、生成、Blender 与 coding-agent 能力整合起来的端到端系统。它所说的 reconstruction 同时包含测量驱动的布局恢复和生成驱动的语义补全，不能简单等同于逐像素或逐表面的几何重建。

## 一句话总结

LiteReality-Agent 把手机扫描转化为结构化证据包，先用 RoomPlan、参考帧、procedural generation 和 TRELLIS 建立尺度与语义受约束的 seed room，再让 coding agent 以 Blender Python 文件 `Room.py` 为可执行场景语言，循环执行“编辑—渲染—比较—批评”；其核心价值是得到可编辑、可交互、可重新布置并能输出完美渲染标签的场景，但当前公开证据尚不足以判断真实重建精度、稳定性和成本。

## 背景与动机

- **传统扫描输出难以交互**：NeRF、Gaussian Splatting 或融合点云擅长重放观测外观，却通常不直接提供干净的对象边界、关节、材质、碰撞体和可编辑场景图。
- **纯生成式 3D 缺少现场约束**：image-to-3D 可以补出漂亮资产，但单张参考图不能可靠恢复房间尺度、对象朝向、遮挡部分及真实布局。
- **手工建模成本高**：把一次手机扫描变成可在 Blender、游戏引擎或浏览器中使用的场景，通常需要人工整理几何、资产、材质、灯光和交互关系。
- **作者切入点**：不要求某个单体模型一次性完成所有任务，而是把传感器测量、RoomPlan 结构先验、专用 3D 生成器、程序化建模和视觉语言模型组织成可恢复、可检查的 agentic pipeline。

**核心研究问题：能否从普通用户的一次 iPhone 扫描出发，自动得到既接近现场观感，又具有对象级结构、可编辑性与交互能力的完整室内场景？**

## 核心问题

1. 如何把 RGB、深度、相机参数、点云和 RoomPlan USDZ 中坐标系不同、噪声不同的证据统一起来？
2. 对象应该采用程序化建模还是 image-to-3D，怎样在结构可控性和视觉复杂度之间分流？
3. 初始布局正确但观感粗糙时，Agent 如何基于真实扫描持续修正，而不是自由幻觉式“装修”？
4. 如何让长流程可恢复、可检查，并把结果压缩为一个可编辑的场景程序和标准 3D 资产？
5. 规则检查、VLM critic 和最终 checklist 能否真正保证几何正确与物理可交互？

## 方法详解

### 整体框架

<div class="mermaid">
flowchart LR
    A[iOS 扫描<br/>RGB / depth / confidence / poses] --> B[ingest]
    R[RoomPlan USDZ<br/>房间、开口、对象框与语义] --> B
    B --> C[reconstruct<br/>对象参考图与资产生成]
    C --> D{complexity router}
    D -->|结构、关节、交互优先| E[procedural generation<br/>Blender Python / Articraft]
    D -->|外观复杂、静态物体| F[TRELLIS image-to-3D]
    E --> G[seed<br/>按扫描位姿与尺寸组装]
    F --> G
    G --> H[author<br/>Agent 编辑 Room.py]
    H --> I[select views / render / grid]
    I --> J[compare / critic / collision QC]
    J -->|继续修改| H
    J -->|完成| K[publish]
    K --> L[Blender scene / GLB / Web walkthrough]
    K --> M[depth / normal / albedo / segmentation]
</div>

公开实现把流程分成两个大阶段、五个可恢复 stage：

```text
scene initialization                 realism authoring
ingest → reconstruct → seed          author → publish
```

前半段尽量采用确定性测量和受控生成，确保“房间里有什么、在哪里、多大”；后半段才让 Agent 反复观察与编辑，解决材质、灯光、局部形状、摆放关系和整体真实感。

### 1. 多模态扫描证据的结构化

LiteReality iOS 扫描同时记录：

- ARKit RGB frames；
- depth 与 confidence maps；
- camera intrinsics、poses 和 timestamps；
- 由深度与位姿导出的 point cloud；
- RoomPlan USDZ 中的墙面、门窗、家具 bounding boxes 与语义类别。

这里最关键的不是把所有信号融合成单一隐式场，而是保留其不同用途：RGB 用于外观参考，depth/point cloud 用于尺寸和表面证据，相机参数用于选择可见帧与对照渲染，RoomPlan 则提供稳定的 room shell、开口和对象级 layout。这样后续每个资产都能关联回原始观测，而不是只依赖一张任意裁剪图。

### 2. 确定性的房间骨架与对象重建

系统先根据 RoomPlan 恢复墙、地板、天花板、门窗等结构，再为每个检测到的对象收集扫描参考图。随后通过 complexity router 选择两类生成路径：

| 路径 | 适合对象 | 优势 | 主要风险 |
| --- | --- | --- | --- |
| Procedural generation | 橱柜、桌椅等需要明确部件、关节或交互的对象 | 尺寸、拓扑、部件命名和 articulation 可控制；可直接修改 Blender Python | 形状模板和 Agent 编程能力限制细节上限 |
| TRELLIS image-to-3D | 雕塑、装饰物等外观复杂但主要静态的对象 | 能从参考图快速生成视觉丰富的 mesh/texture | 遮挡面与尺度可能幻觉，拓扑和可动结构难保证 |

程序化路径借鉴 Articraft，把 Blender Python 当作 scene language。生成资产最终不是孤立地摆到“看起来差不多”的位置，而是按扫描估计的 translation、rotation 和 bounding-box dimensions 组装成 seed room。

这个设计比“整张房间图直接 text/image-to-3D”更稳，因为几何测量和生成先验承担不同职责；但 router 的分类准确率、两种路径的失败率与人工返工量目前没有报告。

### 3. 以 `Room.py` 为中心的 Agentic realism authoring

后半段的核心状态是单一 `Room.py`：它定义对象、变换、材质、灯光和场景组装逻辑，可由 Blender 执行并重新导出。Agent 不直接在不可追踪的 GUI 状态上操作，而是在代码中完成编辑，然后通过渲染观察结果。

典型循环是：

1. 读取 seed room、对象清单与扫描证据；
2. `select_views` 选择能够观察目标对象、墙面或整个房间的 capture frames；
3. 修改 `Room.py`；
4. `render` 生成当前场景视图，`grid` 汇总多个角度；
5. 将渲染与扫描参考比较，使用 `critic` 找出缺失、比例、材质或布局问题；
6. 必要时用 `fetch_material` 获取材质，用 `check_collisions` 检查摆放冲突；
7. 在预算内继续修改，最后执行 quality pass 与发布。

代码中的 author 默认 step budget 为 100，保留 15 steps 用于结束阶段，hard maximum 为 140 turns。博客还描述了一种工程策略：较晚阶段关闭部分自检工具，把剩余 calls 留给收尾。它带来较可预测的 Agent 调用上限，但本质是固定预算，不是根据视觉误差收敛自动停止。

### 4. 双层质量控制

系统没有把质量完全交给 VLM。seed room 阶段先基于 AABB/OBB 和房间结构做确定性检查，包括：

- 对象低于地板、穿入地面或高于天花板；
- floating / sunk placement；
- 对象越出房间；
- 与墙面或其他对象冲突；
- fixture 覆盖门窗 opening。

最终 model-driven checklist 再检查门窗和玻璃、窗帘/窗户 articulation、材质与灯光、幻觉出的 props、对象和参考图的一致性，以及 collision-free placement。

两层检查分别覆盖“容易编码的硬约束”和“需要视觉判断的软约束”，架构上是合理的。不过 box-level overlap 不等于 mesh-level collision，更不能证明质量、摩擦、稳定性、关节范围等物理属性正确，因此作者也明确表示当前结果尚未 simulation-ready。

### 5. 发布与合成数据能力

流水线最终可输出：

- 可继续编辑和执行的 `Room.py`；
- Blender scene；
- GLB 与浏览器 walkthrough；
- 从 authored scene 精确渲染的 depth、normal、albedo、instance segmentation 和 semantic segmentation。

这里的“精确”是指这些通道由最终数字场景的 renderer 直接产生，属于该场景的 ground truth。它不意味着渲染 depth 与真实房间逐像素一致；若资产几何本身偏离现场，标签仍然只对合成场景内部正确。

### 6. Claude Code 与 Codex harness 并不等价

仓库支持 Claude Code 和 Codex CLI 驱动 Agent，但 `ARCHITECTURE.md` 明确列出能力差异：

| 能力 | Claude harness | Codex harness |
| --- | --- | --- |
| capability tools | 进程内调用 | 通过 stdio MCP bridge |
| step-budget 收尾 | `PreToolUse` 可触发 wind-down | 退化为 hard stop |
| shell allowlist | 可限制 | 不能禁用 shell |
| cost reporting | 支持 | 不报告 |
| object refinement | 支持 | 当前不支持 |

因此“支持两种 Agent”不代表两种配置会得到可直接比较的质量、成本或安全边界。复现时应记录 provider、模型版本、prompt、tool budget 和外部服务版本。

## 公开证据，而非论文式实验

### 当前展示了什么

- bedroom、meeting room 等 real scan 与 reconstruction 对照视频；
- 多个可在浏览器浏览的 gallery rooms；
- text-driven material editing、decoration、object rearrangement 与 intrinsic-buffer rendering；
- 一个完整开源流水线，可检查 stage、Agent tools、QC 和导出实现。

这些材料能支持以下较弱但有意义的结论：系统确实可以从真实扫描构造完整室内场景；输出不是固定 neural rendering，而是能移动对象、换材质和修改程序的结构化资产；代码架构也体现了项目页描述的主要闭环。

### 当前没有报告什么

| 评价维度 | 缺失证据 |
| --- | --- |
| 数据规模 | 扫描场景总数、对象数、失败场景与选择标准 |
| 几何 fidelity | Chamfer distance、depth error、layout/尺寸误差、pose error |
| 外观 fidelity | PSNR、SSIM、LPIPS、感知人评或 novel-view protocol |
| 系统稳定性 | 各 stage 成功率、重试率、崩溃率、人工修复比例 |
| 成本与效率 | 总 wall-clock、各 stage 时间、Agent tokens/cost、峰值 VRAM |
| 可交互性 | articulation 正确率、collision/physics 测试、任务成功率 |
| 用户价值 | 用户研究、编辑时间节省、与人工 Blender workflow 的比较 |

README 只给出了局部工程信息：image-generation API 通常低于 1 美元/scene；运行需要 macOS Apple Silicon 或 Linux、Blender 5.x（测试于 5.1），TRELLIS 与 GroundingDINO 可使用 Modal，或在本地使用至少 24 GB 显存的 NVIDIA GPU。这些不能替代端到端耗时与总成本报告。

## 亮点与洞察

### 系统亮点

- **输出目标选得好**：以可执行 `Room.py` 和 Blender/GLB 为终点，天然支持审计、局部重建、重排与二次编辑。
- **测量与生成职责分离**：RoomPlan/layout 约束全局结构，生成模型负责难以扫描完整的对象外观，减少纯生成式场景的自由度。
- **Agent 有真实反馈闭环**：不是一次性生成脚本，而是通过选视角、渲染、对照和 critic 迭代修正。
- **确定性 QC 与视觉 QC 结合**：几何规则不必浪费 VLM tokens，语义和美学问题又不被硬规则勉强表达。
- **工程上可恢复**：五个 stage 有明确产物边界，长流程失败后不必从扫描重新开始。
- **场景可派生多种标签**：若场景质量足够，能作为具身智能、视觉训练或渲染数据的低成本 authoring source。

### 我的洞察

- **个人分析：它更接近“测量约束的场景再创作”而不是纯 reconstruction。** 房间 shell 和对象布局来自扫描，但具体资产由 procedural/TRELLIS 与 Agent 补全；评价时必须把 layout accuracy、asset identity、visual plausibility 和 pixel fidelity 分开。
- **个人分析：`Room.py` 是系统真正的中间表示。** Agent 可用文本编辑、版本控制和执行反馈操作它；相比直接让 Agent 改 `.blend`，代码表示更可追踪，也更容易做规则检查与局部重建。
- **个人分析：系统的上限由 evidence grounding 决定。** Agent 越强并不自动越忠实；如果选到的参考帧遮挡严重或 RoomPlan 漏检，强生成模型反而可能输出更可信但更错误的细节。
- **个人分析：synthetic ground truth 的价值与 reconstruction fidelity 解耦。** authored scene 可以产生内部完美的 segmentation/depth，但要用于真实世界任务，还需测量 sim-to-real gap 和对象语义正确率。
- **个人分析：固定调用预算是一种产品约束，不是算法收敛保证。** 更合理的后续方案应结合 reference-view residual、未解决 checklist、碰撞数量和跨视角一致性来分配剩余预算。

## 局限与展望

### 作者明确承认的边界

- 当前场景还不是 simulation-ready；
- 主要测试约不超过 50 m² 的 single-room capture；
- 尚未验证大型空间、复杂 layout 和 multi-room 场景。

### 独立分析

- **缺少可复核评测**：案例视频无法回答成功率、平均误差和失败分布，也可能存在 showcase selection bias。
- **传感器误差未量化**：ARKit pose drift、LiDAR depth noise、confidence filtering、反光/透明表面和 RoomPlan 漏检都会传播到后续资产与布局。
- **复杂度分流没有验证**：procedural 与 TRELLIS 的 routing 准确率、跨类别泛化和错误路由代价均未知。
- **VLM critic 可能自洽而不真实**：author 与 critic 若共享模型家族或相似视觉偏好，可能共同接受一个语义合理但与现场不符的结果；需要独立人评或冻结的第三方 evaluator。
- **碰撞检查仍较粗**：AABB/OBB 规则能发现明显摆放错误，但不能证明网格无穿插、物体稳定、关节正确或可完成机器人交互。
- **复现依赖外部服务**：reference image generation、TRELLIS、Agent CLI 和模型版本都会变化；同一代码 commit 不必然产生同一场景。
- **provider 行为不一致**：Codex 与 Claude 的预算终止、工具权限、成本观测和 object refinement 能力不同，论文式比较必须固定 harness。
- **安全边界值得强化**：Agent 编辑可执行 Blender Python，且 Codex harness 无法关闭 shell；在处理不可信 capture/package 或共享运行环境时，需要更严格的 sandbox 与产物审计。

### 建议的后续实验

1. 发布固定 train/dev/test capture set，覆盖小房间、大空间、透明物体、镜面、严重遮挡和 multi-room。
2. 分别测量 room layout、object detection、asset geometry、pose/scale、appearance 和 articulation，而不是只给一个“realism”主观分数。
3. 报告每个 stage 的成功率、wall-clock、GPU/CPU 峰值、API 与 Agent cost、人工介入分钟数。
4. 做 `RoomPlan only`、`+ generated assets`、`+ agent authoring`、`+ quality pass` 的逐级消融。
5. 对比 procedural/TRELLIS oracle routing、自动 routing 与单一路径，统计每类对象的失败模式。
6. 用独立人类评审和第三方 VLM 交叉评估，避免 author-critic 自我偏好。
7. 在固定输入、固定模型版本和多个 random seeds 下报告稳定性，并保存完整 prompts、tool traces 与中间 `Room.py`。
8. 若主张 simulation-ready，增加 mesh collision、support relation、mass/friction、joint limits 和机器人任务成功率测试。

## 与相关工作的对比

| 系统/方向 | 主要输出 | 现场几何约束 | 对象级可编辑/交互 | 主要差异 |
| --- | --- | ---: | ---: | --- |
| NeRF / 3D Gaussian Splatting reconstruction | radiance field / splats | 强 | 通常弱 | 视图重放强，但对象结构和交互需额外提取 |
| Apple RoomPlan | 参数化房间布局与对象框 | 强 | 中等 | 尺度与语义稳定，但缺少高保真对象外观 |
| TRELLIS | 单图生成 3D asset | 弱 | 取决于生成 mesh | 外观补全强，缺少房间级测量与真实布局约束 |
| Articraft | Agent 生成可交互程序化资产 | 依赖参考输入 | 强 | 聚焦单资产结构与 articulation，不负责完整扫描流水线 |
| **LiteReality-Agent** | `Room.py`、Blender scene、GLB 与渲染标签 | **RoomPlan + RGB-D** | **强，目标如此** | 把测量、资产生成、Agent authoring 与发布连接成端到端系统 |

LiteReality-Agent 最值得关注的不是发明了新的底层 reconstruction representation，而是重新定义交付物：从“看起来像现场的可渲染表示”推进到“能继续编程、编辑和交互的场景工程”。这也使它必须接受比 novel-view synthesis 更复杂的评价体系。

## 启发与关联

- **Scene-as-code**：用可执行脚本作为 Agent 与 DCC 工具之间的共享状态，适合版本控制、单元测试、约束修复和可重复发布。
- **Evidence-budgeted agent**：让 Agent 的每次编辑都绑定一组扫描视角、几何约束和未解决 checklist，可减少无依据的视觉美化。
- **按对象自适应工具路由**：静态视觉细节交给生成模型，结构和 articulation 交给程序化建模；相同思想可迁移到机器人资产生成和数字孪生。
- **硬约束先行、软判断后置**：把尺寸、地面接触、房间边界和 collision 等规则检查放在 VLM 之前，可显著降低搜索空间与 token 消耗。
- **假设：基于不确定性的预算分配**。可用 RoomPlan confidence、可见视角数、depth variance、render-reference residual 和 critic disagreement 决定哪些对象值得更多 Agent calls。
- **假设：把 provenance 写入场景图**。每个 mesh/材质/尺寸保存来源帧、生成器版本与置信度，能够让后续用户区分“扫描到的事实”和“模型补出的猜测”。

## 为什么暂不评分

公开材料足以分析系统设计与代码实现，却不足以公平评价实验充分度和技术可靠性：没有正式 technical report、统一测试集、主表、消融、成本统计或失败率。此时给出 10 分制总分会把项目展示质量误当成研究证据，因此本笔记仅给出暂定阅读建议。

**暂定阅读建议：值得关注并复现 pipeline，但不宜据现有案例断言达到高保真、稳定或 simulation-ready 的室内重建。**

## 阅读结论

- **最值得记住的点**：把手机测量、对象生成和 Agent 视觉反馈统一到 `Room.py` 这一可执行场景表示中，使 reconstruction 的终点从“可看”变成“可编辑、可交互、可发布”。
- **最需要怀疑的点**：项目展示的 realism 是否能稳定推广到未筛选场景；目前没有规模、误差、成功率、耗时、总成本或人工修复量支持这一判断。
- **最值得复现或继续验证的点**：固定 20–50 个公开 capture，完整记录每个 stage 的产物、失败、Agent trace、成本和人工介入，并分别评估 layout、asset、appearance 与 articulation。

## 相关工作与资源

- [LiteReality](https://arxiv.org/abs/2507.02861) — 该项目所使用的移动端扫描与场景捕获基础。
- [Articraft](https://articraft3d.github.io/) — 以 Agent 和 Blender Python 构造可交互 3D 资产，支撑 procedural 路线的思想来源。
- [TRELLIS](https://arxiv.org/abs/2412.01506) — image-to-3D 资产生成模块，用于视觉复杂的静态对象。
- [Apple RoomPlan](https://developer.apple.com/augmented-reality/roomplan/) — 提供房间结构、开口、对象框与语义先验。
- [BlenderMCP](https://github.com/ahujasid/blender-mcp) — 相邻的自然语言/Agent 控制 Blender 工具方向；LiteReality-Agent 的重点则是扫描证据驱动的完整重建流程。
- [项目页](https://litereality.github.io/agent/) / [官方代码](https://github.com/LiteReality/LiteReality-Agent) / [LiteReality iOS App](https://apps.apple.com/gb/app/litereality/id6774158260)
