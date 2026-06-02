# TRACE 当前模型架构说明

更新时间：2026-06-02

本文档记录当前 TRACE 模型从输入到输出的完整架构、训练损失、设计动机、创新点和仍未解决的问题。它是后续继续训练、审查和上传共享的架构基准文档。

当前结论需要分开看：

- 架构层面：当前实现已经从普通 multi-head prototype 推进为 `interaction sites -> pair interaction field -> finite-state AR trace program -> token-conditioned Site/Pose/Affinity realizers` 的 strict trace-centered 主路径。
- 实验层面：当前严格 generated trace 路径还没有达到最终性能目标。最新全量 CASF generated path 约为 RMSE `1.73`，可以说明架构链路跑通，但不能宣称 TRACE 已经优于成熟 SOTA 或完成最终有效性证明。

## 1. 目标定位

TRACE 的目标不是把亲和力、结合位点、接触、pose/refinement 做成几个并列 head，而是把它们统一到一条可解释的自回归 interaction trace program 中：

```text
protein structure + ligand structure
-> protein/ligand interaction sites
-> pair interaction field
-> Unified AR Trace Program
-> SiteRealizer / PoseRealizer / AffinityRealizer
-> binding site, contact, local pose evidence, affinity
```

核心思想是：模型先生成“相互作用程序”，再由程序驱动下游任务。亲和力不是直接从 pair encoder 读数，而是由 generated trace、site/contact evidence、pose confidence 和局部几何 evidence 汇聚得到。这样做的目的是让 affinity prediction 的依据能够追溯到具体 protein site、ligand site、interaction type、contact density、pose quality 和 affinity evidence token。

## 2. 输入

### 2.1 数据 split

当前 Phase-1 默认数据设置是：

- Train：PDBbind refined set minus CASF/core complexes。
- Validation：refined train 内划分 random validation，并保留 core-like validation 用于诊断 refined 到 CASF 的 domain shift。
- Test：CASF-2016 core set。
- Labels：实验 affinity label，加 PLIP/Arpeggio fused atommap interaction trace label。

最近完整实验报告中的 split 数量：

- train：`24546`
- val：`505`
- test：`285`
- core_like_val：`454`

CASF-2016 core 只作为 held-out test。不能用 CASF 标签调参、校准、早停、checkpoint selection 或后处理。

### 2.2 Protein 输入

Protein 不直接作为裸 PDB atom list 进入主模型，而是先转成 formal protein interaction-site tensor。当前 site feature version 为 `formal`，主要包含：

- residue identity 和 chain/residue/site id；
- functional group / sidechain group；
- donor、acceptor、charged、hydrophobic、aromatic、metal-coordination candidate 等 role；
- atom ids 和坐标；
- local frame / local coordinates；
- sidechain functional atom group；
- coarse chi / rotamer bin；
- residue local density、backbone/sidechain coverage fallback 标记；
- 可选 pretrained protein representation。

设计原因：结合位点和相互作用不是纯 residue-level 或 atom-level 二分类。formal site abstraction 能把 residue functional group、局部几何和 atom identity 统一起来，同时保留和 PLIP/Arpeggio exact atom labels 的对齐能力。

### 2.3 Ligand 输入

Ligand 当前使用 `ligand_site_mode=atom_group`，包括 atom-level site 和 group-level interaction site。group-level site 由 RDKit/OpenBabel atom identity 和 RDKit ChemicalFeatures 构造，保留 member atom ids：

- donor；
- acceptor；
- aromatic；
- hydrophobe；
- positive ionizable；
- negative ionizable；
- zinc binder / metal coordination proxy；
- aromatic ring centroid、ring plane/normal；
- charged group；
- rotatable bond context；
- local conformer state；
- atom index 到 PDB/OpenBabel atom id 的映射。

设计原因：protein-ligand interaction 往往发生在 ligand pharmacophore group 上，而不是单个孤立 atom 上。atom_group 输入既保留 exact atom supervision，又让 trace 可以生成更稳定的 ligand site token。

### 2.4 PLIP + Arpeggio fused trace label

严格 trace label 入口为 `plip_arpeggio_fused`。融合逻辑：

```text
PLIP events + Arpeggio events
-> exact atom/site agreement
-> exact single-source
-> coordinate match
-> distance fallback
-> unified InteractionEvent JSONL
```

每个 event 记录：

- protein site / ligand site；
- protein atom ids / ligand atom ids；
- interaction type；
- distance、distance bin；
- angle / orientation bin；
- geometry confidence；
- affinity evidence；
- label source：agreed、PLIP-only、Arpeggio-only、conflict、fallback。

最近完整实验的 contact label source coverage：

- train exact fraction：`1.0000`
- val exact fraction：`1.0000`
- test exact fraction：`1.0000`
- coordinate/distance fallback：`0.0000`

这说明当前第一阶段 fused atommap label 已经不是主要依赖人工距离 fallback。

### 2.5 Pretrained representation

当前预训练输入通过统一 `pretrained_provider` schema 接入，支持：

- `none`
- `protein_only`
- `ligand_only`
- `protein_ligand`

当前主实验配置：

- provider：`saprot_unimol`
- mode：`protein_ligand`
- protein pretrained dim：`1280`
- ligand pretrained dim：`512`
- cache dir：`data/processed/phase1/pretrained/saprot_unimol_3di`

需要注意：SaProt 最标准输入是 AA + 3Di token。当前接口和 cache 已经按 3Di-aware provider 设计，但真实完整 3Di 质量依赖 Foldseek/3Di token 的覆盖情况；如果缺失，必须在报告中显示 masked fallback fraction，不能把纯序列 fallback 伪装成完整 structure-aware SaProt。

## 3. 主干架构

### 3.1 总体计算图

```text
ProteinSiteTensor, LigandSiteTensor
    -> SiteEncoder
    -> PairInteractionEncoder
    -> PairInteractionField
    -> ARTraceDecoder with finite-state grammar
    -> GeneratedTraceStates
    -> TokenRealization
         -> SiteRealizer
         -> PoseRealizer
         -> AffinityRealizer
```

主要代码位置：

- `src/models/proposed_model.py`：TRACEModel。
- `src/models/ar_trace_decoder.py`：AR trace decoder。
- `src/models/token_realization.py`：strict trace bottleneck、Site/Pose/Affinity realization。
- `src/models/local_patch_tokenizer.py`：VQ/local patch tokenizer。
- `src/data/trace_dataset.py`：task prefix、program token、trace token labels。
- `src/eval/neural_trace_experiment.py`：训练、评估、loss、audit、case export。

### 3.2 Strict Trace Bottleneck

主模型默认 `trace_bottleneck_mode=strict_trace`。严格模式下：

- AffinityRealizer 的有效 context 为 `trace_only`；
- direct pair residual 默认关闭；
- PoseRealizer 默认使用 event-aligned trace pose field；
- raw pair field 只能在 `direct_pair_ablation` 中启用；
- audit 会记录 direct shortcut 是否开启。

当前 affinity 主公式是：

```text
affinity =
    affinity_bin_expectation
  + affinity_residual
  + event_affinity_contribution
  + site_pose_affinity_contribution
  + local_patch_affinity_contribution
  + evidence_affinity_residual
  + affinity_pair_residual
```

其中 `affinity_pair_residual` 在 strict mode 下为 0，只在 direct-pair ablation 中启用。

设计原因：如果亲和力可以直接从 pair encoder 走捷径，模型可能绕过 trace，最终变成普通 shared encoder + head。strict bottleneck 的作用是强制主结果依赖 generated trace evidence。

## 4. Unified AR Trace Program

### 4.1 任务 token

模型通过 task prefix 自适应切换任务：

```text
<CTX_PDBBIND_REFINED> <TASK_AFFINITY> <TRACE_STRICT_ATOMMAP>
<CTX_PDBBIND_REFINED> <TASK_SITE>     <TRACE_STRICT_ATOMMAP>
<CTX_PDBBIND_REFINED> <TASK_DOCK>     <TRACE_STRICT_ATOMMAP>
<CTX_PDBBIND_REFINED> <TASK_REFINE>   <TRACE_STRICT_ATOMMAP>
<CTX_PDBBIND_REFINED> <TASK_TRACE>    <TRACE_STRICT_ATOMMAP>
```

同一个模型在不同 task token 下生成不同程序：

- `AFFINITY`：interaction events + affinity evidence + affinity bins/residuals。
- `SITE`：site proposal/anchor/expand/validate program + contact/site evidence。
- `DOCK`：interaction events + anchor sites + pose program + pose quality。
- `REFINE`：interaction events + anchor sites + pose program + clash fix + pose quality。
- `TRACE`：只学习 trace token generation。

设计原因：任务自适应不是换模型，而是换上下文 token 和生成 grammar。这样才符合统一生成模型的主线。

### 4.2 Canonical trace order

当前 trace generation 使用有限状态 grammar，约束生成顺序：

```text
context/task tokens
-> site program tokens
-> interaction event tokens
-> local patch tokens
-> pose program tokens
-> affinity evidence/bin/residual tokens
-> end
```

非法 token 在对应任务中被 mask，例如纯 `<TASK_SITE>` 不应生成无关 pose torsion 程序，`<TASK_DOCK>` 必须保留 pose program 段。

设计原因：AR 生成空间很大。如果不约束顺序，模型会学到大量无意义 token transition，导致下游 realizer 拿不到稳定程序信号。canonical order 和 constrained decoding 是让 AR trace 可训练、可解释、可执行的关键。

### 4.3 Interaction event tokens

标准 event token 细粒度包括：

```text
<EVENT>
<PROTEIN_SITE_*>
<LIGAND_SITE_*>
<INTERACTION_*>
<DIST_BIN_*>
<ANGLE_BIN_*>
<ORIENT_BIN_*>
<GEOM_CONF_*>
<AFF_EVIDENCE_*>
</EVENT>
```

当前支持 interaction type：

- contact；
- hydrogen bond；
- salt bridge；
- hydrophobic；
- pi-pi；
- cation-pi；
- halogen；
- metal coordination；
- steric clash；
- unsatisfied polar。

### 4.4 Site program tokens

为结合位点任务加入显式 site recognition program：

```text
<SITE_STEP_PROPOSE>
<SITE_STEP_ANCHOR>
<SITE_STEP_EXPAND>
<SITE_STEP_VALIDATE>
<BINDING_SITE_CONF_HIGH/MED/LOW>
<CONTACT_DENSITY_BIN_*>
<SITE_COVERAGE_BIN_*>
<ANCHOR_PROTEIN_SITE_k>
<ANCHOR_LIGAND_SITE_k>
```

设计原因：结合位点预测不是简单对每个 residue 输出 0/1。TRACE 需要先生成“提出 anchor、扩展周边接触、验证覆盖和密度”的程序，再由 SiteRealizer 执行。评价上仍可输出 protein binding-site binary/probability map，并报告 MCC/AUPRC/precision-at-k 等指标。

### 4.5 Pose program tokens

当前 pose/refinement 不是完整 DiffDock/FlowDock 式 SE(3) diffusion/flow 主生成器，而是 trace-centered local redocking/refinement program：

```text
<POSE_START>
<POSE_ANCHOR_PAIR_*>
<POSE_TRANSLATION_BIN_*>
<POSE_ROTATION_BIN_*>
<POSE_TORSION_BIN_*>
<POSE_CLASH_BIN_*>
<POSE_CONF_*>
<POSE_CANDIDATE_RANK_*>
<POSE_QUALITY_*>
<POSE_END>
```

DOCK 和 REFINE 的区别：

- `DOCK`：生成 interaction events、anchor sites、pose program、pose quality，用于从 perturbed ligand pose 生成候选 local pose。
- `REFINE`：在 DOCK 基础上强调 clash fix 和局部修正，用于从已有候选或 crystal-near pose 做 refinement。

设计原因：TRACE 的主创新是 AR interaction trace，不是复刻 SE(3) diffusion docking。当前 pose 模块补齐了 local perturb/refine 所需的 translation/rotation/torsion/clash/quality 程序，使 pose evidence 能闭环进入 affinity，而不让 docking 机制喧宾夺主。

### 4.6 Local patch VQ tokens

Pair interaction field 的局部几何 patch 会通过 VQ tokenizer 离散化成：

```text
<LOCAL_PATCH_CODE_k>
```

这些 token 可以进入 trace 语言，并通过 `local_patch_affinity_contribution` 影响 strict affinity。

设计原因：distance bin、angle bin 等人工 bins 表达力有限。VQ/local patch tokenizer 让模型把局部 interaction geometry 学成离散 token，补充 AR 语言的连续几何表达能力。

## 5. Realizers

### 5.1 SiteRealizer

SiteRealizer 接收 generated trace states 和 program masks，输出：

- pair contact logits；
- protein site logits；
- site_program_attention；
- protein_anchor_attention；
- ligand_anchor_attention；
- contact_density_logits；
- site_coverage_logits；
- top protein site evidence；
- top ligand site evidence。

Protein site prediction 当前主要输出是每个 protein site/residue-like site 的 binding-site probability/logit。训练和评价可按 ligand-aware binding site task 报告：

- AUPRC；
- AUROC；
- MCC；
- precision；
- recall；
- F1；
- precision@k。

设计原因：site logits 仍从 contact field 派生，但必须被 AR site program attention 调制，而不是简单 max pooling。这样 SITE 任务的生成程序能真正控制结合位点 executor。

### 5.2 PoseRealizer

PoseRealizer 接收 generated pose program states、anchor pair attention 和 trace-aligned pair field，输出：

- translation residual；
- rotation representation / rotation bin；
- torsion bin / torsion delta；
- candidate poses；
- clash correction；
- pose confidence；
- pose quality；
- RMSD、centroid distance、top-k success metric where available。

当前报告中已有：

- residual RMSE；
- torsion accuracy；
- rotamer accuracy；
- pose confidence accuracy；
- ligand RMSD；
- centroid distance；
- top1/top-k success@2A。

限制：这仍是 local redocking/refinement，不是完整 blind docking search，也不是 SE(3) diffusion/flow matching。没有把全局 pocket 搜索、多随机初始 pose 的长轨迹采样、equivariant score field 作为主生成器。

设计原因：让 pose 作为 trace evidence 的组成部分进入 affinity，同时保留可扩展到更强 pose generator 的接口。

### 5.3 AffinityRealizer

AffinityRealizer 只在 strict trace path 中汇聚以下 evidence：

- generated affinity bin expectation；
- generated affinity residual token；
- event affinity contribution；
- site/contact evidence；
- pose confidence evidence；
- local patch evidence；
- evidence affinity residual。

它不从 raw pair encoder 直接读 affinity，除非显式进入 `direct_pair_ablation`。

设计原因：亲和力预测要回答“为什么这个复合物强/弱结合”。因此输出分数必须能回溯到 generated events、contact density、site coverage、pose quality 和 local patch evidence。

## 6. 损失函数

当前训练不是单一 affinity MSE，而是 task-routed multi-objective training。

### 6.1 Trace generation losses

- `trace_token`：teacher token cross entropy。
- `trace_program_token`：对 site/pose/affinity/local-patch program token 加权。
- `trace_program_ce`：显式 program token CE。
- `generated_step_token`：free-running/generated token 的 step-level supervision。
- `generated_affinity_summary`：generated trace 中 affinity summary token 的监督。
- finite-state grammar constrained decoding：训练和评估保持合法 trace。

目的：让 AR decoder 生成可执行 trace，而不是只在 teacher-forced 状态下表现正常。

### 6.2 Generated downstream losses

- `generated_trace_realization`；
- `generated_contact`；
- `generated_site`；
- `generated_geometry`；
- `generated_pose`；
- `generated_affinity_all_task`；
- `generated_affinity_pairwise_rank`；
- `generated_affinity_spread`；
- `generated_evidence_affinity_residual`。

目的：把 generated trace 直接对下游 affinity/contact/site/pose 目标优化，减少 teacher-forced 训练和 generated evaluation 的 mismatch。主报告必须以 generated realization 为准，teacher-forced 只能作为诊断。

### 6.3 Affinity losses

- normalized affinity MSE；
- affinity bin CE；
- affinity residual MSE；
- all-task affinity auxiliary loss；
- pairwise ranking loss；
- prediction spread / dynamic range regularization；
- evidence affinity residual auxiliary loss。

目的：PDBbind/CASF 的亲和力不仅要均方误差低，还要排序合理、动态范围不过度塌缩，并且 generated affinity bin/residual 与连续 pKd/pKi/pIC50 语义对齐。

### 6.4 Site/contact losses

- contact BCE / focal-style weighted objective；
- protein site BCE；
- event-to-contact alignment loss；
- site executor anchor loss；
- contact density token loss；
- site coverage token loss。

目的：结合位点预测要通过 contact/site evidence 形成，而不是独立 residue classifier。

### 6.5 Pose/geometry losses

- ligand coordinate residual loss；
- translation / rotation loss；
- torsion bin loss；
- protein rotamer bin loss；
- pose confidence loss；
- clash/quality program loss；
- RMSD/top-k success metrics where coordinates are available。

目的：pose/refinement 不只是 proxy head，而是让 `<TASK_DOCK>` / `<TASK_REFINE>` 生成的 pose program 对候选结构有可评价输出。

### 6.6 Local patch VQ loss

- VQ commitment/codebook loss；
- local patch code perplexity；
- local patch affinity contribution supervision through downstream affinity losses。

目的：把局部 interaction geometry 离散成可生成 token，减少纯人工 distance/angle bin 的表达瓶颈。

## 7. 输出

模型可输出三类主结果。

### 7.1 Affinity

- predicted affinity；
- RMSE、MAE、Pearson、Spearman；
- affinity bin token accuracy；
- affinity residual token accuracy；
- prediction mean/std 和 target mean/std；
- generated trace evidence breakdown。

### 7.2 Binding site/contact

- pair contact logits；
- protein binding site logits；
- site MCC/AUPRC/AUROC/precision/recall/F1/precision@k；
- top protein site evidence；
- top ligand site evidence；
- contact density / site coverage tokens。

当前一版不把 ligand binding site 作为独立主指标，但保留 ligand anchor/evidence，供 trace 和 case study 使用。

### 7.3 Pose/refinement

- translation / rotation / torsion；
- candidate poses；
- pose confidence；
- clash score / pose quality；
- ligand RMSD；
- centroid distance；
- top1/top-k success@2A where available。

### 7.4 Case study

每个样本可导出：

- top generated events；
- protein site；
- ligand site；
- event type；
- distance/evidence；
- site/contact evidence；
- pose confidence；
- true affinity；
- predicted affinity。

目的：验证 generated trace 是否具有结构意义，而不是只看 affinity 数字。

## 8. Baselines 和公平性

当前公平比较保留：

- `no_trace`；
- `multi_head`；
- `no_trace_wide`；
- `multi_head_wide`。

公平性要求：

- 同一 split；
- 同一 site tensor；
- 同一 pretrained mode；
- 同一 optimizer/lr/epoch budget；
- 同一 affinity normalization；
- baseline 不减少数据、不削弱容量；
- wide baseline 是容量审计，不是故意压低 baseline。

区别只在结构：

- `no_trace`：site/pair encoder + direct affinity realization。
- `multi_head`：site/pair encoder + affinity/contact/event heads，无 AR trace generation。
- `trace`：site/pair encoder + causal AR trace decoder + trace-conditioned realizers。

## 9. 当前验证状态

最近严格 generated trace full-task 实验：

- 输出目录：`outputs/trace_supervised_20260601_fullprogram_programce_b24_gpu3_from32`
- 配置：formal site、atom_group ligand site、PLIP/Arpeggio fused trace、SaProt/UniMol provider、strict trace bottleneck、generated realization、local patch VQ。
- CASF test：RMSE `1.7344`，MAE `1.3700`，Pearson `0.6078`，Spearman `0.6118`。
- prediction std / true std ratio：`0.5260`，说明 affinity 动态范围仍不足。
- Site protein AUPRC：`0.1809`。
- Site MCC：`0.0135`。
- Contact AUPRC：`0.0623`。
- Free-running token accuracy：`0.6459`。
- Interaction token accuracy：`0.0264`。
- Program token accuracy 多项仍接近 `0.0000`，说明程序 token 还没有学成稳定可解释的生成过程。

Top-k checkpoint 诊断中，近期最好 CASF RMSE 约为 `1.7271`。后续 evidence residual 训练让 validation dynamic range 有所改善，但 CASF 仍约 `1.73`，没有达到用户希望的 `~1.5` 或更强目标。

最新 targeted verification：

```text
151 passed in 5.22s
```

该测试覆盖 strict bottleneck、generated realization、pose program、local patch、pretrained provider、evidence residual 等核心路径。

## 10. 创新点

### 10.1 Trace-centered multi-task generation

TRACE 把亲和力、结合位点、contact 和 pose/refinement 统一为 AR interaction trace program，而不是普通多头多任务。

### 10.2 Strict trace bottleneck

主结果强制经过 generated trace states 和 event evidence，direct pair shortcut 只作为 ablation。这样能审计模型是否真的使用 trace。

### 10.3 PLIP/Arpeggio fused atommap supervision

标签不再主要依赖手工距离近邻，而是通过 PLIP/Arpeggio strict atom/site identity 构造，并记录 exact/coordinate/fallback 覆盖率。

### 10.4 Formal interaction-site abstraction

Protein 和 ligand 都先转为 interaction-site tensor，保留 atom identity、group-level pharmacophore、local geometry、rotamer/conformer context。

### 10.5 Program-token controlled executors

SITE/POSE/AFFINITY/local-patch token 不是装饰性 vocabulary，而是转成 program masks 控制对应 realizer。

### 10.6 Generated trace downstream self-optimization

训练中引入 generated downstream losses，使 free-running trace 也要对下游 affinity/contact/site/pose 负责。

### 10.7 Local patch VQ trace tokens

把局部 interaction geometry 学成离散 code，增强 trace token 对局部几何的表达力。

## 11. 当前不足

### 11.1 性能还未证明最终有效

当前 CASF RMSE 约 `1.73`，与目标 `~1.5` 仍有差距。不能宣称最终成功。

### 11.2 Program token 生成质量不足

多个 site/pose/affinity program token accuracy 仍接近 0。说明 AR program 还没有稳定学会细粒度任务程序。

### 11.3 Site/contact 信号弱

Site MCC 和 contact AUPRC 仍偏低。虽然架构链路有 site executor，但过程监督尚未充分转化为有效结合位点预测。

### 11.4 Pose 仍是 local redocking/refinement

当前 pose 模块可以从 perturbed/crystal-near pose 做 local refinement 和候选评价，但不是完整 global blind docking，不包含 DiffDock/FlowDock 式 SE(3) diffusion/flow 主生成器。

### 11.5 Pretrained 3Di coverage 需要持续审计

SaProt-like provider 已接入，但必须持续报告 3Di token 覆盖和 fallback，避免 structure-aware 特征名义化。

## 12. 后续执行标准

后续所有训练和架构修改必须遵守：

1. CASF 只作为最终 held-out test，不能用于调参或校准。
2. 主结果使用 generated strict trace realization，teacher-forced 只作为诊断。
3. Direct pair shortcut 默认关闭，只能作为 ablation。
4. Baseline 使用同一数据、site tensor、pretrained mode、checkpoint protocol 和训练预算。
5. 每次实验输出 fairness audit、architecture audit、contact label source summary、case studies 和 metrics。
6. 如果 TRACE 没有优于 fair baselines，报告为 failure analysis，不宣称成功。
7. 架构优化优先解决 generated trace program、site/contact evidence、affinity dynamic range 和 core-like validation domain shift，而不是针对 CASF 做后处理拟合。

## 13. 下一阶段优先级

为了达到 md 文件的最终目标，下一阶段建议按以下顺序推进：

1. 诊断高误差样本：比较 generated trace、site/contact evidence、pose evidence 和 affinity residual 是否一致。
2. 强化 program token learning：提高 site/pose/affinity program token 的可学性，减少无效 token。
3. 强化 SiteRealizer：让 site/contact evidence 对 affinity 产生可解释且稳定的帮助。
4. 强化 generated affinity calibration：只用 train/refined-val/core-like-val，冻结后再测 CASF。
5. 扩展到 PDBbind general set 或更多结构数据：先解决 refined-only 数据规模和分布不足的问题，再谈最终 SOTA。
6. 若 docking 成为主任务目标，再考虑引入轻量 equivariant pose dynamics；但不能让 SE(3) diffusion/flow 取代 TRACE 的 AR trace innovation。
