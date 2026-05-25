# TRACE 当前模型结构设计与进展说明

更新时间：2026-05-25  
项目代号：TRACE  
当前阶段：标准版主架构已基本搭建，仍处于性能诊断和训练策略修正阶段

## 1. 总体判断

TRACE 现在已经不是一个普通的 shared encoder + 多任务 head 原型。当前代码里的主线已经明确组织为：

```text
Protein/Ligand Interaction Sites
-> Pair Interaction Field
-> Unified AR Trace Program
-> SiteRealizer
-> PoseRealizer
-> AffinityRealizer
```

这条主线对应 md 文件里的核心想法：先让模型生成或消费一条可解释的 protein-ligand interaction trace，再由 trace states 驱动 affinity、binding site、contact、pose/refine evidence。也就是说，TRACE 的核心创新不是多加几个预测头，而是把多个任务都绑定到同一条自回归 interaction program 上。

当前已经实现的架构覆盖：

- PLIP + Arpeggio fused atommap trace pipeline。
- formal protein/ligand interaction-site representation。
- task/context routed AR trace decoder。
- finite-state constrained decoding。
- SITE program tokens。
- DOCK/REFINE pose program tokens。
- strict trace bottleneck。
- generated trace realization。
- candidate trace generation and reranking。
- SaProt/Uni-Mol/MolFormer-style pretrained provider schema。
- local patch VQ tokenizer 接口。
- no_trace、multi_head、wide baseline 公平对照入口。

当前还没有完成的是最终性能收敛验证。最近的诊断显示，contact/site/pose 指标能持续学习，但 affinity 没有稳定吃到这些过程监督带来的收益。因此最新一次结构修正集中在 AffinityRealizer 的 evidence pooling，避免强 contact/site 证据被大量负 pair 的均值稀释。

## 2. 目标任务和数据设定

第一阶段仍然按 md 文件要求使用 PDBbind refined / CASF-2016：

| Split | 当前用途 | 说明 |
|---|---|---|
| train | PDBbind refined minus CASF/core | 用于训练，训练时会按 task variants 展开 |
| validation | refined 内部划分 | 用于 checkpoint selection |
| core-like validation | refined train 内构造 | 用于更接近 CASF 分布的验证信号 |
| test | CASF-2016 core set | 第一阶段 held-out test |

当前标准全任务实验中的数据规模：

| Split | 数量 |
|---|---:|
| train | 22725 |
| validation | 505 |
| test CASF-2016 | 285 |

这里的 `train=22725` 是 task expansion 后的样本数，来自 `AFFINITY,SITE,TRACE,DOCK,REFINE` 五类任务展开。它不是说有 22725 个唯一 complex。

当前 trace label 覆盖率已经达到严格 atommap 级别为主：

| Split | exact fraction | coordinate fraction | distance fallback fraction |
|---|---:|---:|---:|
| train | 0.9485 | 0.0039 | 0.0476 |
| validation | 0.9484 | 0.0044 | 0.0472 |
| CASF test | 0.9543 | 0.0220 | 0.0238 |

这个结果说明 label pipeline 已经不是主要依赖纯距离近邻 fallback，而是大部分由 exact atom/site map 支撑。

## 3. 当前输入是什么

模型的输入不是单纯蛋白序列或 ligand SMILES，而是结构化后的 complex representation：

```text
protein formal site tensor
ligand formal site tensor
optional protein pretrained site embedding
optional ligand pretrained site embedding
task/context routed trace token ids
ligand initial/reference coordinates
task loss mask
```

### 3.1 Protein site input

protein side 使用 formal interaction-site 表示。每个 site 包含：

- residue identity。
- chain/residue index。
- atom ids / sidechain atom ids。
- functional group，例如 carboxylate、amine、guanidinium、imidazole、hydroxyl、aromatic、amide 等。
- role，例如 donor、acceptor、cationic、anionic、hydrophobic、aromatic。
- charge。
- 3D position。
- backbone local frame。
- sidechain functional center。
- chi angle / rotamer coarse state。
- geometry source 和 fallback 标记。

逻辑意义：binding 不只由 residue 名字决定，还与 sidechain functional atom、局部方向、rotamer 几何有关。formal site 让模型能在 interaction-site 层看到更接近药物设计任务的局部物理化学语义。

### 3.2 Ligand site input

ligand side 当前支持 `atom`、`group`、`atom_group`，标准版默认使用 `atom_group`。每个 ligand site 包含：

- atom index / atom identity。
- member atom indices。
- aromatic ring centroid / normal / radius。
- hydrophobic patch。
- donor / acceptor group。
- positive ionizable / negative ionizable group。
- halogen site。
- zinc binder / metal coordination proxy。
- rotatable bond context。
- conformer/local chemical environment。
- pharmacophore typing。

逻辑意义：配体侧如果只看单个 atom，会丢失 aromatic ring、charged group、hydrophobic patch 这些真正参与相互作用的 group-level site。当前实现通过 RDKit ChemicalFeatures 和 graph/ring 信息，把 atom-level 和 group-level site 同时保留。

### 3.3 Pretrained representation input

当前 provider schema 支持：

| Provider | Protein | Ligand | 当前定位 |
|---|---|---|---|
| `esm2_chemberta` | ESM2 | ChemBERTa | 稳定 baseline provider |
| `saprot_unimol` | SaProt | Uni-Mol | md 标准 provider |
| `esm2_unimol` | ESM2 | Uni-Mol | fallback provider |
| `saprot_unimol_molformer` | SaProt | Uni-Mol + MolFormer | 扩展 provider |

`pretrained_mode` 支持：

```text
none
protein_only
ligand_only
protein_ligand
```

逻辑意义：预训练特征不能只给 TRACE 或只给 baseline，否则公平性有问题。当前 provider schema 会把同一套 pretrained cache 接入 dataset 和 model，并让 TRACE 与 baseline 在同一输入策略下比较。

当前 SaProt 接入支持 Foldseek 3Di 路径。如果 3Di 不可用，会进入 masked structure token fallback；标准目标仍是 `AA + 3Di` structure-aware input。

## 4. Trace token schema

TRACE 的核心中间语言是 interaction event token sequence。标准事件包含：

```text
protein site
ligand site
interaction type
distance bin
angle bin
orientation bin
geometry confidence
affinity evidence
affinity bin / residual summary token
atom/site identity fields
```

示例：

```text
<CTX_PDBBIND_REFINED>
<TASK_AFFINITY>
<TRACE_STRICT_ATOMMAP>
<EVENT>
<RES_ASP_A45>
<LIG_ATOM_7>
<INTERACTION_SALT_BRIDGE>
<DIST_2.5_3.0>
<ANGLE_BIN_UNKNOWN>
<ORIENT_BIN_UNKNOWN>
<GEOM_CONF_HIGH>
<AFF_EVIDENCE_POS_STRONG>
</EVENT>
<AFF_BIN_...>
<AFF_RESIDUAL_...>
<EOS>
```

逻辑意义：affinity 不再只是一个回归标签，而是被拆成一组 interaction event evidence，再由 AR decoder 学习事件序列和证据组合。`AFF_BIN` 和 `AFF_RESIDUAL` 让亲和力既有离散生成式表达，也保留连续回归精度。

## 5. Task program 设计

当前 task 由 route token 控制：

| Task | 输入 token | 主要目标 |
|---|---|---|
| TRACE | `<TASK_TRACE>` | 学 interaction trace 生成 |
| AFFINITY | `<TASK_AFFINITY>` | 从 trace evidence 预测 affinity |
| SITE | `<TASK_SITE>` | 生成 site program 并实现 protein binding-site evidence |
| DOCK | `<TASK_DOCK>` | site/contact/pose program + local docking/refine |
| REFINE | `<TASK_REFINE>` | 在 DOCK 基础上强调 clash fix 和 pose quality |

### 5.1 SITE program

SITE 任务不是直接输出“某 residue 是不是 binding site”，而是生成一个 binding-site identification program：

```text
<SITE_STEP_PROPOSE>
<SITE_STEP_ANCHOR>
<ANCHOR_PROTEIN_SITE_k>
<ANCHOR_LIGAND_SITE_k>
<BINDING_SITE_CONF_HIGH/MED/LOW>
<CONTACT_DENSITY_BIN_*>
<SITE_STEP_EXPAND>
<SITE_COVERAGE_BIN_*>
<SITE_STEP_VALIDATE>
```

逻辑意义：这更接近 ligand-aware binding site identification。模型先提出 anchor，再扩展 contact neighborhood，最后验证 site confidence。最终 `protein_site_logits` 仍然是 per protein site 的 binding-site evidence，但它由 AR site program attention 调制，而不是简单 max pooling。

当前 SiteRealizer 输出：

- `site_program_attention`
- `protein_anchor_attention`
- `ligand_anchor_attention`
- `contact_density_logits`
- `site_coverage_logits`
- `protein_site_logits`
- `contact_logits`

### 5.2 DOCK / REFINE pose program

DOCK/REFINE 任务包含 pose program token：

```text
<POSE_START>
<POSE_STEP_COARSE>
<POSE_STEP_ANCHOR>
<ANCHOR_PROTEIN_SITE_k>
<ANCHOR_LIGAND_SITE_k>
<CONTACT_PAIR_CONF_HIGH/MED/LOW>
<POSE_STEP_REFINE>
<POSE_TRANS_DIR_X_*>
<POSE_TRANS_DIR_Y_*>
<POSE_TRANS_DIR_Z_*>
<POSE_TRANS_MAG_BIN_*>
<POSE_ROT_ANGLE_BIN_*>
<POSE_ROT_AXIS_X_*>
<POSE_ROT_AXIS_Y_*>
<POSE_ROT_AXIS_Z_*>
<POSE_TORSION_DELTA_BIN_*>
<POSE_CANDIDATE_RANK_*>
<POSE_STEP_CLASH_FIX>
<CLASH_BIN_*>
<POSE_RMSD_PROXY_*>
<POSE_QUALITY_*>
<POSE_CONF_*>
<POSE_END>
```

逻辑意义：TRACE 不把 pose 当作一个外接 regression head，而是要求 AR trace 里显式出现 pose program，再由 PoseRealizer 读取这些 program states 执行 translation、rotation、torsion、candidate confidence、clash correction 和 pose quality 输出。

当前这仍是 local redocking/refinement，不是 DiffDock/FlowDock 那种完整 SE(3) diffusion / flow matching docking generator。它能输出 pose candidates 和 reconstructed ligand coordinates，但还没有实现 full blind docking 的全局搜索、随机轨迹去噪、多步 SE(3) score/flow dynamics。

## 6. 主模型结构

### 6.1 PretrainedFeatureAdapter

位置：`src/models/pretrained_feature_adapter.py`

功能：

```text
site_features + gate(site_features, pretrained_features) * projection(pretrained_features)
```

逻辑意义：

- 保留 formal site tensor 作为稳定主输入。
- pretrained embedding 作为可门控增强。
- cache 缺失或 ablation 时不会破坏主模型结构。

### 6.2 SiteEncoder

位置：`src/models/site_encoder.py`

protein 和 ligand 使用独立 encoder：

```text
protein formal sites -> protein_site_encoder
ligand formal sites  -> ligand_site_encoder
```

逻辑意义：protein residue site 与 ligand atom/group site 的分布不同，分开编码能避免强行共享一套输入统计。

### 6.3 PairInteractionEncoder

位置：`src/models/pair_interaction_encoder.py`

构造 pair interaction field：

```text
pair(i,j) = MLP([protein_i, ligand_j, protein_i * ligand_j])
```

输出：

```text
[batch, protein_sites, ligand_sites, hidden_dim]
```

逻辑意义：所有 interaction event 都发生在 protein site 与 ligand site 的 pair field 上。AR decoder 的 memory 和 realizer 的 site/contact/pose evidence 都从这个 pair field 来。

### 6.4 ARTraceDecoder

位置：`src/models/ar_trace_decoder.py`

结构：Transformer decoder with causal mask。

输入：

```text
trace token ids + flattened pair memory
```

输出：

```text
token_logits
trace_states
```

逻辑意义：AR decoder 是 TRACE 的生成核心。它不是简单把 trace 当标签分类，而是产生每个 token position 的 hidden state，这些 state 后续被 SiteRealizer、PoseRealizer、AffinityRealizer 使用。

### 6.5 TokenRealization

位置：`src/models/token_realization.py`

当前 TokenRealization 负责三个 executor：

```text
trace_states + pair_memory
-> SiteRealizer outputs
-> PoseRealizer outputs
-> AffinityRealizer outputs
```

主要输出：

| 输出 | 含义 |
|---|---|
| `affinity` | z-score 空间 affinity prediction |
| `affinity_bin_logits` | affinity 离散 bin |
| `affinity_residual` | bin 内连续残差 |
| `event_affinity_contribution` | event-level affinity evidence 聚合 |
| `site_pose_affinity_contribution` | site/contact/pose evidence 对 affinity 的贡献 |
| `contact_logits` | pair contact prediction |
| `protein_site_logits` | ligand-aware protein binding site prediction |
| `distance_logits` | pair distance bin |
| `orientation_logits` | orientation bin |
| `event_confidence_logits` | event geometry confidence |
| `ligand_pose_candidate_translations` | 多候选平移 |
| `ligand_pose_candidate_quaternions` | 多候选旋转 |
| `ligand_torsion_delta_candidates` | 多候选 torsion delta |
| `pose_candidate_confidence` | pose candidate confidence |
| `ligand_candidate_coords` | reconstructed candidate ligand coordinates |

## 7. Strict trace bottleneck

当前默认：

```text
trace_bottleneck_mode = strict_trace
trace_effective_affinity_context_mode = trace_only
```

strict trace 下 affinity 由以下部分组成：

```text
affinity =
    affinity_bin_expectation
  + affinity_residual
  + event_affinity_contribution
  + site_pose_affinity_contribution
```

其中：

```text
affinity_pair_residual = 0
```

只有在：

```text
trace_bottleneck_mode = direct_pair_ablation
```

时才允许 direct pair residual。

逻辑意义：这是为了避免模型绕过 AR trace，直接从 pair encoder 读出 affinity。如果 direct_pair_ablation 明显更好，而 strict_trace 不好，就说明瓶颈过窄或 realizer 不够强，而不是证明 trace 主线有效。

## 8. AffinityRealizer 当前最新修正

最近一次诊断发现：

- contact MCC、site MCC、pose local refine 指标持续改善。
- affinity validation 在早期达到较好值后变差。
- 原 `site_pose_affinity_contribution` 使用均值汇聚，强 contact/site 证据容易被大量负 pair 稀释。

因此当前已改为 gated sparse evidence pooling：

```text
site_pose_affinity_evidence =
[
  top-k sparse contact probability,
  protein-anchor-weighted site probability,
  sparse pose high-confidence probability,
  contact density expectation,
  site coverage expectation,
  site peak probability
]

evidence = evidence * gate(trace_summary)
site_pose_affinity_contribution = MLP([trace_summary, evidence])
```

逻辑意义：

- contact/site 通常是稀疏信号，不能只做全局平均。
- anchor site 是 SITE program 的核心，必须显式进入 affinity 汇聚。
- pose confidence 只在局部候选上强，不应该被全 pair 平均稀释。
- density/coverage token 让 AR program 生成的 site-level summary 参与 affinity。

这个修正保持 strict trace 主线，不打开 direct pair shortcut。

## 9. Generated trace path

当前支持三种相关路径：

### 9.1 Teacher-forced realization

输入 gold trace token：

```text
gold trace ids -> trace_states -> realizers
```

用途：验证“如果 trace 正确，realizer 是否能用它做 affinity/contact/site/pose”。

### 9.2 Free-running generated realization

先生成 trace，再实现下游输出：

```text
start/context/task token
-> AR generate trace
-> generated trace_states
-> realizers
```

用途：验证“模型自己生成的 trace 是否能支持下游任务”。

### 9.3 Differentiable generated realization

训练中使用 soft token probabilities：

```text
soft generated token probabilities
-> decoder hidden states
-> generated affinity/contact/site/pose loss
```

用途：让 generated trace 本身为下游目标自优化，而不是只靠 teacher-forced trace CE。

当前诊断显示 generated/scheduled sampling 不能过早过强，否则会干扰 affinity 收敛。后续需要 curriculum，而不是直接把所有 generated loss 拉满。

## 10. Constrained decoding 和 reranking

位置：`src/trace/generation_constraints.py`

当前实现 finite-state grammar：

- AFFINITY 任务禁止 SITE/POSE program token。
- SITE 任务必须按 propose -> anchor -> expand -> validate。
- DOCK/REFINE 必须先完成 SITE program，再进入 POSE program。
- 禁止 `<PAD>`、`<UNK>`、route token 被自由生成。

逻辑意义：AR 生成空间很大，如果没有 grammar，模型会生成语法上无意义的 token。finite-state grammar 相当于 TRACE 版本的生成过程约束，对应 diffusion/flow 里的 sampling dynamics 约束。

候选生成支持 beam / multi-candidate，并通过 consistency reranking 选择：

```text
trace logprob
+ contact evidence
+ site evidence
+ pose confidence
+ contact-distance consistency
+ pocket containment
- clash penalty
```

逻辑意义：不只选 token probability 高的 trace，也选结构上更一致的 trace。

## 11. PoseRealizer 当前程度

当前 PoseRealizer 已实现：

- pose program attention。
- translation context。
- rotation context。
- torsion context。
- clash context。
- quality context。
- anchor pair attention。
- multi-candidate translation / quaternion / torsion。
- top candidate selection。
- reconstructed ligand candidate coordinates。
- local RMSD / centroid distance / top-k success evaluation。

当前 PoseRealizer 没有实现：

- SE(3) diffusion / flow matching 主生成器。
- 多步随机去噪轨迹。
- 全局 pocket search。
- 完整 flexible docking conformer search。
- PoseBusters 化学合理性评分，当前环境未安装 PoseBusters。

结论：当前 pose 是 local redocking/refinement executor，不应在论文或报告中声称是完整 blind docking SOTA。但它已经符合 TRACE 主线中的“AR pose program -> PoseRealizer -> pose evidence”的阶段性正式架构。

## 12. Binding site 当前输出

当前 binding-site 输出是：

```text
protein_site_logits: [batch, protein_sites]
```

它表示每个 protein interaction site 是 ligand-aware binding site 的概率/logit。

同时还有：

- `contact_logits: [batch, protein_sites, ligand_sites]`
- `site_program_attention`
- `protein_anchor_attention`
- `ligand_anchor_attention`
- `contact_density_logits`
- `site_coverage_logits`

逻辑意义：最终 site 预测不是只输出 0/1，而是可以解释为：

```text
AR site program 生成 anchor / density / coverage
-> contact field 给出 protein-ligand pair evidence
-> SiteRealizer 汇聚成 protein binding-site logits
```

评估时可以计算 protein-site MCC、F1、AUPRC、precision@k。当前第一版不预测 ligand binding site 作为独立主指标，但 ligand anchor attention 和 pair contact 已经保留 ligand-side evidence。

## 13. LocalPatchTokenizer / VQ geometry token

位置：`src/models/local_patch_tokenizer.py`

当前实现：

```text
pair_memory -> VQ codebook -> local_patch_code_ids
```

输出：

- `local_patch_code_logits`
- `local_patch_code_ids`
- `local_patch_quantized_memory`
- `local_patch_vq_loss`
- `local_patch_perplexity`

逻辑意义：这是把局部相互作用几何学习成离散 token 的接口。当前它是架构扩展点，已经接入但不是性能主线。后续如果要进一步提高 AR trace 的生成表达力，可以把 VQ local patch codes 纳入 trace token schema。

## 14. Baseline 和公平性

当前 baseline：

| Baseline | 说明 |
|---|---|
| `no_trace` | site/pair encoder + affinity head，不生成 trace |
| `multi_head` | site/pair encoder + affinity/contact/site/geometry heads |
| `no_trace_wide` | 更大容量 no_trace |
| `multi_head_wide` | 更大容量 multi_head |

公平性原则：

- baseline 和 TRACE 使用相同 split。
- 使用相同 formal site tensor。
- 使用相同 pretrained mode。
- 使用相同 train/val/test 数据。
- 不恶意压低 baseline。
- direct pair path 只作为 ablation，不作为 strict TRACE 默认主路径。

## 15. 当前实验结果摘要

当前最完整一次标准全任务实验：

```text
outputs/trace_standard_fulltask_gpu5_30epoch_eventalign_pairfusion_capacity
```

配置要点：

| 项 | 值 |
|---|---|
| epochs | 30 |
| trace pretrain epochs | 4 |
| site feature | formal |
| ligand site mode | atom_group |
| pretrained provider | saprot_unimol |
| pretrained mode | protein_ligand |
| trace backend | plip_arpeggio_fused |
| bottleneck | strict_trace |
| realization source | both |
| validation realization | teacher_forced |
| test realization | both |
| trace candidates | 4 |
| pose candidates | 4 |

CASF-2016 test affinity：

| Model | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| TRACE teacher-forced | 1.724 | 1.379 | 0.623 | 0.626 |
| TRACE generated | 1.786 | 1.445 | 0.570 | 0.579 |
| multi_head | 1.878 | 1.490 | 0.521 | 0.508 |
| no_trace | 2.530 | 2.017 | 0.389 | 0.357 |
| no_trace_wide | 2.053 | 1.641 | 0.563 | 0.559 |
| multi_head_wide | 3.506 | 2.979 | 0.396 | 0.431 |

这个结果说明 TRACE 在该 run 中优于 no_trace 和 multi_head，但还没有达到预期的 SOTA 级别目标。后续在 AffinityRealizer 修正后必须重跑，不能把旧结果作为最终结论。

CASF-2016 contact/site：

| Model | Contact MCC | Contact AUPRC | Site MCC | Site AUPRC |
|---|---:|---:|---:|---:|
| TRACE teacher-forced | 0.092 | 0.140 | 未在摘要中完整截取 | 未在摘要中完整截取 |
| TRACE generated | 0.091 | 0.159 | 待完整导出 | 待完整导出 |
| multi_head | 0.106 | 0.161 | 0.087 | 0.382 |

诊断含义：contact/site 还不是绝对强，且 multi_head 在 contact 上不弱。TRACE 的优势主要来自 affinity 相关性和整体生成解释路径，site/contact 仍需要继续优化。

Pose/refine：

| 指标 | 当前状态 |
|---|---|
| RMSD available | true |
| top-k success 2A | 1.0 |
| PoseBusters | unavailable |
| 任务性质 | crystal/reference local refine，不是完整 docking search |

这个 pose 数字不能解释成 blind docking SOTA。它说明 local pose reconstruction/refine path 可跑通。

## 16. 最近一次失败/中断诊断

`outputs/trace_phase2a_refined_affaux_coreval_topk_gpu5_30epoch`：

| Model | best val/core-like | CASF test |
|---|---:|---:|
| no_trace | checkpoint score 1.312 | RMSE 2.134 |
| multi_head | checkpoint score 1.309 | RMSE 1.897 |
| TRACE | checkpoint score 1.331 | 未完成 |

`outputs/trace_affinity_dominant` 在中断前：

| 指标 | 值 |
|---|---:|
| best epoch | 8 |
| best val RMSE | 1.330 |
| best core-like val RMSE | 1.311 |
| final CASF test | 未完成 |

关键诊断：

- 关闭 generated/scheduled 后，early validation 更稳定。
- site/contact/pose 指标继续改善。
- affinity 没有稳定从过程监督中获益。
- 因此先修 AffinityRealizer 的 evidence pooling，而不是继续重复同样训练。

## 17. 当前距离最终版的差距

### 17.1 已达到标准版主线的部分

- `interaction sites -> pair field -> AR trace -> realizers` 主线已经成立。
- strict trace bottleneck 默认开启。
- direct pair shortcut 默认关闭。
- SITE program / POSE program token 已经进入生成语法。
- PLIP+Arpeggio fused atommap trace 已经接入。
- formal protein/ligand site tensor 已经接入。
- pretrained provider schema 已经接入。
- teacher-forced 和 generated realization 都已经接入。
- candidate reranking 已经接入。
- case study / audit / metrics 输出已经接入。

### 17.2 仍未达到最终论文级性能验证的部分

- AffinityRealizer 修正后还没有重新完整跑 CASF。
- generated trace curriculum 仍需调参，不能过早引入强 generated loss。
- contact/site 指标还需提升，尤其是 TRACE contact 不能明显弱于 multi_head。
- Pose 仍是 local refine，不是完整 SOTA docking generator。
- SaProt 需要确保 Foldseek 3Di 全量可用，避免 fallback 到 masked structure token。
- VQ/local patch code 尚未作为主 trace token 使用。
- 尚未引入 general set / CrossDocked / DockGen 等 Phase-2 大规模数据。

## 18. 下一步推荐执行顺序

### Step 1: 重跑修正后的 affinity-dominant TRACE

目的：验证新的 gated sparse evidence pooling 是否让 site/contact/pose evidence 真正帮助 affinity。

判据：

- CASF RMSE 必须优于 1.878 的 multi_head。
- 目标先看是否进入 1.5-1.7 区间。
- 若不能提升，检查 evidence contribution distribution。

### Step 2: direct_pair_ablation

目的：判断 strict trace bottleneck 是否过窄。

判据：

- 如果 direct_pair_ablation 明显提升，说明 pair field 有信息但 strict realizer 没用好。
- 如果 direct_pair 也不提升，说明数据分布、split、label 或整体训练协议有问题。

### Step 3: generated curriculum

目的：让 generated trace 真正服务下游，而不是扰乱 affinity。

策略：

- delayed scheduled sampling。
- 先 teacher-forced 收敛，再逐渐加 generated contact/site/affinity loss。
- 分别报告 teacher-forced 和 generated metrics。

### Step 4: site/contact 专项增强

目的：让 binding site/contact 成为有效过程监督。

方向：

- threshold calibration。
- positive/negative balance。
- protein anchor attention 与 contact label 对齐。
- case study 导出 top protein site evidence。

### Step 5: pose 保持轻量，不喧宾夺主

目的：保留 TRACE 解释链路，不把项目变成 DiffDock 复刻。

当前适合继续做：

- multi-candidate local refinement。
- clash/contact/pocket consistency rerank。
- pose confidence calibration。

暂不建议马上做：

- 完整 SE(3) diffusion。
- full blind docking global search。
- 大规模 conformer generation engine。

## 19. 面向后续版本共享的文件建议

建议在 GitHub 仓库中保留以下路径：

```text
docs/TRACE_current_architecture_status_2026-05-25.md
```

后续每次关键结构或实验结论变化，可以新增：

```text
docs/TRACE_current_architecture_status_YYYY-MM-DD.md
```

或者维护一个滚动版：

```text
docs/TRACE_current_architecture_status.md
```

当前这份文档的定位是“截至 2026-05-25 的模型结构与进展快照”，不是最终论文结果。

## 20. 简短结论

TRACE 当前已经完成了标准版主架构的大部分关键设计：AR interaction trace 是中心对象，SITE/DOCK/AFFINITY 都围绕 trace program 和 realizer 执行；它不是普通多头模型。

但目前还不能宣称最终算法完全有效。最完整 run 中 TRACE 的 CASF affinity 已经优于 no_trace 和 multi_head，但距离预期 SOTA 级目标还有差距。最近诊断显示主要卡点在 affinity 如何吸收 site/contact/pose 过程证据，因此已将 AffinityRealizer 从简单均值 evidence pooling 改为 gated sparse evidence pooling。下一步必须重跑 full CASF test，确认该结构修正是否带来真实泛化提升。

