# TRACE 统一三任务生成模型要求与训练策略

更新时间：2026-06-02

本文档记录本轮讨论后对 TRACE 的最新收敛要求。它不是替代 `TRACE_CURRENT_ARCHITECTURE.md` 的完整架构说明，而是给下一版实现提供更窄、更可训练的目标边界。

当前结论：下一阶段不应继续把 full TRACE 作为第一目标；应先实现一个轻量的三任务统一生成模型，让同一模型既能做理解任务，也能做生成任务。可解释 trace 应先作为辅助监督和诊断信号，而不是一开始就成为所有任务的 strict bottleneck。

---

## 1. 最新目标

目标从“完整 trace-centered mechanistic program system”收敛为：

```text
一个统一三任务的结构生成模型：
protein / ligand structure input
-> shared structural tokenization / interaction representation
-> unified autoregressive decoder
-> task-conditioned generation for understanding and generation tasks
```

第一版目标不是证明所有预测都必须经过一条完整可解释 interaction program，而是先证明：

1. 同一模型可以通过 task token 切换任务。
2. 同一结构输入可以支持理解任务和生成任务。
3. 三个任务可以共享主干，而不是各自独立 head 或独立模型。
4. 生成式输出在 teacher-forced 与 free-running / generated setting 下都可评估。
5. 架构复杂度必须和当前数据规模、训练稳定性匹配。

建议代号：

```text
TRACE-Lite
UniTrace-3T
```

---

## 2. 与当前 full TRACE 的关系

当前 full TRACE 主线大致为：

```text
formal protein / ligand interaction sites
-> pair interaction field
-> finite-state AR trace program
-> token-conditioned SiteRealizer / PoseRealizer / AffinityRealizer
-> affinity / site / contact / pose outputs
```

这个方向有研究价值，但当前对下一阶段目标过重。它的问题不是单个模块没有道理，而是：

1. 中间程序链太长。
2. 训练目标太多。
3. generated trace 尚未稳定时过早启用 strict bottleneck。
4. site/contact/pose/affinity/VQ/program token 同时训练，失败后难以定位。
5. 当前目标只是统一三任务生成，不需要一开始证明完整可解释程序链。

因此下一版应从 full TRACE 降级为更小闭环：

```text
interaction sites
-> compact interaction tokens
-> unified AR decoder
-> three task-specific generated outputs
```

---

## 3. 方法论原则

下一版 TRACE 应更接近统一生成模型方法论，而不是专家系统式全链路程序执行。

核心原则：

```text
1. 先统一 task interface，不要先统一完整解释链。
2. 先做 shared decoder，不要先做多个 heavy realizer。
3. 先允许 direct path / auxiliary heads，不要一开始 strict bottleneck。
4. 先生成最终任务输出，不要生成过长中间程序。
5. 先证明三任务最小闭环，再逐步加 trace/event/pose/VQ 解释模块。
```

不要把“最终论文故事里希望出现的所有组件”都放进第一版主架构。第一版要以可训练、可诊断、可 ablation 为优先。

---

## 4. 建议保留、降级、暂缓的组件

### 4.1 建议保留

保留当前 TRACE 中真正有价值的结构先验：

```text
formal protein site representation
ligand atom_group representation
protein-ligand pair interaction features
task token / context token
generated vs teacher-forced evaluation split
fair baseline / direct path audit
```

这些组件服务于统一结构输入，不应删除。

### 4.2 建议降级

以下组件不应作为第一版硬瓶颈：

```text
PLIP / Arpeggio fused trace labels
finite-state grammar
trace event tokens
site/contact program tokens
pose program tokens
```

它们可以作为 auxiliary supervision、diagnostic target 或轻量 constrained decoding 使用，但不要强制所有任务输出都必须经过它们。

### 4.3 第一版暂缓

以下组件第一版建议先关闭或只做 ablation：

```text
strict trace bottleneck
full SiteRealizer
full PoseRealizer
full AffinityRealizer
local patch VQ tokenizer
candidate trace reranking
full generated downstream loss stack
direct pair shortcut 默认禁用策略
```

这些组件只有在三任务最小闭环稳定后再逐步加入。

---

## 5. 建议架构：TRACE-Lite / UniTrace-3T

### 5.1 总体结构

```text
Protein structure + Ligand structure
        |
        v
Formal Site Tokenizer
        |
        v
Protein site tokens + Ligand atom_group tokens
        |
        v
Pair Interaction Mixer
        |
        v
Interaction Resampler / Perceiver Adapter
        |
        v
M compact interaction tokens
        |
        v
Unified decoder-only / prefix-LM Transformer
        |
        +-- <TASK_AFFINITY>      -> affinity bin / residual tokens
        +-- <TASK_SITE_CONTACT>  -> site / contact pointer tokens
        +-- <TASK_POSE_REFINE>   -> pose anchor / SE(3) / torsion tokens
```

### 5.2 两个核心模块

第一版核心只保留两个大模块。

#### 模块 1：Structure-to-Token Adapter

功能：把 protein / ligand 结构变成统一结构 token。

输入：

```text
protein formal site tensor
ligand atom_group site tensor
optional pretrained protein / ligand embeddings
pairwise geometric / chemical features
```

输出：

```text
protein site tokens
ligand site tokens
compact interaction latent tokens
```

建议增加 Interaction Resampler，把 protein site x ligand site 的 pair field 压缩成固定数量的 latent interaction tokens：

```text
M = 32 或 64
```

目的：避免把所有 pair token 直接送进 decoder 导致序列过长、训练不稳。

#### 模块 2：Unified AR Decoder

功能：一个共享 decoder 根据 task token 生成不同任务输出。

```text
[interaction latent tokens] + [task token] + [output prefix]
-> autoregressive output tokens
```

可以使用 decoder-only 或 prefix-LM。任务差异优先用 task embedding / adapter / light MoE 处理，不要一开始给每个任务设计完整 realizer。

---

## 6. 3D Tokenizer 与结构输入要求

3D tokenizer 是 TRACE-Lite 的地基。decoder 可以保持简洁，但 protein / ligand / interface tokenizer 必须认真设计。

### 6.1 Protein tokenizer

Protein 不应只用 sequence token，也不应只用每个 residue 一个粗 token。第一版建议使用：

```text
residue token
+ functional-site token
+ continuous local geometry
+ optional pretrained / structure token channel
```

Protein site token 至少应包含：

```text
residue type
chain / residue id
member atom ids
site center coordinate
site type / functional group
donor / acceptor flags
charged / hydrophobic / aromatic flags
metal-coordination candidate flag
local frame / direction vector
sidechain chi / rotamer coarse state
mask / fallback flag
optional pretrained embedding
```

核心目标：让模型看到真正可能参与 binding 的 functional site，而不是只看到 residue 名字。

### 6.2 Ligand tokenizer

Ligand 不应只用 SMILES / SELFIES，也不应只用裸 atom list。第一版建议使用：

```text
atom token
+ pharmacophore group token
+ conformer geometry features
+ atom mapping metadata
```

Ligand site token 至少应包含：

```text
atom ids / member atom ids
atom or group type
site center coordinate
formal charge
donor / acceptor flags
aromatic / hydrophobic flags
positive / negative ionizable flags
halogen / metal-binder flags
ring centroid / ring normal
rotatable bond context
local conformer descriptor
```

核心目标：同时保留 exact atom grounding 和 group-level pharmacophore 语义。

### 6.3 Complex-level interaction tokenizer

Protein tokenizer 和 ligand tokenizer 只是前半步。binding 任务真正需要的是 interface tokenizer：

```text
protein site i + ligand site j
-> pair interaction token pair_ij
```

Pair token 应包含：

```text
protein site embedding
ligand site embedding
p_i * l_j interaction feature
distance RBF
relative direction / orientation
local-frame relation
donor-acceptor compatibility
charge complementarity
hydrophobic compatibility
aromatic compatibility
metal coordination compatibility
steric clash feature
```

原则：

```text
离散 token 给 decoder 使用；
连续几何给 Pair Interaction Mixer 使用；
metadata / atom-site mapping 给 site/contact/pose grounding 使用。
```

不要在 tokenizer 阶段把所有 3D 几何都压扁成离散词。第一版必须保留连续坐标、局部 frame、方向和 pair 几何特征。

---

## 7. 信息压缩三阶段路线

Pair Interaction Mixer 会产生大量 pair token：

```text
pair_field: [B, Np, Nl, d_pair]
```

如果直接送入 decoder，长度会随 protein 和 ligand 尺寸爆炸。因此需要 Interaction Resampler：

```text
[Np, Nl, d_pair]
-> [M, d_model]
```

其中：

```text
M = 32 / 64 / 128
```

信息压缩不是简单 mean pooling。真实 protein-ligand interaction 是稀疏的，强 contact / anchor / geometry evidence 会被大量无关 pair 稀释。因此压缩模块必须做到：

```text
1. 固定长度。
2. 保留关键相互作用。
3. 保留 attention / grounding 映射。
4. 同时服务 affinity、site/contact、pose 三任务。
5. 训练稳定，不在早期做过硬离散选择。
```

下一版建议分三阶段推进。

---

### 7.1 第一阶段：Continuous Perceiver / Q-Former 风格 Resampler

第一版主路径采用：

```text
learnable query cross-attention resampler
```

形象理解：

```text
把所有 protein-ligand pair interaction 当成大量证据材料；
派出 M 个可学习 query 像 M 个审稿人；
每个 query 从所有 pair 里 cross-attend 并写出一张摘要卡片；
最后得到 M 个 compact interaction tokens。
```

形式：

```text
pair_tokens:       [B, Np*Nl, d_pair]
learnable queries: [M, d_model]

interaction_tokens = CrossAttention(
    query = learnable_queries,
    key   = pair_tokens,
    value = pair_tokens,
    mask  = pair_mask,
    bias  = optional_pair_attention_bias
)

interaction_tokens = SelfAttention(interaction_tokens)
```

建议加入轻量 bias：

```text
distance bias
chemistry compatibility bias
pair importance bias
```

第一版选择它的原因：

```text
1. soft selection，训练稳定。
2. 固定长度，适合接 unified decoder。
3. 不要求早期 pair scorer 已经准确。
4. 可以保存 attention map 做解释。
5. 兼容三任务共享。
```

第一版不要使用：

```text
mean pooling
single CLS pooling
hard top-k only
slot attention only
VQ only
```

Continuous Resampler 是第一版最优、最规范的主路径。

---

### 7.2 第二阶段：Sparse top-k / Slot Attention 增强解释性

第二阶段不是替换第一阶段，而是在 continuous resampler 稳定后增强 grounding 和解释性。

#### Sparse top-k

形象理解：

```text
先让助理从几千/几万个 pair 里挑出最可能重要的 K 条证据；
再让 M 个 query 精读这些证据并总结。
```

形式：

```text
score_ij = PairImportanceHead(pair_ij)
topK_pairs = select_topK(pair_tokens, score_ij)
interaction_tokens = CrossAttention(queries, topK_pairs, topK_pairs)
```

建议训练节奏：

```text
early: dense / soft attention
middle: soft top-k / sparse attention
late: optional hard top-k candidate pruning
```

不建议第一版直接 hard top-k，因为早期 pair importance score 不准，可能把真正关键的 contact / anchor pair 排除掉。

#### Slot Attention

形象理解：

```text
给模型 M 个文件夹，让它把 pair evidence 自动分组：
文件夹 1 可能收集氢键证据；
文件夹 2 可能收集疏水 pocket；
文件夹 3 可能收集盐桥；
文件夹 4 可能收集 pose anchor。
```

价值：

```text
1. 可能形成更清楚的 interaction group。
2. 有利于 case study 和解释。
3. 有利于把 compact token 与具体 pair 区域对应。
```

风险：

```text
1. 训练比普通 cross-attention 更难。
2. 容易 slot collapse。
3. 不同 seed 下 slot 语义可能不稳定。
4. 需要 diversity / orthogonality / coverage regularization。
```

因此 Slot Attention 可作为第二阶段增强或 ablation，不作为第一版主路径。

---

### 7.3 第三阶段：VQ / Discrete Interaction Code

VQ 全称是：

```text
Vector Quantization
```

中文是：

```text
向量量化
```

VQ 的核心不是“连续表示”，而是：

```text
continuous interaction vector
-> nearest codebook vector
-> discrete code id
-> <INTERACTION_CODE_k>
```

形象理解：

```text
连续表示像高清照片，细节很多但难命名；
VQ 像给照片归类贴标签，把复杂模式放进一本有限词典。
```

在 TRACE-Lite 中，VQ 的潜在价值是：

```text
1. 把 interaction pattern 变成离散词表。
2. 让 decoder 更像语言模型，可以预测 / 生成 interaction code。
3. 方便统计哪些 code 与 affinity、site、pose 相关。
4. 让相似 interaction pattern 复用同一个 code，提高抽象能力。
5. 后期支持 masked interaction-code pretraining 或 trace language。
```

例如：

```text
CODE_17 可能对应强氢键模式
CODE_42 可能对应疏水 pocket 模式
CODE_81 可能对应芳香堆叠模式
CODE_103 可能对应金属配位模式
CODE_140 可能对应 steric clash 模式
```

但是，VQ 不适合第一版主路径。原因：

```text
1. 会产生量化误差。
2. 可能丢掉 pose/contact 所需的精细距离和方向。
3. codebook 可能 collapse 或使用不均。
4. 需要 commitment loss / codebook usage regularization。
5. 如果 continuous pair representation 还没学稳，VQ 会放大不稳定。
```

因此第三阶段建议是：

```text
先用 continuous interaction tokens 跑通三任务；
再对 continuous tokens 做 offline clustering / prototype analysis；
如果自然形成有意义的 interaction clusters，再训练 VQ / FSQ / discrete interaction code；
VQ 作为解释性和离散生成增强，而不是第一版性能主路径。
```

---

### 7.4 三阶段最终顺序

最终顺序应为：

```text
第一阶段：continuous resampler
    目标：先学会看懂 protein-ligand interaction。

第二阶段：sparse / slot grounding
    目标：增强解释性、定位能力和 pair-level grounding。

第三阶段：VQ / discrete code
    目标：把稳定的 continuous interaction pattern 命名成离散结构语言。
```

不要反过来。特别是不要在三任务闭环未稳定前，把 VQ 或 hard top-k 作为主压缩机制。

---

## 8. Interaction Resampler 训练与监督要求

为了让压缩模块不是自由乱学，第一版建议加入轻量辅助监督。

### 8.1 Pair contact auxiliary loss

```text
pair_contact_logit_ij = ContactHead(pair_ij)
L_pair_contact = BCE / focal loss over pair contacts
```

目标：让 pair field 具备基本 contact recognition 能力。

### 8.2 Protein site auxiliary loss

```text
protein_site_logit_i = SiteHead(protein_site_i, pair_context_i)
L_site_aux = BCE / focal loss over protein site labels
```

目标：让 protein site token 和 pair context 对 binding site 有可学习信号。

### 8.3 Resampler attention coverage loss

目标：真实 contact / anchor pair 应该被至少一部分 interaction query attend 到。

形式可以是：

```text
max_m attention[m, true_contact_pair] should be high
```

或者：

```text
sum attention mass over true contacts should exceed threshold
```

### 8.4 Diversity / anti-collapse regularization

避免所有 interaction tokens 关注同一个区域。

可选：

```text
attention diversity loss
slot / query off-diagonal cosine penalty
coverage entropy regularization
```

原则：不同 compact token 应该覆盖不同 interaction evidence，而不是全部看同一个高分 pair。

---

## 9. 三任务如何使用压缩结果

### 9.1 Affinity

Affinity 任务主要使用 compact interaction tokens 的全局 summary：

```text
interaction_tokens
-> unified decoder
-> <AFF_BIN_k> <AFF_RES_r>
```

可以加一个训练辅助 head：

```text
affinity_aux = MLP(pool(interaction_tokens))
```

但主接口仍然是生成式 affinity token。

### 9.2 Site / Contact

Site/contact 不能只依赖压缩 token。必须保留 pair field 作为高分辨率底图。

推荐：

```text
compact interaction tokens 负责理解和全局上下文；
pair field 负责 pointer decoding、contact localization 和解释；
resampler attention map 负责把 compact token 映射回具体 site-pair。
```

### 9.3 Pose / Refinement

Pose 任务需要 anchor pair 和局部几何。可以设置一部分 pose-oriented queries：

```text
pose_anchor_queries
```

它们 cross-attend 到 pair field 后，为 decoder 生成：

```text
<ANCHOR P_i L_j>
<TRANS_BIN_*>
<ROT_BIN_*>
<TORSION_BIN_*>
```

同时保留 pair-level anchor auxiliary loss。

---

## 10. 三个任务的建议输出格式

以下输出格式是第一版最小闭环。后续可以扩展，但不要一开始使用完整 program language。

### 10.1 Affinity / property understanding

任务 token：

```text
<TASK_AFFINITY>
```

输出：

```text
<AFF_BIN_k> <AFF_RES_r> <EOS>
```

推荐使用 bin + residual：

```text
affinity = bin_center(k) + residual(r)
```

损失：

```text
L_aff = CE(affinity_bin) + MSE(affinity_residual)
```

可以保留 direct scalar regression head 作为训练辅助，但主接口仍然是生成 affinity tokens。

### 10.2 Binding site / contact understanding

任务 token：

```text
<TASK_SITE_CONTACT>
```

输出建议使用 pointer / copy 风格，而不是巨大固定词表：

```text
<SITE_LIST> <P_SITE_i> <P_SITE_j> ... </SITE_LIST>
<CONTACT_LIST> <PAIR_p_i_l_j> ... </CONTACT_LIST>
<EOS>
```

训练损失：

```text
L_site = BCE / focal loss over protein sites
L_contact = BCE / pair ranking loss over protein-ligand pairs
L_site_gen = pointer CE for generated site/contact tokens
```

第一版不要生成完整 SITE program：

```text
<SITE_STEP_PROPOSE>
<SITE_STEP_ANCHOR>
<SITE_STEP_EXPAND>
<SITE_STEP_VALIDATE>
```

这些可以后续作为 explanation extension。

### 10.3 Pose / local refinement generation

任务 token：

```text
<TASK_POSE_REFINE>
```

第一版只做 local pose / refinement，不做 full blind docking。

输出：

```text
<POSE>
<ANCHOR P_i L_j>
<TRANS_BIN_x> <TRANS_BIN_y> <TRANS_BIN_z>
<ROT_BIN_a> <ROT_BIN_b> <ROT_BIN_c>
<TORSION_BIN_1> ... <TORSION_BIN_n>
</POSE>
<EOS>
```

训练损失：

```text
L_pose =
    CE(anchor_pair)
  + CE(translation_bins)
  + CE(rotation_bins)
  + CE(torsion_bins)
  + coordinate RMSD auxiliary loss
```

第一版暂缓：

```text
multi-candidate pose
clash correction program
pose quality program
VQ local patch
complex reranking
```

---

## 11. 训练阶段

### Stage 0：数据和标签审计

必须先输出：

```text
train / val / test split summary
site token coverage
ligand atom_group coverage
contact / event label source summary
pretrained feature coverage
fallback fraction
```

任何结构-aware pretrained feature 都必须报告 fallback 情况。

### Stage 1：结构表示预热

目标：先证明结构 tokenizer + interaction mixer 有用。

训练：

```text
site tokens + interaction latents
-> direct auxiliary heads
```

任务：

```text
affinity regression / bin
site map
contact map
optional pose auxiliary
```

此阶段不要求 AR decoder 完整生成。

### Stage 2：teacher-forced 统一生成

目标：让统一 decoder 学会三个任务的输出语法和条件分布。

训练：

```text
complex tokens + task token + gold output prefix
-> next-token prediction
```

此阶段以 teacher forcing 为主。

### Stage 3：联合训练但保留 direct auxiliary path

目标：生成接口和直接任务监督共同稳定模型。

损失：

```text
L_total =
    w_aff  * L_affinity
  + w_site * L_site_contact
  + w_pose * L_pose
  + w_lm   * L_next_token
  + small w_event * L_event_aux
```

direct auxiliary heads 只作为训练稳定器和 ablation，不作为最终生成接口。

### Stage 4：confidence-gated probabilistic scheduled sampling

目标：缓解 teacher-forced 训练和 generated inference 的 mismatch。

不要直接全量 free-running。应使用 confidence gate + random gate。

### Stage 5：generated evaluation / self-training

目标：用模型自己生成的输出做下游评估和少量训练。

可以加入：

```text
generated site/contact overlap loss
generated pose RMSD / anchor consistency loss
generated affinity calibration loss
teacher-forced vs generated consistency loss
```

但不建议第一版大量使用 sequence-level RL 或 heavy generated downstream loss。

---

## 12. Teacher forcing 的使用要求

Teacher forcing 的作用是：

```text
训练时给模型真实前缀，让它学习在正确上下文下预测下一个 token。
```

在本项目中必须区分：

```text
teacher-forced result
generated / free-running result
```

诊断规则：

```text
teacher-forced 好，generated 差
=> decoder generation / exposure bias 是瓶颈

teacher-forced 也差
=> representation、task schema、loss 或 output head 本身有问题

teacher-forced 和 generated 都好
=> 统一生成路径基本成立
```

任何报告不得只用 teacher-forced 指标宣称生成模型成功。

---

## 13. Scheduled sampling 策略

### 13.1 总体策略

采用：

```text
confidence-gated probabilistic scheduled sampling
```

即：先判断模型当前 token 是否足够可信；过阈值后，再按随机概率决定是否使用模型自己的 token。

```text
eligible_t =
    confidence_t >= tau_conf
and margin_t >= tau_margin
and entropy_t <= tau_entropy 可选
and token_is_legal

use_self_t =
    eligible_t
and Bernoulli(p_self) = 1

input_{t+1} =
    predicted_token_t, if use_self_t
    gold_token_t, otherwise
```

其中：

```text
confidence = top1 probability
margin = top1 probability - top2 probability
entropy = -sum p log p
```

loss 仍然对 gold token 计算，sampling 只影响下一步输入。

### 13.2 不推荐的策略

不要使用：

```text
如果 generated token == gold token，则用自己；否则 teacher forcing
```

这是 oracle-gated scheduled sampling。训练时用了 gold correctness 做 gate，推理时没有这个信息，会造成新的 train-test mismatch。

不要使用：

```text
confidence > tau 就 100% 用自己
```

这会导致某个 epoch 置信度整体升高时 self-sampling 暴涨，训练不稳。

### 13.3 两级门控

推荐实现：

```text
Gate 1: warmup / epoch gate
Gate 2: confidence + margin + legality gate
Gate 3: random probability gate
Gate 4: optional target self-rate cap
```

伪代码：

```python
eligible = (
    (conf >= tau_conf)
    & (margin >= tau_margin)
    & legal_mask_for_pred
)

random_gate = torch.rand_like(conf) < p_self
use_self = eligible & random_gate

next_token = torch.where(use_self, pred_token.detach(), gold_token)
loss_t = cross_entropy(logits_t, gold_token)
```

### 13.4 Schedule

推荐 schedule：

```text
epoch 0 - warmup:
    p_self = 0.00
    全 teacher forcing

early scheduled sampling:
    tau_conf 高
    p_self 低

late scheduled sampling:
    tau_conf 缓慢降低
    p_self 缓慢升高
```

示例：

```text
epoch 0-5:
    p_self = 0.00

epoch 6-10:
    tau_conf = 0.90
    p_self = 0.10

epoch 11-20:
    tau_conf = 0.85
    p_self = 0.25

epoch 20+:
    tau_conf = 0.80
    p_self = 0.40
```

### 13.5 目标 self-rate 控制

比固定 `p_self` 更稳的方式是控制目标 self-sampling 比例：

```text
actual_self_rate ≈ eligible_rate * p_self
```

若希望全局实际比例为 `target_self_rate`：

```text
p_self = min(1.0, target_self_rate / max(eligible_rate, eps))
```

建议：

```text
target_self_rate:
0% -> 10% -> 25% -> 40%
```

### 13.6 任务级阈值建议

不同任务应有不同阈值和 self-sampling 上限。

#### Affinity

输出短，错误传播有限。

```text
tau_conf = 0.85 - 0.95
tau_margin = 0.08 - 0.15
p_self_max = 0.40 - 0.60
```

#### Site / Contact

pointer 候选多，概率天然分散。

```text
tau_conf = 0.60 - 0.80
tau_margin = 0.05 - 0.10
p_self_max = 0.25 - 0.45
```

#### Pose

错误传播严重，应最保守。

```text
tau_conf = 0.85 - 0.95
tau_margin = 0.10 - 0.20
p_self_max = 0.10 - 0.30
```

#### EOS

EOS 必须单独更严格，避免过早结束。

```text
tau_eos >= 0.95
```

### 13.7 Position-aware sampling

可选加入位置因子：

```text
p_self(e, t) = p_task(e) * pos_factor(t)
```

建议：

```text
序列前段更偏 teacher forcing
序列后段逐渐增加 self-sampling
```

Pose 任务中，anchor token 前期尤其应保守。

---

## 14. 推理时 decoding 策略

训练时 scheduled sampling 不等于推理时 sampling。

推理时建议：

```text
AFFINITY:
    greedy 或 small beam

SITE / CONTACT:
    constrained beam / top-k pointer

POSE:
    beam / top-k candidate + geometry rerank
```

不建议对结构任务使用过于开放的 top-p / high-temperature sampling，除非用于候选多样性生成，并且后续有结构约束 reranking。

---

## 15. 评估要求

必须同时报告：

```text
teacher-forced metrics
generated metrics
```

三任务指标建议：

### Affinity

```text
RMSE
MAE
Pearson
Spearman
prediction std / true std ratio
affinity bin accuracy
residual error
```

### Site / Contact

```text
site AUPRC
site AUROC
site MCC
precision@k
contact AUPRC
contact top-k precision
pointer generation accuracy
```

### Pose / Refinement

```text
translation error
rotation error
torsion accuracy
ligand RMSD
centroid distance
top-k success@2A where available
anchor pair accuracy
```

额外必须报告：

```text
generated vs teacher-forced gap
self-sampling rate by task
self-sampling rate by token type
EOS early-stop rate
invalid token / invalid grammar rate
```

---

## 16. Ablation 要求

第一版必须保留可诊断 ablation，不得只报告 full model。

最低要求：

```text
1. direct multi-task baseline
2. shared decoder teacher-forced only
3. shared decoder + confidence-gated scheduled sampling
4. no event auxiliary
5. event auxiliary only, no bottleneck
6. no pretrained features
7. no pair interaction mixer
8. generated vs teacher-forced evaluation
```

### 16.1 信息压缩 ablation

新增必须比较：

```text
1. mean pooling
2. CLS pooling
3. continuous Perceiver / Q-Former resampler
4. continuous resampler + distance / chemistry bias
5. continuous resampler + pair contact auxiliary loss
6. continuous resampler + sparse top-k candidate pruning
7. continuous resampler + slot attention / diversity regularization
8. optional VQ / discrete interaction code, only after continuous version stable
```

重点观察：

```text
Affinity RMSE / Pearson / Spearman
Site AUPRC / MCC / precision@k
Contact AUPRC
Pose RMSD / anchor accuracy
generated vs teacher-forced gap
memory / speed
top attended pairs 是否覆盖真实 contact / anchor pair
不同 query / slot 是否 collapse
```

在三任务闭环稳定之前，不允许把 strict trace bottleneck 或 VQ code 作为唯一主结果。

如果后续重新加入 trace bottleneck，必须比较：

```text
direct path
soft trace coupling
strict trace bottleneck
```

诊断规则：

```text
direct path 好，strict trace 差
=> bottleneck 太窄或 generated trace 不可靠

gold / teacher-forced trace 好，generated trace 差
=> AR decoder 是瓶颈

site/contact 好，但 affinity 无提升
=> evidence pooling / calibration 有问题

prediction std / true std ratio 低
=> affinity dynamic range 塌缩

continuous resampler 好，VQ 差
=> interaction codebook 量化过早或几何损失过大

VQ code usage collapse
=> codebook 设计、commitment loss、usage regularization 或训练时机有问题
```

---

## 17. 禁止事项

在第一版 TRACE-Lite / UniTrace-3T 未稳定前，不建议做以下事情：

```text
1. 不要默认启用 strict trace bottleneck。
2. 不要把 PoseRealizer、SiteRealizer、AffinityRealizer 全量作为第一版主路径。
3. 不要同时打开 full program token、VQ local patch、generated downstream loss、reranking。
4. 不要只用 teacher-forced 指标证明生成模型有效。
5. 不要用测试集调阈值、选 checkpoint 或做后处理校准。
6. 不要把 pose 模块包装成 full blind docking，除非真正实现全局搜索或等变生成机制。
7. 不要把 structure-aware pretrained feature 的 fallback 当作完整结构预训练输入。
8. 不要在 continuous interaction representation 未稳定前，把 VQ / discrete code 作为主压缩路径。
9. 不要把所有 pair interaction 简单平均后送入 decoder。
10. 不要在早期使用 hard top-k 排除大量 pair evidence。
```

---

## 18. 成功门槛

第一阶段成功不要求马上超过所有成熟 SOTA。更现实的门槛是：

```text
1. 三个任务都能通过统一 decoder 生成合法输出。
2. generated metrics 与 teacher-forced metrics 的差距可控，并随训练缩小。
3. shared decoder 模型优于 direct multi-head baseline 或至少接近，同时提供生成接口。
4. scheduled sampling 后 generated performance 改善，而 teacher-forced performance 不显著崩塌。
5. site/contact/pose 至少有一个过程任务能稳定提供对 affinity 有帮助的信号。
6. affinity prediction dynamic range 不明显塌缩。
7. continuous resampler 的 attention 能覆盖真实 contact / anchor pair。
8. sparse / slot 版本相比 continuous base 有解释性提升，且不显著损害主指标。
9. VQ / discrete code 只有在 continuous base 稳定后才作为成功候选；成功时必须证明 codebook 有意义、使用不 collapse、且对生成/解释有增益。
```

如果 full explainable trace 后续加入，但没有超过 direct / soft-coupled baseline，应报告为 failure analysis，不得宣称 strict trace 路径成功。

---

## 19. 下一阶段实现优先级

建议顺序：

```text
1. 实现 TRACE-Lite 数据格式和三任务 output schema。
2. 实现 Hybrid Interface-Aware 3D Tokenizer。
3. 实现 Pair Interaction Mixer。
4. 实现 continuous Perceiver / Q-Former Interaction Resampler。
5. 实现 unified decoder-only / prefix-LM 主干。
6. 先跑 teacher-forced 三任务生成。
7. 加 direct auxiliary heads 稳定表示。
8. 加 confidence-gated probabilistic scheduled sampling。
9. 报告 generated vs teacher-forced gap。
10. 加 pair contact / site / resampler attention coverage auxiliary loss。
11. 再逐步加入 sparse top-k 或 slot attention 做解释性增强。
12. 等 continuous interaction tokens 稳定后，做 offline clustering / prototype analysis。
13. 只有 clustering / prototype 有意义时，再考虑 VQ / FSQ / discrete interaction code。
14. 最后才考虑 soft trace coupling / strict trace bottleneck。
```

---

## 20. 最终定位

TRACE 下一版应从：

```text
trace-centered mechanistic program model
```

暂时调整为：

```text
task-token conditioned structure generation model
```

其底层表示路线应从：

```text
full symbolic trace program first
```

调整为：

```text
continuous interaction representation first
-> sparse / slot grounding second
-> discrete interaction language third
```

先完成统一三任务生成模型的最小闭环，再逐步恢复可解释 trace 作为增强模块。这样更符合当前目标，也更接近可训练、可评估、可迭代的研究路径。
