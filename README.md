
# smart-doctor —— 中文医疗NLP问答系统 | AI智能医生（NER + 知识图谱 + 多轮对话）
# 从模型训练到服务部署的完整实践

> smdoctor 是一个面向医疗领域的智能问答与辅助诊断系统，采用**离线模型训练 + 在线服务部署**的完整架构。系统基于 PyTorch 实现 BiLSTM-CRF 命名实体识别（NER），并结合 BERT 中文预训练模型进行实体审核与语义匹配；底层使用 Neo4j 图数据库构建“疾病-症状”知识图谱，最终通过 Flask + Redis 实现多轮对话管理，全面覆盖了从核心算法到工程落地的全流程。

---
![Python](https://img.shields.io/badge/Python-3.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9-red)
![Neo4j](https://img.shields.io/badge/Neo4j-4.0-green)
![BERT](https://img.shields.io/badge/BERT-base--chinese-orange)
## 一、项目整体架构

```
smdoctor/
├── offline/          # 离线数据处理与模型训练
│   ├── ner_model/           # 医疗命名实体识别（BiLSTM-CRF）
│   │   ├── bilstm_crf.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── preprocess_data.py
│   │   ├── loader_data.py
│   │   ├── evaluate_model.py
│   │   └── data/ log/ model/   # 数据集、日志、模型权重
│   ├── review_model/        # 实体审核模型（BERT + RNN）
│   │   ├── bert_chinese_encode.py
│   │   ├── RNN_MODEL.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── bilstm.py
│   │   └── reviewed/        # 审核通过的输出目录
│   └── neo4j_write.py       # 知识图谱构建脚本
│
├── online/           # 在线服务层
│   ├── bert_server/         # 句子相关性微服务（BERT + 全连接）
│   │   ├── app.py
│   │   ├── train.py
│   │   ├── finetuning_net.py
│   │   └── bert_chinese_encode.py
│   └── main_server/         # 主逻辑/对话服务
│       ├── app.py
│       ├── config.py
│       └── test.py
│
├── requirements.txt         # Python依赖
└── README.md
```

---

## 二、离线层：offline

### 2.1 命名实体识别（NER）— `ner_model/`

**目标**：从非结构化医疗文本中抽取**疾病（dis）**和**症状（sym）**两类实体。

#### 核心方案：BiLSTM-CRF

| 文件 | 职责 |
|------|------|
| `bilstm_crf.py` | 定义 **BiLSTM-CRF** 模型，包含词嵌入、双向LSTM、发射矩阵、转移矩阵、前向算法、维特比解码 |
| `train.py` | 训练入口，使用 `Adam` 优化器，记录 Loss / Acc / Recall / F1 曲线并保存模型 |
| `predict.py` | 单条/批量预测，支持**滑动窗口（offset）**处理超长文本 |
| `preprocess_data.py` | 将字符级 BIO 标注数据转换为 `.npz` 二进制训练集 |
| `loader_data.py` | 封装 PyTorch `DataLoader`，按 80/20 划分训练集与验证集 |
| `evaluate_model.py` | 实体级别评估：准确率（Accuracy）、召回率（Recall）、F1-Score |

#### BIO 标注体系

```text
O     : 非实体
B-dis : 疾病实体起始
I-dis : 疾病实体中间
B-sym : 症状实体起始
I-sym : 症状实体中间
```

#### 关键知识点

- **CRF（条件随机场）**：通过转移矩阵建模标签间依赖，避免非法标签序列（如 `I-dis` 紧跟 `B-sym`）。
- **前向算法（Forward Algorithm）**：计算所有路径得分，使用 Log-Sum-Exp 保证数值稳定。
- **维特比解码（Viterbi Decode）**：推理时寻找全局最优标签路径。
- **滑动窗口预测**：长文本以 `sentence_length` 切分，窗口间设置 `offset` 重叠，防止边界处实体被截断丢失。

---

### 2.2 实体审核模型 — `review_model/`

**目标**：对 NER 抽取的实体进行**二分类审核**，过滤错误结果，提升下游知识图谱质量。

#### 核心方案：BERT + RNN

| 文件 | 职责 |
|------|------|
| `bert_chinese_encode.py` | 通过 `torch.hub` 加载 `bert-base-chinese`，编码为 768 维向量序列 |
| `RNN_MODEL.py` | 自定义简单 RNN（`i2h` + `i2o` + `LogSoftmax`） |
| `train.py` | 读取 CSV 数据，BERT 编码后输入 RNN，使用 `NLLLoss`，**手动梯度更新** |
| `predict.py` | 加载 `BERT_RNN.pth`，批量预测并输出审核通过的实体文件 |
| `bilstm.py` | 独立 BiLSTM 实现（备用/对比实验） |

#### 关键知识点

- **预训练模型微调**：BERT 提取深层语义特征，轻量 RNN/全连接层负责分类任务。
- **手动参数更新**：`p.data.add_(-lr, p.grad.data)` 是 `optimizer.step()` 的底层实现，帮助理解优化器本质。
- **NLLLoss + LogSoftmax** vs **CrossEntropyLoss**：前者是后者的分解步骤，分开写便于理解概率输出过程。

---

### 2.3 知识图谱构建 — `neo4j_write.py`

**目标**：将审核后的结构化数据（疾病-症状 CSV）写入 **Neo4j** 图数据库，构建可查询的医疗知识图谱。

#### 图模型设计

- **节点**：
  - `:Disease {name}` — 疾病
  - `:Symptom {name}` — 症状
- **关系**：
  - `(Disease)-[:dis_to_sym]->(Symptom)` — 疾病与症状的关联

#### 关键知识点

- **MERGE vs CREATE**：`MERGE` 具有幂等性，防止重复创建节点。
- **索引优化**：为 `Disease.name` 和 `Symptom.name` 建索引，加速图谱查询。
- **Cypher 查询语言**：图数据库的声明式查询语法，与 SQL 思路不同但有对应关系。

---

## 三、在线层：online

### 3.1 句子相关性服务 — `bert_server/`

**目标**：独立微服务，判断用户**当前输入**与**上一轮输入**的语义相关性，支撑多轮对话决策。

#### 核心方案：BERT 双句编码 + 全连接分类

| 文件 | 职责 |
|------|------|
| `bert_chinese_encode.py` | 对 `text_1` / `text_2` 进行 BERT 编码，含 `segment_ids`、截断/填充（`max_len=10`） |
| `finetuning_net.py` | 微调网络：展平 → `Dropout → FC(8) → ReLU → Dropout → FC(2)` |
| `train.py` | 训练脚本，`CrossEntropyLoss` + `SGD`，保存 Loss / Acc 曲线 |
| `app.py` | Flask 微服务，暴露 `GET /v1/recognition/?text1=...&text2=...`，返回 `0/1` |

#### 关键知识点

- **句子对建模（Sentence Pair Modeling）**：BERT 的 `token_type_ids`（segment embedding）区分两句输入。
- **Padding & Truncation**：固定输入张量形状，适配批量计算。
- **微服务拆分**：语义匹配独立部署，便于主服务按需调用、独立扩缩容。

---

### 3.2 主逻辑/对话服务 — `main_server/`

**目标**：接收用户请求，管理对话状态，调度各模型与数据库，生成回复。

#### 核心方案：Flask + Redis 会话管理 + Neo4j 查询 + 规则模板 + 百度 UNIT 兜底

| 文件 | 职责 |
|------|------|
| `app.py` | 主服务入口，`Handler` 类处理首句/非首句逻辑，暴露 `POST /v1/main_serve/` |
| `config.py` | 集中配置：Redis、Neo4j、bert_server 地址、超时、模板路径、会话过期时间 |
| `test.py` | Redis 与 Neo4j 连通性测试 |

#### 对话处理流程

```
用户输入 → main_server
    │
    ├─ 首句？ → 直接查询 Neo4j（症状→疾病）
    │            → 有结果：规则模板回复疾病列表
    │            → 无结果：百度 UNIT 兜底
    │
    └─ 非首句？ → 调用 bert_server 判断与上一轮文本是否相关
                  → 相关：查 Neo4j → 与历史疾病做并集/差集 → 回复新疾病
                  → 不相关/异常：百度 UNIT 兜底
    │
    └─ Redis：保存 current/previous_d/previous，设置 TTL
```

#### 关键知识点

- **对话状态管理（Dialogue State Tracking）**：Redis Hash 存储用户上下文（`previous_d`、`previous`），TTL 自动过期。
- **服务降级（Fallback）**：bert_server 超时或异常时，自动降级到百度 UNIT，保证服务可用性。
- **规则模板回复**：查询结果填入预定义模板，兼顾可控性与自然度。
- **差集回复策略**：已回复的疾病不重复输出，仅返回**新增**疾病，优化对话体验。

---

## 四、核心技术栈

| 类别 | 技术/框架 | 用途 |
|------|-----------|------|
| 深度学习框架 | PyTorch | 模型定义、训练、推理 |
| 预训练模型 | bert-base-chinese | 中文语义编码 |
| 图数据库 | Neo4j | 知识图谱存储与查询 |
| 缓存/会话 | Redis | 对话状态管理 |
| Web 服务 | Flask | API 服务化 |
| 数据科学 | NumPy / Pandas / scikit-learn / Matplotlib | 数据处理与可视化 |
| 外部 AI | 百度 UNIT | 兜底对话 |
| 部署 | gunicorn / waitress | WSGI HTTP Server |

---

## 五、核心知识点总结

### 5.1 自然语言处理（NLP）

1. **命名实体识别（NER）**
   - BIO 标注规范
   - BiLSTM-CRF 架构：发射分数 + 转移分数
   - 维特比解码求最优路径
   - 实体级别 Precision / Recall / F1 评估

2. **预训练语言模型应用**
   - BERT 编码：`tokenizer.encode`、`last_hidden_state`
   - 句子对输入：`token_type_ids` / `segment_ids`
   - 微调策略：冻结底层 + 上层任务网络

3. **文本分类**
   - 二分类建模
   - `NLLLoss` + `LogSoftmax` 与 `CrossEntropyLoss` 的关系
   - 自定义 RNN Cell 的实现

### 5.2 知识图谱

1. **图数据建模**：节点、关系、属性
2. **Cypher 查询**：`MATCH`、`MERGE`、`CREATE INDEX`
3. **图谱与对话结合**：症状→疾病的图推理

### 5.3 对话系统架构

1. **多轮状态管理**：Redis Hash + TTL
2. **语义相关性判断**：独立微服务解耦
3. **兜底降级**：外部 API 保障可用性
4. **规则与模型结合**：结构化查询 + 模板生成

### 5.4 工程化

1. **数据工程**：字符级编码、Padding、滑动窗口、`.npz` 存储、`DataLoader` 封装
2. **服务拆分**：`main_server` 业务编排 + `bert_server` 语义计算，HTTP 通信
3. **模型持久化**：`state_dict` 保存/加载、训练曲线可视化

---

## 六、数据流转全景图

```
【原始医疗文本】
       │
       ▼
【NER 模型】BiLSTM-CRF ──→ 抽取疾病/症状实体
       │
       ▼
【审核模型】BERT + RNN ──→ 过滤低质量实体
       │
       ▼
【结构化 CSV】疾病-症状对应表
       │
       ▼
【Neo4j 图数据库】构建知识图谱
       ▲
       │
【在线对话】用户输入症状 ──→ 主服务查询图谱 ──→ 返回可能疾病
       │
       └─ 多轮场景 ──→ bert_server 判断相关性 ──→ 继续追问/补充疾病
```

---
---
## 最后说几句

如果正在往大模型应用方向走，智能在线医生是一个很实在的起点。它具体到能写进简历，又通用到能带你去更远的地方。

这个项目我会一直开源，一直免费。

但有些东西，确实放不进 README。

凌晨一点向着如何解决大模型幻觉，搜遍全网找不到一个能用的答案，翻了四页都是同一篇文章的搬运。理论看了不少，一落地就抓瞎。代码能跑通，却说不清为什么这么设计。想找人聊聊，翻遍通讯录，没有一个在做同样的事。

这些时刻，我一个人经历过太多次。

后来慢慢意识到，**技术上的坎，有时候不是卡在代码本身，是卡在"只有你一个人"**。

所以我把这个项目写出来，也维护了一个小圈子，**AI大模型工程社**。里面没有噱头，就是一群在做同一件事的人。分享一些文档里写不下的东西：关键代码的注释版、设计决策的推演过程、面试高频题的拆解思路，还有那些我踩坑时最希望有人提前告诉我的一句话。

有人问得浅，有人答得深，没人觉得谁的问题蠢。因为这个阶段，大家都走过。

**如果你现在也觉得一个人学习大模型应用有点难，可以来看看。不为了别的，就为了下次凌晨卡住的时候或想实现token自由，知道去哪儿问一句。**

👉 [一起聊聊技术](https://share.note.youdao.com/s/3Hc9ju2)

**顺手给个 Star 吧。让我知道，这条路上不止我一个人。** 🚀