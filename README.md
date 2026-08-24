# smdoctor —— AI智能医生 项目架构与知识点分析

> 本项目是一个面向医疗领域的智能问答/辅助诊断系统，整体采用**离线模型训练 + 在线服务部署**的架构，涵盖命名实体识别、实体审核、知识图谱构建、语义匹配、多轮对话管理等核心模块。

---

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

- **CRF（条件随机场）**：通过转移矩阵建模标签之间的依赖关系，避免非法标签序列。
- **前向算法（Forward Algorithm）**：计算所有可能路径的得分（Log-Sum-Exp 技巧保证数值稳定性）。
- **维特比解码（Viterbi Decode）**：推理时寻找全局最优标签序列。
- **滑动窗口预测**：长文本以 `sentence_length` 切分，相邻窗口间设置 `offset` 重叠，避免边界截断导致实体丢失。

---

### 2.2 实体审核模型 — `review_model/`

**目标**：对 NER 抽取出的实体进行**二分类审核**，过滤错误抽取，提升下游知识图谱质量。

#### 核心方案：BERT + RNN

| 文件 | 职责 |
|------|------|
| `bert_chinese_encode.py` | 通过 `torch.hub` 加载 `bert-base-chinese`，将文本编码为 768 维向量序列 |
| `RNN_MODEL.py` | 自定义简单 RNN（`i2h` + `i2o` + `LogSoftmax`） |
| `train.py` | 读取 CSV 训练数据，BERT 编码后输入 RNN，使用 `NLLLoss`，**手动梯度更新**参数 |
| `predict.py` | 加载 `BERT_RNN.pth`，批量预测并输出审核通过的实体文件 |
| `bilstm.py` | 独立的 BiLSTM 实现（备用/对比实验） |

#### 关键知识点

- **预训练模型微调思路**：BERT 负责提取高质量语义特征，轻量 RNN/全连接层负责下游分类。
- **RNN 的自定义实现**：显式拼接 `X(t)` 与 `h(t-1)`，分别经过线性层得到新的隐状态和输出。
- **手动参数更新**：`p.data.add_(-lr, p.grad.data)`，区别于 `optimizer.step()` 的底层演示写法。

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

- **MERGE vs CREATE**：使用 `MERGE` 防止重复节点。
- **索引优化**：为 `Disease.name` 和 `Symptom.name` 创建索引，加速后续 `MATCH ... WHERE ...` 查询。
- **Cypher 查询语言**：图数据库的声明式查询语法。

---

## 三、在线层：online

### 3.1 句子相关性服务 — `bert_server/`

**目标**：作为独立微服务，判断用户**当前输入**与**上一轮输入**之间的语义相关性，支撑多轮对话决策。

#### 核心方案：BERT 双句编码 + 全连接分类

| 文件 | 职责 |
|------|------|
| `bert_chinese_encode.py` | 对 `text_1` / `text_2` 进行 BERT 编码，含 `segment_ids`、截断/填充（`max_len=10`） |
| `finetuning_net.py` | 微调网络 `Net`：输入展平后经过 `Dropout → FC(8) → ReLU → Dropout → FC(2)` |
| `train.py` | 训练脚本，交叉熵损失 `CrossEntropyLoss`，`SGD` 优化器，保存 Loss / Acc 曲线 |
| `app.py` | Flask 微服务，暴露 `GET /v1/recognition/?text1=...&text2=...`，返回 `0/1` |

#### 关键知识点

- **句子对建模（Sentence Pair Modeling）**：利用 BERT 的 `token_type_ids`（segment embedding）区分两句话。
- **文本截断与填充（Padding & Truncation）**：保证输入张量形状固定，适配神经网络批量计算。
- **微服务拆分**：将语义匹配能力独立为一个服务，便于主服务按需调用、独立扩缩容。

---

### 3.2 主逻辑/对话服务 — `main_server/`

**目标**：接收用户请求，管理对话状态，调度各模型与数据库，生成回复。

#### 核心方案：Flask + Redis 会话管理 + Neo4j 查询 + 规则模板 + 百度 UNIT 兜底

| 文件 | 职责 |
|------|------|
| `app.py` | 主服务入口，定义 `Handler` 类处理首句/非首句逻辑，暴露 `POST /v1/main_serve/` |
| `config.py` | 集中配置：Redis、Neo4j、bert_server 地址、超时时间、规则模板路径、会话过期时间等 |
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

- **多轮对话管理（Dialogue State Tracking）**：利用 Redis 的 Hash 结构存储用户维度上下文（`previous_d`、`previous`），并设置过期时间实现自动清理。
- **服务熔断/降级**：bert_server 超时或返回空时，自动降级到百度 UNIT 对话接口，保证可用性。
- **规则模板回复**：将查询结果填入预定义模板（如 `"根据您的描述，可能患有：%s"`），兼顾可控与自然。
- **差集回复策略**：已回复过的疾病不再重复输出，仅将**新增**疾病返回给用户，提升体验。

---

## 四、核心技术栈

| 类别 | 技术/框架 | 用途 |
|------|-----------|------|
| 深度学习框架 | PyTorch | 模型定义、训练、推理 |
| 预训练模型 | bert-base-chinese (HuggingFace) | 中文文本语义编码 |
| 图数据库 | Neo4j (neo4j-driver) | 医疗知识图谱存储与查询 |
| 缓存/会话 | Redis (redis-py) | 用户对话状态管理 |
| Web 服务 | Flask | 在线 API 服务化 |
| 数据科学 | NumPy, Pandas, scikit-learn, Matplotlib | 数据处理、可视化 |
| 外部 AI | 百度 UNIT | 兜底对话/闲聊接口 |
| 部署工具 | gunicorn / waitress | WSGI HTTP Server |

---

## 五、核心知识点总结

### 5.1 自然语言处理（NLP）

1. **命名实体识别（NER）**
   - BIO 标注规范
   - BiLSTM-CRF：发射分数（LSTM 输出）+ 转移分数（CRF 层）
   - 维特比解码求最优路径
   - 实体级别评估（Precision / Recall / F1）

2. **预训练语言模型应用**
   - BERT 中文编码：`tokenizer.encode`、`last_hidden_state`
   - 句子对输入：`token_type_ids` / `segment_ids`
   - 微调策略：冻结/半冻结 BERT + 上层任务网络

3. **文本分类与审核**
   - 二分类问题建模
   - `NLLLoss` + `LogSoftmax` vs `CrossEntropyLoss`
   - 自定义 RNN  Cell 的实现细节

### 5.2 知识图谱

1. **图数据建模**：节点（Entity）、关系（Relation）、属性（Property）
2. **Cypher 查询**：`MATCH`、`MERGE`、`CREATE INDEX`
3. **图谱与对话结合**：症状→疾病的多跳/单跳推理

### 5.3 对话系统架构

1. **多轮对话状态管理**：Redis Hash + TTL
2. **语义相关性判断**：独立微服务解耦意图/语义层
3. **兜底与降级**：外部机器人 API 作为异常/闲聊场景的安全网
4. **规则与模型结合**：结构化查询 + 模板生成，保证医疗领域回复的严谨性

### 5.4 工程化与服务化

1. **数据工程**
   - 字符级数字化编码、Padding、滑动窗口
   - `.npz` 高效存储大规模训练数据
   - PyTorch `DataLoader` + `Dataset` 封装

2. **服务拆分**
   - `main_server`：业务编排、状态管理
   - `bert_server`：语义计算、模型推理
   - 服务间通过 HTTP 通信，配置化地址与超时

3. **模型持久化**
   - `torch.save(state_dict)` / `torch.load(state_dict)`
   - 训练曲线可视化（Matplotlib）

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

## 七、总结

`smdoctor` 项目完整展示了从**数据预处理 → 模型训练（NER + 审核 + 语义匹配） → 知识图谱构建 → 在线服务化 → 多轮对话管理**的全链路 AI 医疗应用开发流程。其架构设计兼顾了算法深度与工程落地，适合作为医疗 NLP、对话系统、知识图谱入门的综合实践案例。
扫码获取全部项目文档![img.png](img.png)