# DeepSearch vs ChainOfRAG 功能对比

## 核心设计理念

### DeepSearch
- **目标**: 适合处理**通用和简单的查询**，如给定主题后撰写报告、调查或文章
- **策略**: **广度优先** - 一次性分解为多个子查询，并行搜索，然后通过gap queries补充信息

### ChainOfRAG
- **目标**: 适合处理**具体的多跳事实查询**
- **策略**: **深度优先** - 逐步生成后续查询，每次迭代深入一个方向

## 主要功能差异

### 1. 查询分解策略

| 特性 | DeepSearch | ChainOfRAG |
|------|-----------|------------|
| **初始分解** | 一次性生成**最多4个子查询** | 不预先分解，从原始查询开始 |
| **后续查询** | 每次迭代生成**最多3个gap queries** | 每次迭代生成**1个follow-up query** |
| **查询类型** | 子查询 + gap queries（补充性查询） | Follow-up queries（递进式查询） |

**DeepSearch示例**:
```
原始查询: "Explain deep learning"
子查询: ["What is deep learning?", "What is the difference between deep learning and machine learning?", "What is the history of deep learning?"]
Gap queries: ["What are the applications of deep learning?", "What are the challenges?"]
```

**ChainOfRAG示例**:
```
原始查询: "Who invented the transformer architecture?"
迭代1: "What is transformer architecture?"
迭代2: "Who are the authors of the Attention Is All You Need paper?"
迭代3: "What are the key contributions of transformer?"
```

### 2. Rerank机制

| 特性 | DeepSearch | ChainOfRAG |
|------|-----------|------------|
| **Rerank时机** | **检索时立即rerank** - 对每个检索到的chunk使用LLM判断 | **检索后rerank** - 先检索，生成中间答案，再筛选文档 |
| **Rerank方式** | LLM判断chunk是否有助于回答**任何子查询**（YES/NO） | LLM选择**支持中间答案的文档**（索引列表） |
| **Token消耗** | **高** - 每个chunk都需要一次LLM调用 | **中等** - 只在生成中间答案后筛选一次 |

**DeepSearch Rerank流程**:
```
检索 → 对每个chunk调用LLM → 只保留YES的chunks
```

**ChainOfRAG Rerank流程**:
```
检索 → 生成中间答案 → 筛选支持该答案的文档
```

### 3. 迭代搜索流程

#### DeepSearch流程
```
1. 生成子查询（最多4个）
2. 对每个子查询并行搜索 + LLM rerank
3. 迭代（最多maxIter次）:
   - 生成gap queries（最多3个）
   - 对每个gap query搜索 + rerank
   - 如果gap queries为空则停止
4. 汇总所有结果
```

#### ChainOfRAG流程
```
1. 从原始查询开始
2. 迭代（最多maxIter次）:
   - 生成1个follow-up query
   - 检索文档
   - 生成中间答案
   - 筛选支持的文档
   - 如果earlyStopping且信息足够则停止
3. 汇总所有结果
```

### 4. 文档筛选机制

| 特性 | DeepSearch | ChainOfRAG |
|------|-----------|------------|
| **筛选时机** | 检索时（rerank阶段） | 检索后（基于中间答案） |
| **筛选标准** | 是否有助于回答**任何查询** | 是否**支持中间答案** |
| **筛选结果** | 保留/丢弃整个chunk | 返回文档索引列表 |

**DeepSearch**:
- 使用 `rerankPrompt`: "Is the chunk helpful in answering any of the questions?"
- 返回 YES/NO，只保留YES的chunks

**ChainOfRAG**:
- 使用 `getSupportedDocsPrompt`: "select the ones that support the Q-A pair"
- 返回文档索引列表，只保留支持的文档

### 5. 最终答案生成

| 特性 | DeepSearch | ChainOfRAG |
|------|-----------|------------|
| **Prompt类型** | `summaryPrompt` - 强调**总结和报告** | `finalAnswerPrompt` - 强调**结合中间信息** |
| **输入信息** | 原始查询 + 所有子查询 + 所有chunks | 原始查询 + 中间查询和答案 + 所有chunks |
| **输出风格** | 报告、调查、文章风格 | 直接答案风格 |

**DeepSearch Prompt**:
```
"You are a AI content analysis expert, good at summarizing content. 
Please summarize a specific and detailed answer or report..."
```

**ChainOfRAG Prompt**:
```
"Given the following intermediate queries and answers, generate a final answer 
by combining relevant information..."
```

### 6. 停止条件

| 特性 | DeepSearch | ChainOfRAG |
|------|-----------|------------|
| **停止条件1** | Gap queries为空 | Early stopping + 信息足够 |
| **停止条件2** | 达到maxIter | 达到maxIter |
| **Early Stopping** | ❌ 不支持 | ✅ 支持（可选） |

### 7. Token消耗对比

| 操作 | DeepSearch | ChainOfRAG |
|------|-----------|------------|
| **查询分解** | 1次LLM调用（生成子查询） | 每次迭代1次（生成follow-up） |
| **检索** | N次（N=子查询数+gap queries数） | N次（N=迭代次数） |
| **Rerank** | **M次**（M=检索到的chunk总数） | **N次**（N=迭代次数，每次筛选一次） |
| **总结** | 1次（生成最终答案） | 1次（生成最终答案） |

**DeepSearch**: Token消耗 = 1 + N + M + 1（M通常很大）
**ChainOfRAG**: Token消耗 = N + N + 1 = 2N + 1（N通常较小）

### 8. 适用场景

#### DeepSearch 适合:
- ✅ **撰写报告、调查、文章**
- ✅ **需要全面覆盖多个方面的问题**
- ✅ **探索性查询**（"Tell me about X"）
- ✅ **需要高质量文档筛选**（通过LLM rerank）

#### ChainOfRAG 适合:
- ✅ **多跳事实查询**（"Who invented X?" → "What paper?" → "Who are the authors?"）
- ✅ **需要逐步深入的问题**
- ✅ **具体的事实查询**
- ✅ **需要早期停止优化**（early stopping）

## 代码结构对比

### DeepSearch关键方法
```go
_generateSubQueries()        // 生成子查询
_searchChunksFromVectorDB()  // 搜索 + LLM rerank
_generateGapQueries()        // 生成gap queries
_formatChunkTexts()          // 格式化chunks
```

### ChainOfRAG关键方法
```go
reflectGetSubquery()         // 生成follow-up query
retrieveAndAnswer()          // 检索 + 生成中间答案
getSupportedDocs()           // 筛选支持的文档
checkHasEnoughInfo()         // 检查信息是否足够
formatRetrievedResults()     // 格式化结果
```

## 总结

| 维度 | DeepSearch | ChainOfRAG |
|------|-----------|------------|
| **查询策略** | 广度优先，多查询并行 | 深度优先，单查询递进 |
| **Rerank方式** | 检索时LLM rerank（高精度） | 检索后基于答案筛选 |
| **Token效率** | 较低（每个chunk都rerank） | 较高（只筛选一次） |
| **适用场景** | 报告、调查、全面探索 | 多跳事实查询、逐步深入 |
| **复杂度** | 高（多查询管理） | 中（单查询迭代） |

## 选择建议

- **选择 DeepSearch** 当:
  - 需要撰写报告或文章
  - 需要全面覆盖多个方面
  - 对文档质量要求高（愿意付出更多token）
  
- **选择 ChainOfRAG** 当:
  - 处理多跳事实查询
  - 需要逐步深入探索
  - 对token消耗敏感
  - 需要early stopping优化

