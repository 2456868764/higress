# RAG Service API - cURL Test Examples

本文档提供了 RAG Service API 的完整 cURL 测试示例，包括 Embedding 和 Rerank 接口的各种使用场景。

## 基础信息

- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`

---

## 1. Health Check

检查服务健康状态：

```bash
curl -X GET "http://localhost:8000/health" \
  -H "Content-Type: application/json"
```

**响应示例**:
```json
{
  "status": "healthy",
  "embedding_service": true,
  "reranker_service": true
}
```

---

## 2. Root Endpoint

获取 API 基本信息：

```bash
curl -X GET "http://localhost:8000/" \
  -H "Content-Type: application/json"
```

---

## 3. Embedding API

### 3.1 单个文本嵌入

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key-here" \
  -d '{
    "input": "What is artificial intelligence?",
    "model": "Qwen3-Embedding-0.6B"
  }'
```

**响应示例**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5,
    "processing_time": 0.123
  }
}
```

### 3.2 多个文本嵌入

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "What is machine learning?",
      "How does deep learning work?",
      "Explain neural networks"
    ],
    "model": "Qwen3-Embedding-0.6B"
  }'
```

**响应示例**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, -0.2, 0.3, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.4, -0.5, 0.6, ...],
      "index": 1
    },
    {
      "object": "embedding",
      "embedding": [0.7, -0.8, 0.9, ...],
      "index": 2
    }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 15,
    "processing_time": 0.234
  }
}
```

### 3.3 完整参数示例

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Natural language processing is a branch of artificial intelligence.",
    "model": "Qwen3-Embedding-0.6B",
    "encoding_format": "float",
    "dimensions": 1024,
    "user": "test-user-123"
  }'
```

### 3.4 长文本嵌入

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Retrieval-Augmented Generation (RAG) is a technique that enhances the accuracy and reliability of generative AI models with facts fetched from external knowledge bases. By grounding LLMs on the most accurate, up-to-date information and giving users insight into LLMs generative process, RAG helps models produce more reliable outputs."
  }'
```

---

## 4. Rerank API

### 4.1 基础重排序

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "documents": [
      "Artificial intelligence is the simulation of human intelligence by machines.",
      "Machine learning is a subset of AI that enables systems to learn from data.",
      "The weather today is sunny and warm.",
      "Deep learning uses neural networks to process complex patterns."
    ]
  }'
```

**响应示例**:
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": "Artificial intelligence is the simulation of human intelligence by machines."
    },
    {
      "index": 1,
      "relevance_score": 0.87,
      "document": "Machine learning is a subset of AI that enables systems to learn from data."
    },
    {
      "index": 3,
      "relevance_score": 0.72,
      "document": "Deep learning uses neural networks to process complex patterns."
    },
    {
      "index": 2,
      "relevance_score": 0.15,
      "document": "The weather today is sunny and warm."
    }
  ],
  "model": "Qwen3-Reranker-0.6B",
  "usage": {
    "query_tokens": 5,
    "document_count": 4,
    "results_count": 4,
    "processing_time": 0.456
  }
}
```

### 4.2 指定 Top N

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does neural network training work?",
    "documents": [
      "Neural networks are trained using backpropagation algorithm.",
      "Training involves forward pass and backward pass.",
      "The training data is split into batches.",
      "Gradient descent optimizes the network weights.",
      "Overfitting can occur if training goes on too long.",
      "Regularization techniques help prevent overfitting."
    ],
    "top_n": 3
  }'
```

### 4.3 使用阈值过滤

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is transformer architecture?",
    "documents": [
      "Transformers use self-attention mechanism.",
      "BERT and GPT are transformer-based models.",
      "The transformer architecture revolutionized NLP.",
      "Cooking recipes often use fresh ingredients.",
      "Attention mechanism allows models to focus on relevant parts."
    ],
    "top_n": 5,
    "threshold": 0.5
  }'
```

### 4.4 不返回文档内容

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain gradient descent",
    "documents": [
      "Gradient descent is an optimization algorithm.",
      "It minimizes the loss function iteratively.",
      "Learning rate controls the step size.",
      "Stochastic gradient descent uses random samples."
    ],
    "return_documents": false
  }'
```

**响应示例**:
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.92
    },
    {
      "index": 1,
      "relevance_score": 0.85
    },
    {
      "index": 2,
      "relevance_score": 0.78
    },
    {
      "index": 3,
      "relevance_score": 0.71
    }
  ],
  "model": "Qwen3-Reranker-0.6B",
  "usage": {
    "query_tokens": 3,
    "document_count": 4,
    "results_count": 4,
    "processing_time": 0.321
  }
}
```

### 4.5 实际应用示例

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to implement RAG system?",
    "documents": [
      "RAG combines retrieval and generation for better AI responses.",
      "First, retrieve relevant documents from a vector database.",
      "Then, use the retrieved context to generate answers.",
      "Vector embeddings enable semantic search.",
      "Chunking documents improves retrieval accuracy.",
      "The system needs an embedding model and a language model."
    ],
    "top_n": 4,
    "threshold": 0.3,
    "return_documents": true
  }'
```

---

## 5. 错误处理示例

### 5.1 Embedding - 空输入

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ""
  }'
```

**响应**:
```json
{
  "detail": "Input cannot be empty"
}
```

### 5.2 Rerank - 空查询

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "",
    "documents": ["Document 1", "Document 2"]
  }'
```

**响应**:
```json
{
  "detail": "Query cannot be empty"
}
```

### 5.3 Rerank - 空文档列表

**请求**:
```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Test query",
    "documents": []
  }'
```

**响应**:
```json
{
  "detail": "Documents list cannot be empty"
}
```

### 5.4 服务未初始化

如果服务未正确初始化，会返回 503 错误：

**响应**:
```json
{
  "detail": "Embedding service is not initialized"
}
```

或

```json
{
  "detail": "Reranker service is not initialized"
}
```

---

## 6. 使用 jq 美化输出

如果安装了 `jq`，可以使用它来美化 JSON 输出：

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Test text"
  }' | jq '.'
```

---

## 7. 批量测试脚本

可以使用提供的 `curl_examples.sh` 脚本运行所有测试：

```bash
chmod +x curl_examples.sh
./curl_examples.sh
```

---

## 8. 参数说明

### Embedding 请求参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `input` | string \| array | 是 | 要嵌入的文本或文本数组 |
| `model` | string | 否 | 模型名称（用于兼容性） |
| `encoding_format` | string | 否 | 编码格式（默认: "float"） |
| `dimensions` | integer | 否 | 维度数量 |
| `user` | string | 否 | 用户标识符 |

### Rerank 请求参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 查询文本 |
| `documents` | array | 是 | 要重排序的文档列表 |
| `top_n` | integer | 否 | 返回的 top N 结果数量 |
| `return_documents` | boolean | 否 | 是否返回文档内容（默认: true） |
| `threshold` | float | 否 | 分数阈值，低于此值的文档将被过滤（默认: 0.0） |

---

## 9. 性能优化建议

1. **批量处理**: 对于多个文本，使用数组输入而不是多次单独请求
2. **合理设置 top_n**: 根据实际需求设置，避免返回过多不相关结果
3. **使用阈值**: 设置合适的阈值可以过滤低质量结果
4. **缓存结果**: 对于相同的查询，可以缓存 embedding 结果

---

## 10. 常见问题

### Q: 如何知道服务是否正常运行？
A: 调用 `/health` 端点检查服务状态。

### Q: 支持的最大文本长度是多少？
A: 取决于使用的模型，通常建议单个文本不超过 512-1024 tokens。

### Q: 可以同时处理多少个文档的重排序？
A: 取决于模型和硬件配置，建议单次请求不超过 100 个文档。

### Q: 如何提高重排序的准确性？
A: 
- 确保文档与查询相关
- 使用合适的阈值过滤低分结果
- 根据实际需求调整 top_n 参数

---

## 11. 联系与支持

如有问题，请查看日志文件或联系技术支持。

