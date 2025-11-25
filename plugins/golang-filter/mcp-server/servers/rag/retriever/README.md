# Retriever

Retriever 实现了类似 Python `simple_retrieval.py` 中的检索功能，支持可选的 rerank（重排序）功能。

## 功能特性

- 调用 `RAGClient.SearchChunks` 进行向量检索
- 支持可选的 rerank 功能，提升检索质量
- 与 Python 版本的 retriever 行为一致

## 使用方法

### 1. 基本配置

在配置文件中添加 reranker 配置：

```yaml
rag:
  top_k: 10
  threshold: 0.5
  rerank: true          # 启用 rerank
  rerank_top_k: 20      # rerank 前检索的候选数量（默认 20）

reranker:
  enabled: true
  base_url: "http://localhost:8000"  # Python reranker 服务地址
  threshold: 0.0                      # rerank 分数阈值
```

### 2. 创建 Retriever

```go
import (
    "context"
    "github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag"
    "github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/config"
    "github.com/alibaba/higress/plugins/golang-filter/mcp-server/servers/rag/retriever"
)

// 创建配置
cfg := &config.Config{
    RAG: config.RAGConfig{
        TopK:      10,
        Threshold: 0.5,
        Rerank:    true,
        RerankTopK: 20,
    },
    Reranker: config.RerankerConfig{
        Enabled: true,
        BaseURL: "http://localhost:8000",
    },
    // ... 其他配置
}

// 创建 RAG Client
ragClient, err := rag.NewRAGClient(cfg)
if err != nil {
    // 处理错误
}

// 创建 Retriever
retriever, err := retriever.NewRetriever(ragClient, cfg)
if err != nil {
    // 处理错误
}
```

### 3. 使用 Retriever 检索

#### 使用默认配置

```go
query := "What is the capital of France?"
results, err := retriever.RetrieveWithDefaults(context.Background(), query)
if err != nil {
    // 处理错误
}

for _, result := range results {
    fmt.Printf("Score: %.4f, Content: %s\n", result.Score, result.Document.Content)
}
```

#### 使用自定义选项

```go
options := &retriever.RetrieverOptions{
    TopK:      5,
    Threshold: 0.6,
    Rerank:    false,  // 禁用 rerank
}

results, err := retriever.Retrieve(context.Background(), query, options)
```

## 工作原理

### 启用 Rerank 时

1. 首先调用 `SearchChunks` 检索 `rerank_top_k`（默认 20）个候选文档
2. 调用 reranker API 对这些候选文档进行重排序
3. 返回重排序后的前 `top_k` 个结果

### 未启用 Rerank 时

1. 直接调用 `SearchChunks` 检索 `top_k` 个结果
2. 返回检索结果

## 与 Python 版本的对应关系

| Python (`simple_retrieval.py`) | Go (`retriever.go`) |
|-------------------------------|---------------------|
| `index.as_retriever(similarity_top_k=20).retrieve(query)` | `SearchChunks(query, rerankTopK, threshold)` |
| `rerank_postprocessors.postprocess_nodes(...)` | `RerankerClient.RerankSearchResults(...)` |
| `index.as_retriever(similarity_top_k=top_k).retrieve(query)` | `SearchChunks(query, topK, threshold)` |

## 配置说明

### RAGConfig

- `rerank`: 是否启用 rerank（默认 false）
- `rerank_top_k`: rerank 前检索的候选数量（默认 20）

### RerankerConfig

- `enabled`: 是否启用 reranker 服务（默认 false）
- `base_url`: reranker 服务的 API 地址（必需）
- `api_key`: 可选的 API 密钥
- `threshold`: rerank 分数阈值（默认 0.0）

## 注意事项

1. 启用 rerank 需要配置并启动 Python reranker 服务
2. `rerank_top_k` 应该大于等于 `top_k`，以确保有足够的候选进行重排序
3. 如果 reranker 服务不可用，retriever 创建会失败（如果启用了 reranker）

