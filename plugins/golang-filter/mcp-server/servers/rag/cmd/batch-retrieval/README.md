# Batch Retrieval Tool

批量检索工具，用于对查询数据进行批量检索并保存结果，类似于 Python 版本的 `simple_retrieval.py`。

## 使用方法

### 1. 设置环境变量

```bash
export OPENAI_API_KEY=your-api-key
export OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
export MILVUS_HOST=localhost                       # 可选，默认 localhost
export MILVUS_PORT=19530                          # 可选，默认 19530
export MILVUS_DATABASE=default                    # 可选，默认 default
export MILVUS_COLLECTION=rag                      # 可选，默认 rag
```

### 2. 运行程序

#### 基本用法（不使用 rerank）

```bash
cd plugins/golang-filter/mcp-server/servers/rag
go run cmd/batch-retrieval/main.go \
  -input dataset/MultiHopRAG.json \
  -output output/retrieval_test.json \
  -topk 10
```

#### 使用 rerank

```bash
go run cmd/batch-retrieval/main.go \
  -input dataset/MultiHopRAG.json \
  -output output/retrieval_test_rerank.json \
  -rerank \
  -topk 10 \
  -rerank_topk 20 \
  -reranker_url http://localhost:8000/v1
```

### 3. 编译为可执行文件

```bash
go build -o batch-retrieval cmd/batch-retrieval/main.go
./batch-retrieval -input dataset/MultiHopRAG.json -output output/results.json
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-input` | `dataset/MultiHopRAG.json` | 输入 JSON 文件路径 |
| `-output` | `output/retrieval_test.json` | 输出 JSON 文件路径 |
| `-topk` | `10` | 返回的 top K 结果数量 |
| `-threshold` | `0.0` | 分数阈值 |
| `-rerank` | `false` | 是否启用 rerank |
| `-rerank_topk` | `20` | rerank 前检索的候选数量 |
| `-reranker_url` | `http://localhost:8000/v1` | Reranker 服务地址 |
| `-reranker_threshold` | `0.0` | Reranker 分数阈值 |

## 查看帮助

```bash
go run cmd/batch-retrieval/main.go -h
```

## 输出格式

输出文件为 JSON 格式，包含以下字段：

```json
[
  {
    "query": "查询文本",
    "answer": "标准答案",
    "question_type": "问题类型",
    "retrieval_list": [
      {
        "text": "检索到的文档内容",
        "score": 0.95
      }
    ],
    "gold_list": [
      {
        "title": "标准答案文档标题",
        "source": "来源",
        ...
      }
    ]
  }
]
```

## 注意事项

1. 确保已设置 `OPENAI_API_KEY` 环境变量
2. 如果启用 rerank，需要确保 reranker 服务正在运行
3. 输出目录会自动创建（如果不存在）
4. 程序会每处理 100 个查询输出一次进度



