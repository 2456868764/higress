# Index.py 使用说明

## 功能概述

`index.py` 用于将 `dataset/corpus.json` 中的文档索引到 Milvus 向量数据库中。

## 主要功能

1. **读取 corpus.json**: 从 `dataset/corpus.json` 加载文档数据
2. **文本分割**: 使用 `RecursiveCharacterTextSplitter` 将每个文档的 `body` 字段分割成多个 chunks
3. **创建向量**: 使用 embedding 模型为每个 chunk 生成向量
4. **插入 Milvus**: 将 chunks 及其向量插入到 Milvus 数据库中

## 使用方法

### 基本用法

```bash
cd plugins/golang-filter/mcp-server/servers/rag/python
python index.py
```

### 自定义参数

```bash
python index.py \
    --corpus_file dataset/corpus.json \
    --collection_name my_corpus \
    --chunk_size 500 \
    --chunk_overlap 50 \
    --batch_size 100
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--corpus_file` | string | `dataset/corpus.json` | corpus.json 文件路径 |
| `--collection_name` | string | `corpus_collection` | Milvus 集合名称 |
| `--chunk_size` | int | `500` | 每个文本块的大小（字符数） |
| `--chunk_overlap` | int | `50` | 块之间的重叠大小 |
| `--batch_size` | int | `100` | 每批处理的文档数量 |

## 数据流程

```
corpus.json
    ↓
读取 JSON 数据
    ↓
对每个文档:
    ├─ 提取 body 字段
    ├─ 使用 RecursiveCharacterTextSplitter 分割
    └─ 创建 Document 对象（带元数据）
    ↓
创建 CorpusFile 对象
    ↓
调用 MilvusHybridVectorStore.add_doc()
    ├─ 生成向量（embedding）
    ├─ 插入 Milvus
    └─ 刷新集合
```

## 文档结构

### corpus.json 格式

```json
[
    {
        "title": "文档标题",
        "author": "作者",
        "source": "来源",
        "published_at": "2023-11-27T08:45:59+00:00",
        "category": "类别",
        "url": "https://example.com/article",
        "body": "文档正文内容..."
    }
]
```

### 生成的 Document 结构

每个 chunk 会创建一个 `Document` 对象：

```python
Document(
    page_content="chunk 文本内容",
    metadata={
        'id': 'uuid',
        'title': '文档标题',
        'source': '来源',
        'url': 'URL',
        'author': '作者',
        'published_at': '发布时间',
        'category': '类别',
        'chunk_index': 0,  # chunk 在文档中的索引
        'source': 'md5(url)',  # 用于标识文档
        'filename': 'url'
    }
)
```

## 配置要求

### 环境变量

确保设置了以下环境变量（如果需要）：

```bash
# Embedding 模型配置
export EMBEDDING_MODEL_PATH="Qwen/Qwen3-Embedding-0.6B"
export EMBEDDING_MODEL_ENGINE="qwen"
export EMBEDDING_DIM=1024
export EMBEDDING_DEVICE="auto"

# Vector Store 配置
export VECTOR_STORE_HOST=""  # 空字符串使用 Milvus Lite
export VECTOR_STORE_PORT=""
export VECTOR_STORE_NAME="default"
export KB_ROOT_PATH="./knowledge_base"
```

### 依赖安装

确保安装了必要的依赖：

```bash
pip install langchain-core langchain-text-splitters pymilvus tqdm
```

## 示例输出

```
INFO: Loading corpus from: dataset/corpus.json
INFO: Loaded 1000 documents from corpus.json
INFO: Initializing embedding model...
INFO: Embedding model initialized
INFO: Initializing Milvus vector store with collection: corpus_collection
INFO: Milvus vector store initialized
INFO: Starting to index 1000 documents...
INFO: Chunk size: 500, Chunk overlap: 50
Processing batches: 100%|████████████| 10/10 [05:23<00:00, 32.35s/batch]
INFO: Processed batch 1: 1-100 / 1000
...
INFO: Indexing completed!
INFO: Total documents processed: 1000
INFO: Total chunks created: 5234
INFO: Collection name: corpus_collection
```

## 注意事项

1. **内存使用**: 处理大量文档时可能占用较多内存，建议使用合适的 `batch_size`
2. **向量维度**: 确保 `EMBEDDING_DIM` 与 embedding 模型的实际维度匹配
3. **Milvus 连接**: 
   - 如果 `VECTOR_STORE_HOST` 为空，将使用 Milvus Lite（本地文件存储）
   - 如果设置了 host 和 port，将连接到远程 Milvus 服务器
4. **重复索引**: 如果文档已存在，`add_doc` 会先删除旧数据再插入新数据
5. **文本清理**: 文本会自动清理（统一换行、去除多余空白等）

## 故障排查

### 问题：导入错误

**错误**: `ModuleNotFoundError: No module named 'milvus_hybrid'`

**解决**: 确保在正确的目录下运行，或使用 `python -m` 方式：

```bash
cd plugins/golang-filter/mcp-server/servers/rag/python
python -m index
```

### 问题：Milvus 连接失败

**错误**: `Connection refused` 或 `Cannot connect to Milvus`

**解决**: 
- 检查 Milvus 服务是否运行
- 检查 `VECTOR_STORE_HOST` 和 `VECTOR_STORE_PORT` 配置
- 如果使用 Milvus Lite，确保 `KB_ROOT_PATH` 目录可写

### 问题：内存不足

**错误**: `MemoryError` 或进程被杀死

**解决**: 
- 减小 `batch_size` 参数
- 减小 `chunk_size` 参数
- 使用更小的 embedding 模型

## 性能优化建议

1. **批量处理**: 使用合适的 `batch_size`（建议 50-200）
2. **并行处理**: 可以修改代码支持多进程处理（需要处理线程安全）
3. **增量索引**: 对于大型数据集，可以考虑增量索引策略
4. **GPU 加速**: 如果使用 GPU，确保 `EMBEDDING_DEVICE="cuda"`

## 相关文件

- `milvus_hybrid.py`: Milvus 向量存储实现
- `service/local_embedding.py`: 本地 embedding 模型
- `config.py`: 配置文件
- `base.py`: 基础类定义

