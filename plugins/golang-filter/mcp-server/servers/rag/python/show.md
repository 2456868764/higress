1. 生成数据

```
cd /Users/jun/GolandProjects/higress/higress/plugins/golang-filter/mcp-server/servers/rag
python python/index.py --help

python python/index.py \
  --corpus_file ./dataset/corpus.json \
  --collection_name corpus_collection_test_500 \
  --chunk_size 500 \
  --chunk_overlap 50
```

2. 启动本地 Embedding & Reranker 服务

```
cd /Users/jun/GolandProjects/higress/higress/plugins/golang-filter/mcp-server/servers/rag
python python/service.py

```

3. Baseline 检索生成

```
cd  /Users/jun/GolandProjects/higress/higress/plugins/golang-filter/mcp-server/servers/rag

go run cmd/batch-retrieval/main.go --help

go run cmd/batch-retrieval/main.go \
  -input=python/dataset/MultiHopRAG.json \
  -output=python/output/retrieval_default_test_500.json \
  -collection=corpus_collection_500 \
  -agent=default \
  -topk=10 \
  -threshold=0.0 \
  -rerank=false \
  -hybrid_search=false \
  -workers=1 

```

4. Baseline 生成答案
```
cd plugins/golang-filter/mcp-server/servers/rag/python
python python/qa_llm.py --help
python python/qa_llm.py \
  --input ./python/output/retrieval_default_500.json \
  --output ./python/qa_output/default_test_500.json \
  --model qwen-plus \
  --temperature 0.3 \
  --max-tokens 512 \
  --max-workers 10

```

5. Baseline 检索评估
```
cd plugins/golang-filter/mcp-server/servers/rag
python/retrieval_evaluate.py --help
python python/retrieval_evaluate.py \
  --file ./python/output/retrieval_default_500.json

```

6. Baseline QA评估

```
python python/qa_evaluate.py \
  --qa-output ./python/qa_output/default_500.json \
  --dataset ./python/dataset/MultiHopRAG.json
```


7. Chainof RAG 演示

```
go run cmd/batch-retrieval/main.go \
  -input=python/dataset/MultiHopRAG.json \
  -output=python/output/retrieval_default_test_500.json \
  -collection=corpus_collection_500 \
  -agent=chain_of_rag \
  -topk=10 \
  -threshold=0.0 \
  -rerank=false \
  -hybrid_search=false \
  -workers=1 
```

8. RAG Router 演示

```
go run cmd/batch-retrieval/main.go \
  -input=python/dataset/MultiHopRAG.json \
  -output=python/output/retrieval_default_test_500.json \
  -collection=corpus_collection_500 \
  -agent=router \
  -topk=10 \
  -threshold=0.0 \
  -rerank=false \
  -hybrid_search=false \
  -workers=1 
```