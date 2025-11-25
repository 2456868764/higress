# vllm Reranker Service

使用 vllm 启动 Qwen3-Reranker-0.6B 服务：
```
ssh -L 8090:127.0.0.1:8090 -p 38376 root@connect.nmb1.seetacloud.com
```


```bash
export VLLM_USE_V1=0
# 运行修正后的命令
vllm serve /root/autodl-tmp/models/Qwen/Qwen3-Reranker-0.6B \
  --task score \
  --trust-remote-code \
  --hf_overrides '{"architectures": ["Qwen2ForSequenceClassification"], "classifier_from_token": ["no", "yes"]}'

python3 -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/Qwen/Qwen3-Reranker-0.6B \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype auto \
    --trust-remote-code \
    --served-model-name reranker \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.6 \
    --task score \
    --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'

```

**注意**：如果使用本地模型路径，请替换为实际路径，例如：
```bash
--model /path/to/models/Qwen/Qwen3-Reranker-0.6B
```

## 注意事项

1. **反斜杠格式**：确保反斜杠 `\` 后面没有空格，否则 bash 会将后续参数当作独立命令执行
2. **模型路径**：根据实际情况修改 `--model` 参数中的模型路径
3. **端口**：默认使用 8080 端口，确保端口未被占用
4. **GPU 内存**：根据 GPU 显存调整 `--gpu-memory-utilization` 参数（0.0-1.0）

## 常见错误

### 1. AttributeError: 'dict' object has no attribute 'model_type'

这个错误通常是因为：
- 模型配置文件格式不正确
- 使用了不兼容的模型版本
- vllm 版本与模型不兼容

**解决方法**：
- 检查模型路径是否正确
- 确认模型支持 vllm 的 `--task score` 模式
- 尝试更新 vllm 版本：`pip install --upgrade vllm`
- 检查模型目录下的 `config.json` 文件是否包含 `model_type` 字段

### 2. Bash 命令错误

如果看到类似 `bash: --served-model-name: command not found` 的错误，说明反斜杠后面有空格。

**解决方法**：
- 确保每行末尾的反斜杠 `\` 后面没有空格
- 或者将整个命令写在一行（不使用反斜杠换行）

**正确格式示例**：
```bash
# 方式1：每行末尾反斜杠后无空格
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Reranker-0.6B \
    --host 0.0.0.0 \
    --port 8080 \
    --dtype auto \
    --trust-remote-code \
    --served-model-name reranker \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.6 \
    --task score \
    --disable-log-requests

# 方式2：单行命令（推荐）
python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Reranker-0.6B --host 0.0.0.0 --port 8080 --dtype auto --trust-remote-code --served-model-name reranker --enable-prefix-caching --gpu-memory-utilization 0.6 --task score --disable-log-requests
```

## Qwen3-Reranker-0.6B 特殊说明

Qwen3-Reranker-0.6B 是一个基于 CausalLM 的 reranker 模型，与传统的 SequenceClassification 模型不同：

1. **模型要求**：需要 vllm >= 0.8.5
2. **任务模式**：使用 `--task score` 参数
3. **内存需求**：相比 bge-reranker-large，Qwen3-Reranker-0.6B 模型更小，内存占用更少
4. **性能**：支持更长的输入序列（最大长度 4096）

### 3. 替代方案：使用 Python Reranker Service

如果 vllm 有问题，可以使用项目自带的 Python reranker 服务：

```bash
cd plugins/golang-filter/mcp-server/servers/rag/python
python service.py
```

该服务会自动从环境变量读取配置，支持本地模型加载。